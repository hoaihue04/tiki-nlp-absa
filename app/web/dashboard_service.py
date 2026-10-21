from __future__ import annotations

import re
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from app.assistant.repository import V2Repository
from app.assistant.risk import detect_risks
from app.assistant.vector_store import PgVectorOpinionStore, opinion_chunks_from_aspect_items
from app.web.constants import (
    ASPECT_DISPLAY_NAMES,
    ASPECT_FORBIDDEN_KEYWORDS,
    ASPECT_REQUIRED_KEYWORDS,
    SENTIMENT_TO_SCORE,
)
from app.web.csv_fallback import (
    _get_product_info_from_csv_fallback,
    _get_reviews_from_csv_fallback,
    _pick_quote_from_asqp,
)
from app.web.product_parser import (
    _safe_str,
    _sentiment_label_from_score,
    extract_product_id_from_url,
    split_sentences,
)
from app.web.tiki_client import fetch_tiki_product_info_live, fetch_tiki_reviews_live

ProgressCallback = Callable[[int, str], None]


@dataclass
class DashboardConfig:
    max_reviews_for_dashboard: int = 200
    phobert_batch_size: int = 8
    category_level: str = "lv3"
    confidence_threshold: float = 0.6
    reviews_per_page: int = 20
    max_review_pages: int = 15
    request_timeout: int = 10
    inter_page_sleep: Tuple[float, float] = (0.5, 1.2)


class DashboardService:
    def __init__(self):
        self.config = DashboardConfig()
        self._predictor: Optional[Any] = None
        self._latest_analysis: Dict[str, Dict[str, Any]] = {}
        self._v2_repository = V2Repository()
        self._v2_vector_store = PgVectorOpinionStore()
        if self._v2_repository.configured and not self._v2_repository.enabled:
            print(f"[V2 DB] Disabled: {self._v2_repository.status.get('error')}")

    def get_latest_analysis(self, product_id: str) -> Optional[Dict[str, Any]]:
        return self._latest_analysis.get(str(product_id).strip())

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            from app.web.phobert_predictor import RealPhoBERTPredictor

            self._predictor = RealPhoBERTPredictor(
                max_len=256,
                confidence_threshold=self.config.confidence_threshold,
            )
        return self._predictor

    def _safe_progress(self, cb: Optional[ProgressCallback], pct: int, msg: str):
        if cb:
            try:
                cb(pct, msg)
            except Exception:
                pass

    def _clean_text(self, text: str) -> str:
        text = _safe_str(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _preprocess_reviews(self, reviews: List[Dict]) -> pd.DataFrame:
        if not reviews:
            return pd.DataFrame()
        df = pd.DataFrame(reviews)
        for col in ("review_id", "title", "content"):
            if col not in df.columns:
                df[col] = ""
        if "helpful_count" not in df.columns:
            df["helpful_count"] = 0
        if "rating" not in df.columns:
            df["rating"] = 3
        df["title"] = df["title"].fillna("").astype(str)
        df["content"] = df["content"].fillna("").astype(str)
        df["full_text"] = (df["title"] + ". " + df["content"]).str.strip(". ")
        df["clean_text"] = df["full_text"].apply(self._clean_text)
        df = df[df["clean_text"].str.len() > 20].copy()
        df["helpful_count"] = pd.to_numeric(df["helpful_count"], errors="coerce").fillna(0)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3)
        return df

    def _batch_predict(
        self,
        reviews_df: pd.DataFrame,
        progress_callback: Optional[ProgressCallback],
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        predictor = self._get_predictor()
        sentence_meta: List[Dict] = []
        for _, rv in reviews_df.iterrows():
            rid = str(rv.get("review_id", ""))
            text = str(rv.get("clean_text", ""))
            helpful = int(rv.get("helpful_count", 0))
            rating = float(rv.get("rating", 3)) / 5.0
            for sent in split_sentences(text):
                if 15 <= len(sent) <= 500:
                    sentence_meta.append({
                        "review_id": rid,
                        "sentence": sent,
                        "helpful_count": helpful,
                        "rating_weight": rating,
                    })
        total = len(sentence_meta)
        print(f"[PhoBERT] Processing {total} sentences")
        self._safe_progress(progress_callback, 45, f"Đang phân tích {total} câu với PhoBERT...")
        tuples: List[Dict] = []
        review_bucket: Dict[str, List[Dict]] = {}
        batch_size = max(1, int(self.config.phobert_batch_size))
        for start in range(0, total, batch_size):
            batch_meta = sentence_meta[start:start + batch_size]
            batch_sentences = [meta["sentence"] for meta in batch_meta]
            if start == 0:
                self._safe_progress(progress_callback, 46, "Đang nạp PhoBERT vào bộ nhớ...")
            batch_preds = predictor.predict_sentences(batch_sentences)
            for meta, preds in zip(batch_meta, batch_preds):
                for p in preds:
                    raw_sentiment = p.get("sentiment", "neutral")
                    score = SENTIMENT_TO_SCORE.get(raw_sentiment, 0.0)
                    confidence = float(p.get("confidence", 0.5))
                    rating_weight = meta["rating_weight"]
                    row = {
                        "review_id": meta["review_id"],
                        "sentence": meta["sentence"],
                        "aspect": p.get("aspect", "PRODUCT#QUALITY"),
                        "sentiment": raw_sentiment,
                        "confidence": confidence,
                        "sentiment_score": score,
                        "helpful_count": meta["helpful_count"],
                        "weight": rating_weight * (1 + meta["helpful_count"] / 10) * confidence,
                    }
                    tuples.append(row)
                    review_bucket.setdefault(meta["review_id"], []).append(row)
            processed = min(start + len(batch_meta), total)
            pct = 45 + int(35 * processed / total) if total > 0 else 80
            self._safe_progress(
                progress_callback, pct,
                f"Đang phân tích câu {processed}/{total}..."
            )
        review_scores = []
        sentiment_bucket = {"positive": 0, "neutral": 0, "negative": 0}
        for rid, items in review_bucket.items():
            if not items:
                continue
            total_w = sum(item["weight"] for item in items)
            if total_w > 0:
                ws = sum(item["sentiment_score"] * item["weight"] for item in items) / total_w
            else:
                ws = sum(item["sentiment_score"] for item in items) / len(items)
            label = _sentiment_label_from_score(ws)
            sentiment_bucket[label] += 1
            review_scores.append({
                "review_id": rid,
                "score": ws,
                "label": label,
                "confidence": max(item["confidence"] for item in items),
            })
        return tuples, review_scores, sentiment_bucket

    def _pick_quote_for_aspect(
        self, tuples: List[Dict], aspect: str, sentiment: str, max_len: int = 300
    ) -> str:
        asqp_quote = _pick_quote_from_asqp(aspect, sentiment, max_len)
        if asqp_quote:
            return asqp_quote

        pool = [t for t in tuples if t.get("aspect") == aspect and t.get("sentiment") == sentiment]
        if not pool:
            return ""
        required_kws = ASPECT_REQUIRED_KEYWORDS.get(aspect, [])
        forbidden_kws = ASPECT_FORBIDDEN_KEYWORDS.get(aspect, [])

        def score_item(t):
            return (t.get("helpful_count", 0), t.get("confidence", 0), len(t.get("sentence", "")))

        tier1 = [t for t in pool
                 if required_kws and any(kw in t.get("sentence", "").lower() for kw in required_kws)
                 and not any(kw in t.get("sentence", "").lower() for kw in forbidden_kws)]
        if tier1:
            return re.sub(r"\s+", " ", random.choice(sorted(tier1, key=score_item, reverse=True)[:5]).get("sentence", "")).strip()[:max_len]

        tier2 = [t for t in pool
                 if required_kws and any(kw in t.get("sentence", "").lower() for kw in required_kws)]
        if tier2:
            return re.sub(r"\s+", " ", random.choice(sorted(tier2, key=score_item, reverse=True)[:5]).get("sentence", "")).strip()[:max_len]

        return ""

    def _build_aspect_payload(self, tuples: List[Dict]) -> Dict:
        if not tuples:
            return {"table": [], "radar": [], "strengths": [], "weaknesses": []}
        df = pd.DataFrame(tuples)
        aspects = df["aspect"].unique().tolist()
        table: List[Dict] = []
        radar: List[Dict] = []
        for asp in aspects:
            sub = df[df["aspect"] == asp]
            pos = int(len(sub[sub["sentiment"] == "positive"]))
            neu = int(len(sub[sub["sentiment"] == "neutral"]))
            neg = int(len(sub[sub["sentiment"] == "negative"]))
            total = pos + neu + neg
            total_weight = sub["weight"].sum() if "weight" in sub.columns else total
            if total_weight > 0:
                ws = (sub["sentiment_score"] * sub.get("weight", 1)).sum() / total_weight
            else:
                ws = sub["sentiment_score"].mean()
            normalized = float(round((ws + 1) / 2, 4))
            display_name = ASPECT_DISPLAY_NAMES.get(asp, asp.replace("PRODUCT#", "").replace("DELIVERY#", ""))
            row = {
                "aspect": asp,
                "display_name": display_name,
                "positive": pos, "neutral": neu, "negative": neg, "mentions": total,
                "avg_score_0_1": normalized,
                "best_positive_quote": self._pick_quote_for_aspect(tuples, asp, "positive"),
                "best_positive_quote_count": 0,
                "best_negative_quote": self._pick_quote_for_aspect(tuples, asp, "negative"),
                "best_negative_quote_count": 0,
            }
            table.append(row)
            radar.append({"aspect": asp, "score": normalized})
        table.sort(key=lambda x: x["mentions"], reverse=True)
        print("\n[DEBUG] Full aspect table:")
        for r in table:
            print(f"  {r['display_name']}: score={r['avg_score_0_1']:.3f} "
                  f"pos={r['positive']} neu={r['neutral']} neg={r['negative']} mentions={r['mentions']}")
        strengths_pool = [r for r in table if r["positive"] > r["negative"] and r["positive"] / max(r["mentions"], 1) >= 0.3]
        strengths_pool.sort(key=lambda x: (x["mentions"], x["avg_score_0_1"]), reverse=True)
        weaknesses_pool = [r for r in table if r["negative"] > 0 and (r["negative"] >= r["positive"] or r["negative"] / max(r["mentions"], 1) >= 0.3)]
        weaknesses_pool.sort(key=lambda x: (x["negative"], x["mentions"]), reverse=True)
        return {
            "table": table,
            "radar": radar,
            "strengths": [dict(r) for r in strengths_pool[:4]],
            "weaknesses": [dict(r) for r in weaknesses_pool[:4]],
        }

    def _build_opinion_table(self, tuples: List[Dict]) -> List[Dict]:
        if not tuples:
            return []
        groups: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {"count": 0, "confidence": 0.0, "valid_examples": []})
        for item in tuples:
            aspect = item["aspect"]
            sentiment = item["sentiment"]
            key = (aspect, sentiment)
            sentence = item.get("sentence", "")
            groups[key]["count"] += 1
            groups[key]["confidence"] += item["confidence"]
            required_kws = ASPECT_REQUIRED_KEYWORDS.get(aspect, [aspect.split("#")[-1].lower()])
            forbidden_kws = ASPECT_FORBIDDEN_KEYWORDS.get(aspect, [])
            sl = sentence.lower()
            if any(kw in sl for kw in required_kws) and not any(kw in sl for kw in forbidden_kws):
                if len(groups[key]["valid_examples"]) < 10:
                    groups[key]["valid_examples"].append(sentence)
        result = []
        for (aspect, sentiment), data in groups.items():
            if sentiment != "none":
                valid = data["valid_examples"]
                result.append({
                    "aspect": aspect,
                    "display_name": ASPECT_DISPLAY_NAMES.get(aspect, aspect.replace("PRODUCT#", "")),
                    "sentiment": sentiment,
                    "count": data["count"],
                    "confidence": round(data["confidence"] / data["count"], 3),
                    "example": random.choice(valid) if valid else "",
                })
        return sorted(result, key=lambda x: x["count"], reverse=True)[:25]

    def _representative_reviews(
        self, reviews_df: pd.DataFrame, review_scores: List[Dict]
    ) -> List[Dict]:
        if reviews_df.empty:
            return []
        score_map = {r["review_id"]: r for r in review_scores}
        reviews_df = reviews_df.copy()
        reviews_df["_score"] = reviews_df["review_id"].apply(
            lambda rid: float(score_map.get(str(rid), {}).get("score", 0.0))
        )
        reviews_df["_label"] = reviews_df["review_id"].apply(
            lambda rid: str(score_map.get(str(rid), {}).get("label", "neutral"))
        )
        valid = reviews_df[reviews_df["_score"] != 0.0]
        if valid.empty:
            valid = reviews_df
        result = []
        if len(valid) > 0:
            best = valid.nlargest(1, "_score").iloc[0]
            result.append({"text": str(best.get("clean_text", ""))[:500], "label": str(best.get("_label", "positive")), "rating": int(best.get("rating", 0))})
        if len(valid) > 1:
            worst = valid.nsmallest(1, "_score").iloc[0]
            result.append({"text": str(worst.get("clean_text", ""))[:500], "label": str(worst.get("_label", "negative")), "rating": int(worst.get("rating", 0))})
        if "helpful_count" in valid.columns and len(result) < 3:
            helpful = valid.nlargest(1, "helpful_count").iloc[0]
            if str(helpful.get("review_id")) != str(result[0].get("review_id") if result else ""):
                result.append({"text": str(helpful.get("clean_text", ""))[:500], "label": str(helpful.get("_label", "neutral")), "rating": int(helpful.get("rating", 0))})
        return result[:3]


    # MAIN ENTRY POINT
    def analyze_product(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:

        self._safe_progress(progress_callback, 3, "Trích xuất product ID từ URL...")
        if not product_id and product_url:
            product_id = extract_product_id_from_url(product_url)
        if not product_id:
            raise ValueError("Không thể xác định product ID từ URL đã nhập")
        product_id = str(product_id).strip()
        print(f"\n[Analysis] ═══ Starting for product_id={product_id} ═══")

        self._safe_progress(progress_callback, 8, "Đang lấy thông tin sản phẩm từ Tiki...")
        product_info = fetch_tiki_product_info_live(product_id)

        if product_info["name"] == f"Sản phẩm #{product_id}":
            print("[Analysis] API returned empty — trying CSV fallback for product info")
            csv_info = _get_product_info_from_csv_fallback(product_id)
            if csv_info["name"] != f"Sản phẩm #{product_id}":
                product_info.update(csv_info)

        self._safe_progress(progress_callback, 15, f"Tìm thấy: {product_info['name'][:40]}...")

        self._safe_progress(progress_callback, 18, "Đang thu thập đánh giá từ Tiki API...")
        reviews = fetch_tiki_reviews_live(
            product_id,
            max_pages=self.config.max_review_pages,
            per_page=self.config.reviews_per_page,
            progress_callback=progress_callback,
        )

        if not reviews:
            print(f"[Analysis] Live crawl returned 0 reviews — trying CSV fallback")
            self._safe_progress(progress_callback, 28, "Không tìm thấy review trực tiếp, dùng dữ liệu offline...")
            reviews = _get_reviews_from_csv_fallback(product_id)

        if not reviews:
            raise ValueError(
                f"Không tìm thấy đánh giá nào cho sản phẩm {product_id}. "
                "Sản phẩm có thể chưa có review hoặc URL không hợp lệ."
            )

        self._safe_progress(progress_callback, 30, f"Đã thu thập {len(reviews)} đánh giá. Đang tiền xử lý...")

        reviews_df = self._preprocess_reviews(reviews)
        if reviews_df.empty:
            raise ValueError("Không có đánh giá hợp lệ sau tiền xử lý văn bản")
        reviews_df = reviews_df.head(self.config.max_reviews_for_dashboard)
        print(f"[Analysis] Using {len(reviews_df)} reviews for ABSA")

        self._safe_progress(progress_callback, 40, "Đang phân tích cảm xúc với PhoBERT...")
        tuples, review_scores, sentiment_bucket = self._batch_predict(reviews_df, progress_callback)
        if not tuples:
            raise ValueError("PhoBERT không trả về kết quả — đánh giá quá ngắn hoặc không có nội dung")

        self._safe_progress(progress_callback, 82, "Tổng hợp kết quả theo khía cạnh...")
        aspect_payload = self._build_aspect_payload(tuples)
        self._safe_progress(progress_callback, 86, "Trích xuất ý kiến chi tiết...")
        opinion_table = self._build_opinion_table(tuples)
        risk_flags = detect_risks(tuples)

        total_reviews = len(reviews_df)
        total_weight = sum(item.get("weight", 1) for item in tuples)
        if total_weight > 0:
            weighted_score = sum(item["sentiment_score"] * item.get("weight", 1) for item in tuples) / total_weight
        else:
            weighted_score = sum(item["sentiment_score"] for item in tuples) / len(tuples)
        absa_score = max(0.0, min(1.0, (weighted_score + 1) / 2))
        total_sent = max(sum(sentiment_bucket.values()), 1)
        metrics = {
            "rating_average": float(product_info.get("rating_average", 0.0)),
            "absa_score": float(round(absa_score, 4)),
            "total_reviews_used": int(total_reviews),
            "num_aspects_mentioned": int(len(aspect_payload.get("table", []))),
            "positive_ratio": float(round(sentiment_bucket.get("positive", 0) / total_sent, 4)),
            "neutral_ratio": float(round(sentiment_bucket.get("neutral", 0) / total_sent, 4)),
            "negative_ratio": float(round(sentiment_bucket.get("negative", 0) / total_sent, 4)),
        }
        representative = self._representative_reviews(reviews_df, review_scores)


        self._safe_progress(progress_callback, 98, "Đang hoàn thiện báo cáo...")
        print(f"[Analysis] ✅ Complete! ABSA score: {absa_score:.3f} | Reviews: {total_reviews}")

        result = {
            "product_info": {
                "product_id": str(product_info.get("product_id", "")),
                "name": str(product_info.get("name", "")),
                "price": int(product_info.get("price", 0)) if product_info.get("price") else None,
                "original_price": int(product_info.get("original_price", 0)) if product_info.get("original_price") else None,
                "brand_name": str(product_info.get("brand_name", "")),
                "seller_name": str(product_info.get("seller_name", "")),
                "seller_is_official": bool(product_info.get("seller_is_official", False)),
                "rating_average": float(product_info.get("rating_average", 0)),
                "review_count": int(product_info.get("review_count", 0)),
                "thumbnail_url": str(product_info.get("thumbnail_url", "")),
                "images": list(product_info.get("images", [])),
                "product_url": str(product_info.get("product_url", "")),
                "short_description": str(product_info.get("short_description", "")),
                "category_lv1": str(product_info.get("category_lv1", "")),
                "category_lv2": str(product_info.get("category_lv2", "")),
                "category_lv3": str(product_info.get("category_lv3", "")),
                "sold_quantity": int(product_info.get("sold_quantity", 0)),
            },
            "advice": None,
            "metrics": metrics,
            "aspect": {
                "table": [
                    {
                        "aspect": str(a.get("aspect", "")),
                        "display_name": str(a.get("display_name", a.get("aspect", ""))),
                        "positive": int(a.get("positive", 0)),
                        "neutral": int(a.get("neutral", 0)),
                        "negative": int(a.get("negative", 0)),
                        "mentions": int(a.get("mentions", 0)),
                        "avg_score_0_1": float(a.get("avg_score_0_1", 0)),
                        "best_positive_quote": str(a.get("best_positive_quote", "")),
                        "best_positive_quote_count": int(a.get("best_positive_quote_count", 0)),
                        "best_negative_quote": str(a.get("best_negative_quote", "")),
                        "best_negative_quote_count": int(a.get("best_negative_quote_count", 0)),
                    }
                    for a in aspect_payload.get("table", [])[:15]
                ],
                "radar": [
                    {"aspect": str(r.get("aspect", "")), "score": float(r.get("score", 0))}
                    for r in aspect_payload.get("radar", [])
                ],
                "strengths": [
                    {
                        "aspect": str(s.get("aspect", "")),
                        "display_name": str(s.get("display_name", s.get("aspect", ""))),
                        "avg_score_0_1": float(s.get("avg_score_0_1", 0)),
                        "best_positive_quote": str(s.get("best_positive_quote", "")),
                        "best_positive_quote_count": int(s.get("best_positive_quote_count", 0)),
                        "positive": int(s.get("positive", 0)),
                        "mentions": int(s.get("mentions", 0)),
                    }
                    for s in aspect_payload.get("strengths", [])
                ],
                "weaknesses": [
                    {
                        "aspect": str(w.get("aspect", "")),
                        "display_name": str(w.get("display_name", w.get("aspect", ""))),
                        "avg_score_0_1": float(w.get("avg_score_0_1", 0)),
                        "best_negative_quote": str(w.get("best_negative_quote", "")),
                        "best_negative_quote_count": int(w.get("best_negative_quote_count", 0)),
                        "negative": int(w.get("negative", 0)),
                        "mentions": int(w.get("mentions", 0)),
                    }
                    for w in aspect_payload.get("weaknesses", [])
                ],
            },
            "opinion_table": [
                {
                    "aspect": str(o.get("aspect", "")),
                    "display_name": str(o.get("display_name", o.get("aspect", ""))),
                    "sentiment": str(o.get("sentiment", "")),
                    "count": int(o.get("count", 0)),
                    "confidence": float(o.get("confidence", 0)),
                    "example": str(o.get("example", "")),
                }
                for o in opinion_table
            ],
            "representative_reviews": [
                {
                    "text": str(r.get("text", ""))[:500],
                    "label": str(r.get("label", "neutral")),
                    "rating": int(r.get("rating", 0)) if r.get("rating") else None,
                }
                for r in representative
            ],
            "risk_flags": risk_flags,
        }
        try:
            saved = self._v2_repository.persist_analysis(
                product_info=result["product_info"],
                reviews=reviews_df.to_dict(orient="records"),
                aspect_items=tuples,
                risks=risk_flags,
            )
            if saved:
                print(f"[V2 DB] Saved analysis for product_id={product_id}")
                chunks = opinion_chunks_from_aspect_items(tuples)
                indexed = self._v2_vector_store.index_product_opinions(str(product_id), chunks)
                if indexed:
                    print(f"[V2 pgvector] Indexed {indexed} opinion chunks for product_id={product_id}")
        except Exception as exc:
            print(f"[V2 DB] Save skipped: {exc}")
        self._latest_analysis[str(product_id)] = result
        return result
