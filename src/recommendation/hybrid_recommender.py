#!/usr/bin/env python3
"""
Hybrid recommendation: ABSA (PhoBERT, rule-based aggregation) + CBF (TF-IDF cosine).

Flow:
1) Input product URL or product_id.
2) Crawl/load reviews.
3) Clean and normalize text.
4) PhoBERT ABSA inference per sentence (aspect + sentiment + confidence).
5) Compute dynamic alpha/beta for 2 cases:
   - Few reviews.
   - Noisy reviews.
6) Build CBF scores from product content metadata.
7) Final hybrid score and top-K ranking.
8) Save charts and reports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# ═══════════════════════════════════════════════════════════════
# FORCE CPU MODE - Đặt NGAY ĐẦU FILE trước mọi import khác
# ═══════════════════════════════════════════════════════════════
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

# Vô hiệu hóa CUDA hoàn toàn
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
if hasattr(torch.cuda, 'current_device'):
    torch.cuda.current_device = lambda: None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_phobert import CFG, PhoBERTASQP


CATEGORIES: List[str] = [
    "PRODUCT#QUALITY",
    "DELIVERY#SPEED",
    "DELIVERY#PACKAGING",
    "PRICE#AFFORDABILITY",
    "SELLER#SERVICE",
    "PRODUCT#FUNCTION",
    "PRODUCT#COMFORT",
    "PRODUCT#DESIGN",
    "DELIVERY#ACCURACY",
    "PRODUCT#DURABILITY",
    "PRODUCT#SAFETY",
    "SELLER#AUTHENTICITY",
    "PRODUCT#MATERIAL",
    "PRODUCT#SIZE",
    "PRODUCT#VALUE",
    "PRICE#DISCOUNT",
    "SELLER#RESPONSIVENESS",
]

SENTIMENT_NAMES = ["none", "positive", "neutral", "negative"]
SENTIMENT_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


@dataclass
class HybridConfig:
    reviews_csv: str = "data/raw/tiki_reviews.csv"
    details_csv: str = "data/raw/tiki_me_be_product_detail_full.csv"
    categories_csv: str = "data/raw/tiki_me_be_products_id.csv"
    output_root: str = "results/hybrid_recommendation"
    alpha_min: float = 0.10
    alpha_max: float = 0.70
    n_threshold: int = 10
    lambda_noise: float = 0.80
    short_token_threshold: int = 10
    noise_threshold: float = 0.50
    top_k: int = 10
    preselect_cbf: int = 20          # Giảm từ 30 → 20 (ít candidate hơn, vẫn đủ chất lượng)
    max_candidate_reviews: int = 20  # Giảm từ 30 → 20 reviews/candidate khi cache miss
    max_len: int = 256
    min_sentence_chars: int = 4


class PhoBERTABSAPredictor:
    """PhoBERT inference wrapper for aspect-sentiment prediction."""

    def __init__(self, model_path: str = "models/phobert/best_model.pt", max_len: int = 256):
        # ═══════════════════════════════════════════════════════════════
        # FORCE CPU - Không cho phép dùng GPU dù có CUDA
        # ═══════════════════════════════════════════════════════════════
        self.device = torch.device('cpu')
        print(f"🔧 PhoBERTABSAPredictor using device: {self.device}")
        
        self.max_len = max_len
        self.cache: Dict[str, List[Dict[str, Any]]] = {}

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PhoBERT model not found: {model_path}")

        # Load checkpoint với map_location='cpu' để tránh lỗi CUDA
        ckpt = torch.load(model_path, map_location='cpu')
        model_name = ckpt.get("config", {}).get("model_name", CFG.PHOBERT_NAME)

        local_tokenizer_dir = Path("models/phobert/tokenizer")
        if local_tokenizer_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_dir))
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = self._load_model_from_checkpoint(ckpt, model_name=model_name)
        print("✅ PhoBERTABSAPredictor loaded successfully on CPU")

    def _load_model_from_checkpoint(self, ckpt: Dict[str, Any], model_name: str) -> PhoBERTASQP:
        candidates: List[str] = []
        local_backbone_dir = Path("models/phobert/base_model")
        if local_backbone_dir.exists():
            candidates.append(str(local_backbone_dir))
        candidates.append(model_name)

        tried: List[str] = []
        old_model_name = CFG.PHOBERT_NAME
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        try:
            for src in candidates:
                if src in tried:
                    continue
                tried.append(src)
                try:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    CFG.PHOBERT_NAME = src
                    model = PhoBERTASQP(CFG).to(self.device)  # Chuyển sang CPU
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()
                    return model
                except Exception:
                    continue
        finally:
            CFG.PHOBERT_NAME = old_model_name
            if old_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_offline
            if old_tf_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline

        raise RuntimeError(
            "Cannot initialize PhoBERT backbone. "
            "Please either: "
            "(1) keep internet on for first-time HuggingFace download, or "
            "(2) put local PhoBERT base model in models/phobert/base_model/"
        )

    def predict_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        sentence = sentence.strip()
        if not sentence:
            return []
        if sentence in self.cache:
            return self.cache[sentence]

        enc = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits_per_cat = self.model(input_ids, attention_mask)

        outputs: List[Dict[str, Any]] = []
        for cat_idx, logits in enumerate(logits_per_cat):
            probs = F.softmax(logits[0], dim=-1)
            pred_idx = int(torch.argmax(probs).item())
            pred_name = SENTIMENT_NAMES[pred_idx]
            conf = float(probs[pred_idx].item())
            if pred_name == "none":
                continue

            outputs.append(
                {
                    "aspect": CATEGORIES[cat_idx],
                    "sentiment": pred_name,
                    "confidence": conf,
                    "sentiment_score": SENTIMENT_TO_SCORE[pred_name],
                }
            )

        self.cache[sentence] = outputs
        return outputs

    def predict_batch(self, sentences: List[str]) -> List[List[Dict[str, Any]]]:
        """Batch prediction for better performance on CPU."""
        results = []
        for sentence in sentences:
            results.append(self.predict_sentence(sentence))
        return results


def extract_product_id_from_url(product_url: str) -> Optional[str]:
    parsed = urlparse(product_url)
    m = re.search(r"-p(\d+)\.html", parsed.path)
    if m:
        return m.group(1)

    query = parse_qs(parsed.query)
    for key in ("spid", "product_id", "id"):
        if key in query and query[key]:
            return str(query[key][0])

    m2 = re.search(r"(\d+)", parsed.path)
    return m2.group(1) if m2 else None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def to_bool(x: Any) -> bool:
    s = _safe_str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def clean_text(text: str) -> str:
    text = _safe_str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    try:
        from src.data_preprocessing.normalize_text import normalize_text as project_norm

        return project_norm(text)
    except Exception:
        t = clean_text(text).lower()
        t = re.sub(r"\s+", " ", t).strip()
        return t


def split_sentences(text: str) -> List[str]:
    text = _safe_str(text).strip()
    if not text:
        return []

    try:
        from underthesea import sent_tokenize

        sents = sent_tokenize(text)
        sents = [s.strip() for s in sents if s and s.strip()]
        if sents:
            return sents
    except Exception:
        pass

    parts = re.split(r"[\n\.!?;]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _safe_str(text).lower(), flags=re.UNICODE)


def spam_pattern_penalty(text: str) -> float:
    toks = simple_tokenize(text)
    if not toks:
        return 1.0

    unique_ratio = len(set(toks)) / max(len(toks), 1)
    repeated_char = bool(re.search(r"(\w)\1{4,}", text))
    many_links = len(re.findall(r"https?://|www\.", text)) > 1

    penalty = 0.0
    if unique_ratio < 0.40:
        penalty += 0.5
    if repeated_char:
        penalty += 0.3
    if many_links:
        penalty += 0.5
    return min(1.0, penalty)


def compute_noise_score(
    text: str,
    is_verified: bool,
    helpful_count: float,
    short_token_threshold: int = 10,
) -> float:
    toks = simple_tokenize(text)
    score = 0.0

    if len(toks) < short_token_threshold:
        score += 0.5

    if (not is_verified) and helpful_count <= 0:
        score += 0.2

    score += spam_pattern_penalty(text)
    return float(min(1.0, score))


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_dynamic_alpha_beta(
    n_reviews: int,
    noise_ratio: float,
    alpha_min: float,
    alpha_max: float,
    n_threshold: int,
    lambda_noise: float,
) -> Tuple[float, float, float, float]:
    ratio_n = min(1.0, n_reviews / max(n_threshold, 1))
    alpha_case1 = alpha_min + (alpha_max - alpha_min) * ratio_n
    alpha_case2 = alpha_case1 * (1.0 - lambda_noise * noise_ratio)
    alpha = clip(alpha_case2, alpha_min, alpha_max)
    beta = 1.0 - alpha
    return alpha, beta, alpha_case1, alpha_case2


class HybridABSACBFRecommender:
    def __init__(self, config: HybridConfig, predictor: PhoBERTABSAPredictor):
        self.cfg = config
        self.predictor = predictor

        # ── Disk cache cho ABSA scores của candidate products ──────────────
        self._absa_cache_dir = Path("cache/absa_scores")
        self._absa_cache_dir.mkdir(parents=True, exist_ok=True)

        self.reviews_df = pd.read_csv(self.cfg.reviews_csv, dtype=str)
        self.details_df = pd.read_csv(self.cfg.details_csv, dtype=str)
        self.categories_df = pd.read_csv(self.cfg.categories_csv, dtype=str)

        for df in (self.reviews_df, self.details_df, self.categories_df):
            if "product_id" in df.columns:
                df["product_id"] = df["product_id"].astype(str)

        self.products_df = self.details_df.merge(
            self.categories_df[["product_id", "category_id", "category_lv1", "category_lv2", "category_lv3"]],
            on="product_id",
            how="left",
        )

        self.products_df["content_text"] = self.products_df.apply(self._build_content_text, axis=1)

        # ── Pre-compute TF-IDF matrix 1 lần lúc khởi động ─────────────────
        print("⚡ Pre-computing TF-IDF matrix...")
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._tfidf_pid_list: List[str] = []
        self._build_tfidf_index()
        print(f"✅ TF-IDF index built: {len(self._tfidf_pid_list)} products")

    # ── TF-IDF pre-compute ─────────────────────────────────────────────────
    def _build_tfidf_index(self) -> None:
        """Build TF-IDF matrix 1 lần lúc init, dùng lại cho mọi request."""
        df = self.products_df.drop_duplicates(subset=["product_id"]).copy()
        df = df[df["content_text"].str.len() > 0].copy()
        self._tfidf_pid_list = df["product_id"].astype(str).tolist()
        texts = df["content_text"].fillna("").tolist()
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    # ── ABSA disk cache ────────────────────────────────────────────────────
    def _get_absa_cached(self, product_id: str) -> float:
        """Trả về ABSA score từ disk cache. Nếu chưa có thì tính và lưu lại."""
        cache_file = self._absa_cache_dir / f"{product_id}.json"
        if cache_file.exists():
            try:
                return float(json.loads(cache_file.read_text())["p_absa"])
            except Exception:
                pass
        rv = self._load_local_reviews(product_id)
        rv = self.preprocess_reviews(rv)
        score = self.compute_absa_score(rv, max_reviews=self.cfg.max_candidate_reviews)
        try:
            cache_file.write_text(json.dumps({"p_absa": score["p_absa"]}))
        except Exception:
            pass
        return float(score["p_absa"])

    @staticmethod
    def _build_content_text(row: pd.Series) -> str:
        fields = [
            "name",
            "short_description",
            "description",
            "specifications",
            "brand_name",
            "category_lv1",
            "category_lv2",
            "category_lv3",
        ]
        parts = [_safe_str(row.get(f, "")) for f in fields]
        text = " ".join([p for p in parts if p])
        return clean_text(text)

    def _get_product_meta(self, product_id: str) -> pd.Series:
        subset = self.products_df[self.products_df["product_id"] == str(product_id)]
        if subset.empty:
            raise ValueError(f"Cannot find product_id={product_id} in product detail dataset")
        return subset.iloc[0]

    def _load_local_reviews(self, product_id: str) -> pd.DataFrame:
        subset = self.reviews_df[self.reviews_df["product_id"] == str(product_id)].copy()
        if subset.empty:
            return subset

        expected = [
            "review_id",
            "product_id",
            "rating",
            "title",
            "content",
            "helpful_count",
            "is_verified",
        ]
        for c in expected:
            if c not in subset.columns:
                subset[c] = ""

        subset["helpful_count"] = pd.to_numeric(subset["helpful_count"], errors="coerce").fillna(0.0)
        subset["is_verified"] = subset["is_verified"].apply(to_bool)
        return subset

    def _crawl_reviews_live(self, product_id: str, max_pages: int = 10) -> pd.DataFrame:
        base_url = "https://tiki.vn/api/v2/reviews"
        records: List[Dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            params = {
                "product_id": product_id,
                "page": page,
                "limit": 20,
                "sort": "score|desc",
            }
            try:
                response = requests.get(base_url, params=params, timeout=20)
                if response.status_code != 200:
                    break
                payload = response.json()
            except Exception:
                break

            items = payload.get("data", [])
            if not items:
                break

            for rv in items:
                creator = rv.get("created_by") or {}
                records.append(
                    {
                        "review_id": rv.get("id", ""),
                        "product_id": str(product_id),
                        "rating": rv.get("rating", ""),
                        "title": rv.get("title", ""),
                        "content": rv.get("content", ""),
                        "helpful_count": rv.get("thank_count", 0),
                        "is_verified": bool(creator.get("purchased_at")),
                    }
                )

            time.sleep(0.25)

        if not records:
            return pd.DataFrame()

        out = pd.DataFrame(records)
        out["helpful_count"] = pd.to_numeric(out["helpful_count"], errors="coerce").fillna(0.0)
        out["is_verified"] = out["is_verified"].apply(to_bool)
        return out

    def load_reviews(self, product_id: str, reviews_source: str = "auto") -> pd.DataFrame:
        if reviews_source not in {"auto", "local", "live"}:
            raise ValueError("reviews_source must be one of: auto, local, live")

        if reviews_source == "local":
            return self._load_local_reviews(product_id)

        if reviews_source == "live":
            live = self._crawl_reviews_live(product_id)
            if live.empty:
                return self._load_local_reviews(product_id)
            return live

        live = self._crawl_reviews_live(product_id)
        if not live.empty:
            return live
        return self._load_local_reviews(product_id)

    def preprocess_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        if reviews.empty:
            return reviews

        df = reviews.copy()
        df["title"] = df["title"].fillna("").astype(str)
        df["content"] = df["content"].fillna("").astype(str)
        df["raw_text"] = (df["title"].str.strip() + ". " + df["content"].str.strip()).str.strip(". ")
        df["clean_text"] = df["raw_text"].apply(clean_text)
        df["normalized_text"] = df["clean_text"].apply(normalize_text)
        df = df[df["normalized_text"].str.len() > 0].copy()
        if df.empty:
            return df

        df["helpful_count"] = pd.to_numeric(df.get("helpful_count", 0), errors="coerce").fillna(0.0)
        df["is_verified"] = df.get("is_verified", False).apply(to_bool)

        df["noise_score"] = df.apply(
            lambda row: compute_noise_score(
                row["normalized_text"],
                bool(row["is_verified"]),
                float(row["helpful_count"]),
                short_token_threshold=self.cfg.short_token_threshold,
            ),
            axis=1,
        )
        df["is_noisy"] = df["noise_score"] >= self.cfg.noise_threshold

        def _review_weight(row: pd.Series) -> float:
            helpful_factor = 1.0 + math.log1p(float(row["helpful_count"]))
            verified_factor = 1.25 if bool(row["is_verified"]) else 1.0
            noise_factor = max(0.05, 1.0 - float(row["noise_score"]))
            weight = helpful_factor * verified_factor * noise_factor
            if bool(row["is_noisy"]):
                weight *= 0.5
            return float(max(0.01, weight))

        df["review_weight"] = df.apply(_review_weight, axis=1)
        return df

    def compute_absa_score(self, reviews: pd.DataFrame, max_reviews: Optional[int] = None) -> Dict[str, Any]:
        if reviews.empty:
            return {
                "p_absa": 0.5,
                "p_absa_raw": 0.0,
                "used_reviews": 0,
                "used_sentences": 0,
                "used_tuples": 0,
                "per_aspect": {c: 0.5 for c in CATEGORIES},
            }

        df = reviews.copy()
        if max_reviews is not None and max_reviews > 0 and len(df) > max_reviews:
            df = df.sort_values(["helpful_count", "is_verified"], ascending=[False, False]).head(max_reviews)

        rows: List[Dict[str, Any]] = []
        used_sentences = 0

        for _, rv in df.iterrows():
            text = _safe_str(rv["normalized_text"])
            sentences = [s for s in split_sentences(text) if len(s) >= self.cfg.min_sentence_chars]
            if not sentences:
                continue

            for sent in sentences:
                absa_items = self.predictor.predict_sentence(sent)
                used_sentences += 1
                if not absa_items:
                    continue

                for item in absa_items:
                    weight = float(rv["review_weight"])
                    conf = float(item["confidence"])
                    sent_score = float(item["sentiment_score"])
                    rows.append(
                        {
                            "aspect": item["aspect"],
                            "confidence": conf,
                            "sentiment_score": sent_score,
                            "weighted_factor": conf * weight,
                        }
                    )

        if not rows:
            return {
                "p_absa": 0.5,
                "p_absa_raw": 0.0,
                "used_reviews": int(len(df)),
                "used_sentences": int(used_sentences),
                "used_tuples": 0,
                "per_aspect": {c: 0.5 for c in CATEGORIES},
            }

        absa_df = pd.DataFrame(rows)
        numerator = float((absa_df["weighted_factor"] * absa_df["sentiment_score"]).sum())
        denominator = float(absa_df["weighted_factor"].sum())
        raw = numerator / denominator if denominator > 0 else 0.0
        p_absa = (raw + 1.0) / 2.0
        p_absa = float(clip(p_absa, 0.0, 1.0))

        per_aspect: Dict[str, float] = {}
        for asp in CATEGORIES:
            sub = absa_df[absa_df["aspect"] == asp]
            if sub.empty:
                per_aspect[asp] = 0.5
                continue
            num = float((sub["weighted_factor"] * sub["sentiment_score"]).sum())
            den = float(sub["weighted_factor"].sum())
            raw_asp = num / den if den > 0 else 0.0
            per_aspect[asp] = float(clip((raw_asp + 1.0) / 2.0, 0.0, 1.0))

        return {
            "p_absa": p_absa,
            "p_absa_raw": float(raw),
            "used_reviews": int(len(df)),
            "used_sentences": int(used_sentences),
            "used_tuples": int(len(absa_df)),
            "per_aspect": per_aspect,
        }

    def _compute_quality_stats(self, reviews: pd.DataFrame) -> Dict[str, float]:
        if reviews.empty:
            return {"n_reviews": 0, "noise_ratio": 1.0, "n_noisy": 0}
        n_reviews = int(len(reviews))
        n_noisy = int(reviews["is_noisy"].sum())
        noise_ratio = float(n_noisy / max(n_reviews, 1))
        return {"n_reviews": n_reviews, "noise_ratio": noise_ratio, "n_noisy": n_noisy}

    def _compute_cbf_scores(self, query_product_id: str) -> pd.DataFrame:
        """
        Dùng pre-computed TF-IDF matrix — không fit lại mỗi request.
        Chỉ tính cosine similarity 1 hàng (query) vs toàn bộ matrix.
        """
        pid_list = self._tfidf_pid_list
        idx_map  = {pid: i for i, pid in enumerate(pid_list)}
        q_idx    = idx_map.get(str(query_product_id))

        if q_idx is None:
            # Sản phẩm mới chưa có trong index → fallback fit lại (hiếm gặp)
            print(f"⚠️  product {query_product_id} not in TF-IDF index, re-fitting...")
            self._build_tfidf_index()
            idx_map = {pid: i for i, pid in enumerate(self._tfidf_pid_list)}
            q_idx   = idx_map.get(str(query_product_id))
            if q_idx is None:
                raise ValueError(f"Query product {query_product_id} not found in catalog")

        # Cosine similarity 1 vector vs toàn bộ matrix (nhanh hơn nhiều so với fit lại)
        sims = cosine_similarity(self._tfidf_matrix[q_idx], self._tfidf_matrix).flatten()

        # Lấy metadata tương ứng
        meta_df = (
            self.products_df
            .drop_duplicates(subset=["product_id"])
            .set_index("product_id")
            .reindex(pid_list)
            .reset_index()
        )

        keep_cols = ["product_id", "name", "brand_name", "category_id",
                     "category_lv1", "category_lv2", "category_lv3", "product_url"]
        available = [c for c in keep_cols if c in meta_df.columns]
        out = meta_df[available].copy()
        out["cbf_score"] = sims.astype(float)
        out = out[out["product_id"].astype(str) != str(query_product_id)].copy()
        out = out.sort_values("cbf_score", ascending=False).reset_index(drop=True)
        return out

    def _plot_alpha_beta(self, alpha: float, beta: float, out_path: str) -> None:
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            vals = [alpha, beta]
            labels = ["alpha (ABSA)", "beta (CBF)"]
            colors = ["#4e79a7", "#59a14f"]
            ax.bar(labels, vals, color=colors)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_title("Dynamic Weighting")
            ax.set_ylabel("Weight")
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not save alpha/beta plot: {e}")

    def _plot_radar_aspects(self, per_aspect: Dict[str, float], out_path: str) -> None:
        try:
            aspects = list(per_aspect.keys())
            values = [per_aspect[a] for a in aspects]
            labels = [a.split("#")[-1] if "#" in a else a for a in aspects]

            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values_cycle = values + values[:1]
            angles_cycle = angles + angles[:1]

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles_cycle, values_cycle, color="#4e79a7", linewidth=2)
            ax.fill(angles_cycle, values_cycle, color="#4e79a7", alpha=0.25)
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title("ABSA Aspect Sentiment (0..1)", pad=20)
            fig.tight_layout()
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not save radar plot: {e}")

    def _plot_topk_breakdown(self, ranked: pd.DataFrame, out_path: str) -> None:
        try:
            top = ranked.head(min(10, len(ranked))).copy()
            if top.empty:
                return

            top = top.sort_values("hybrid_score", ascending=True)
            y_labels = [f"{pid}" for pid in top["product_id"].tolist()]

            fig, ax = plt.subplots(figsize=(10, 6))
            y = np.arange(len(top))
            ax.barh(y, top["hybrid_score"], color="#f28e2b", alpha=0.8, label="Hybrid")
            ax.barh(y, top["cbf_score"], color="#59a14f", alpha=0.45, label="CBF")
            ax.barh(y, top["absa_score"], color="#4e79a7", alpha=0.45, label="ABSA")

            ax.set_yticks(y)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel("Score (0..1)")
            ax.set_title("Top-K Recommendation Score Breakdown")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not save topk breakdown plot: {e}")

    def run(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        reviews_source: str = "auto",
    ) -> Dict[str, Any]:
        if not product_id:
            if not product_url:
                raise ValueError("Please provide product_url or product_id")
            product_id = extract_product_id_from_url(product_url)

        if not product_id:
            raise ValueError("Cannot extract product_id from input")

        product_id = str(product_id)
        query_meta = self._get_product_meta(product_id)

        query_reviews = self.load_reviews(product_id, reviews_source=reviews_source)
        query_reviews = self.preprocess_reviews(query_reviews)

        quality = self._compute_quality_stats(query_reviews)
        alpha, beta, alpha_case1, alpha_case2 = compute_dynamic_alpha_beta(
            n_reviews=quality["n_reviews"],
            noise_ratio=quality["noise_ratio"],
            alpha_min=self.cfg.alpha_min,
            alpha_max=self.cfg.alpha_max,
            n_threshold=self.cfg.n_threshold,
            lambda_noise=self.cfg.lambda_noise,
        )

        query_absa = self.compute_absa_score(query_reviews)

        cbf_candidates = self._compute_cbf_scores(product_id)
        preselect = cbf_candidates.head(self.cfg.preselect_cbf).copy()

        # ── Parallel ABSA scoring với disk cache ──────────────────────────
        import concurrent.futures as _cf
        pid_list = preselect["product_id"].astype(str).tolist()

        def _score_one(pid: str) -> Tuple[str, float]:
            return pid, self._get_absa_cached(pid)

        absa_cache: Dict[str, float] = {}
        max_workers = min(4, len(pid_list))
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for pid, score in pool.map(_score_one, pid_list):
                absa_cache[pid] = score

        preselect["absa_score"] = preselect["product_id"].astype(str).map(absa_cache).fillna(0.5)
        preselect["hybrid_score"] = alpha * preselect["absa_score"] + beta * preselect["cbf_score"]
        ranked = preselect.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        top_k_df = ranked.head(self.cfg.top_k).copy()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.cfg.output_root) / f"product_{product_id}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ranked_csv = out_dir / "ranked_products.csv"
        top_csv = out_dir / "top_k_products.csv"
        details_json = out_dir / "summary.json"

        ranked.to_csv(ranked_csv, index=False, encoding="utf-8-sig")
        top_k_df.to_csv(top_csv, index=False, encoding="utf-8-sig")

        alpha_beta_png = out_dir / "dynamic_alpha_beta.png"
        radar_png = out_dir / "query_aspect_radar.png"
        topk_png = out_dir / "topk_breakdown.png"

        self._plot_alpha_beta(alpha, beta, str(alpha_beta_png))
        self._plot_radar_aspects(query_absa["per_aspect"], str(radar_png))
        self._plot_topk_breakdown(top_k_df, str(topk_png))

        summary = {
            "input": {
                "product_id": product_id,
                "product_url": _safe_str(query_meta.get("product_url", product_url or "")),
                "product_name": _safe_str(query_meta.get("name", "")),
            },
            "query_review_quality": quality,
            "dynamic_weighting": {
                "alpha": alpha,
                "beta": beta,
                "alpha_case1": alpha_case1,
                "alpha_case2_before_clip": alpha_case2,
                "params": {
                    "alpha_min": self.cfg.alpha_min,
                    "alpha_max": self.cfg.alpha_max,
                    "n_threshold": self.cfg.n_threshold,
                    "lambda_noise": self.cfg.lambda_noise,
                },
            },
            "query_absa": query_absa,
            "outputs": {
                "output_dir": str(out_dir),
                "ranked_csv": str(ranked_csv),
                "top_k_csv": str(top_csv),
                "alpha_beta_chart": str(alpha_beta_png),
                "aspect_radar_chart": str(radar_png),
                "topk_breakdown_chart": str(topk_png),
            },
            "top_k_preview": top_k_df.head(10).to_dict(orient="records"),
        }

        with open(details_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid recommender: ABSA (PhoBERT) + CBF with dynamic weighting"
    )
    parser.add_argument("--product_url", type=str, default=None, help="Tiki product URL")
    parser.add_argument("--product_id", type=str, default=None, help="Product ID (if not using URL)")
    parser.add_argument(
        "--reviews_source",
        type=str,
        default="auto",
        choices=["auto", "local", "live"],
        help="Review source: local CSV, live crawl, or auto fallback",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--preselect_cbf", type=int, default=30)
    parser.add_argument("--max_candidate_reviews", type=int, default=30)
    parser.add_argument("--alpha_min", type=float, default=0.10)
    parser.add_argument("--alpha_max", type=float, default=0.70)
    parser.add_argument("--n_threshold", type=int, default=10)
    parser.add_argument("--lambda_noise", type=float, default=0.80)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = HybridConfig(
        top_k=args.top_k,
        preselect_cbf=args.preselect_cbf,
        max_candidate_reviews=args.max_candidate_reviews,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        n_threshold=args.n_threshold,
        lambda_noise=args.lambda_noise,
    )

    predictor = PhoBERTABSAPredictor(max_len=cfg.max_len)
    recommender = HybridABSACBFRecommender(config=cfg, predictor=predictor)

    summary = recommender.run(
        product_url=args.product_url,
        product_id=args.product_id,
        reviews_source=args.reviews_source,
    )

    print("\n=== HYBRID RECOMMENDATION DONE ===")
    print(f"Product ID: {summary['input']['product_id']}")
    print(f"Product Name: {summary['input']['product_name']}")
    print(f"Alpha (ABSA): {summary['dynamic_weighting']['alpha']:.4f}")
    print(f"Beta (CBF):  {summary['dynamic_weighting']['beta']:.4f}")
    print(f"Query reviews: {summary['query_review_quality']['n_reviews']}")
    print(f"Noise ratio: {summary['query_review_quality']['noise_ratio']:.4f}")
    print(f"Output dir: {summary['outputs']['output_dir']}")


if __name__ == "__main__":
    main()