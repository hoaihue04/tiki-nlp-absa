#!/usr/bin/env python3
"""
improved_hybrid_recommender.py - Hệ thống gợi ý hybrid cải tiến
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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import parse_qs, urlparse

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_phobert import CFG, PhoBERTASQP


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CATEGORIES: List[str] = [
    "PRODUCT#QUALITY", "DELIVERY#SPEED", "DELIVERY#PACKAGING",
    "PRICE#AFFORDABILITY", "SELLER#SERVICE", "PRODUCT#FUNCTION",
    "PRODUCT#COMFORT", "PRODUCT#DESIGN", "DELIVERY#ACCURACY",
    "PRODUCT#DURABILITY", "PRODUCT#SAFETY", "SELLER#AUTHENTICITY",
    "PRODUCT#MATERIAL", "PRODUCT#SIZE", "PRODUCT#VALUE",
    "PRICE#DISCOUNT", "SELLER#RESPONSIVENESS",
]

SENTIMENT_NAMES = ["none", "positive", "neutral", "negative"]
SENTIMENT_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImprovedHybridConfig:
    reviews_csv: str = "data/raw/Tiki_be_reviews.csv"
    details_csv: str = "data/raw/Tiki_be_detail.csv"
    categories_csv: str = "data/raw/Tiki_be_product_id.csv"
    listing_csv: str = "data/raw/Tiki_be_listing.csv"
    output_root: str = "results/improved_hybrid_recommendation"
    alpha_min: float = 0.20
    alpha_max: float = 0.60
    n_threshold: int = 5
    lambda_noise: float = 0.60
    candidate_pool_size: int = 150
    top_k: int = 20
    max_candidate_reviews: int = 30
    confidence_threshold: float = 0.7
    min_sentence_chars: int = 4
    max_len: int = 256
    category_level: str = "lv3"
    boost_same_category: float = 1.2
    cache_dir: str = "cache/improved_absa_scores"


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

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


def compute_noise_score(text: str, is_verified: bool, helpful_count: float, short_token_threshold: int = 10) -> float:
    toks = simple_tokenize(text)
    score = 0.0
    if len(toks) < short_token_threshold:
        score += 0.5
    if (not is_verified) and helpful_count <= 0:
        score += 0.2
    unique_ratio = len(set(toks)) / max(len(toks), 1) if toks else 1.0
    if unique_ratio < 0.40:
        score += 0.3
    repeated_char = bool(re.search(r"(\w)\1{4,}", text))
    if repeated_char:
        score += 0.3
    return min(1.0, score)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ═══════════════════════════════════════════════════════════════════════════
# PHOBERT ABSA PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedPhoBERTPredictor:
    def __init__(self, model_path: str = "models/phobert/best_model.pt", max_len: int = 256):
        self.device = torch.device('cpu')
        print(f"🔧 ImprovedPhoBERTPredictor using device: {self.device}")
        self.max_len = max_len
        self.cache: Dict[str, List[Dict[str, Any]]] = {}

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PhoBERT model not found: {model_path}")

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
        print("✅ ImprovedPhoBERTPredictor loaded successfully on CPU")

    def _load_model_from_checkpoint(self, ckpt: Dict[str, Any], model_name: str) -> PhoBERTASQP:
        candidates = [model_name]
        local_backbone_dir = Path("models/phobert/base_model")
        if local_backbone_dir.exists():
            candidates.insert(0, str(local_backbone_dir))

        old_model_name = CFG.PHOBERT_NAME
        try:
            for src in candidates:
                try:
                    CFG.PHOBERT_NAME = src
                    model = PhoBERTASQP(CFG).to(self.device)
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()
                    return model
                except Exception:
                    continue
        finally:
            CFG.PHOBERT_NAME = old_model_name

        raise RuntimeError("Cannot initialize PhoBERT backbone")

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
            outputs.append({
                "aspect": CATEGORIES[cat_idx],
                "sentiment": pred_name,
                "confidence": conf,
                "sentiment_score": SENTIMENT_TO_SCORE[pred_name],
            })

        self.cache[sentence] = outputs
        return outputs


# ═══════════════════════════════════════════════════════════════════════════
# IMPROVED HYBRID RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedHybridRecommender:
    def __init__(self, config: ImprovedHybridConfig, predictor: ImprovedPhoBERTPredictor):
        self.cfg = config
        self.predictor = predictor

        self._absa_cache_dir = Path(config.cache_dir)
        self._absa_cache_dir.mkdir(parents=True, exist_ok=True)

        print("📂 Loading data...")
        self.reviews_df = pd.read_csv(config.reviews_csv, dtype=str)
        self.details_df = pd.read_csv(config.details_csv, dtype=str)
        self.categories_df = pd.read_csv(config.categories_csv, dtype=str)
        
        if os.path.exists(config.listing_csv):
            self.listing_df = pd.read_csv(config.listing_csv, dtype=str)
        else:
            self.listing_df = pd.DataFrame()

        for df in (self.reviews_df, self.details_df, self.categories_df):
            if "product_id" in df.columns:
                df["product_id"] = df["product_id"].astype(str)

        self.products_df = self.details_df.merge(
            self.categories_df[["product_id", "category_id", "category_lv1", "category_lv2", "category_lv3"]],
            on="product_id",
            how="left",
        )

        print("⚡ Building TF-IDF index...")
        self._build_tfidf_index()
        print(f"✅ TF-IDF index built: {len(self._tfidf_pid_list)} products")
        self._build_category_index()

    def _build_tfidf_index(self) -> None:
        df = self.products_df.drop_duplicates(subset=["product_id"]).copy()
        df["content_text"] = df.apply(self._build_content_text, axis=1)
        df = df[df["content_text"].str.len() > 0].copy()
        self._tfidf_pid_list = df["product_id"].astype(str).tolist()
        texts = df["content_text"].fillna("").tolist()
        self._tfidf_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    def _build_category_index(self) -> None:
        self.category_to_products: Dict[str, List[str]] = defaultdict(list)
        for _, row in self.products_df.iterrows():
            cat_lv3 = str(row.get('category_lv3', ''))
            cat_lv2 = str(row.get('category_lv2', ''))
            cat_id = str(row.get('category_id', ''))
            pid = str(row['product_id'])
            if cat_lv3 and cat_lv3 != 'nan':
                self.category_to_products[f"lv3_{cat_lv3}"].append(pid)
            if cat_lv2 and cat_lv2 != 'nan':
                self.category_to_products[f"lv2_{cat_lv2}"].append(pid)
            if cat_id and cat_id != 'nan':
                self.category_to_products[f"id_{cat_id}"].append(pid)

    @staticmethod
    def _build_content_text(row: pd.Series) -> str:
        fields = ["name", "short_description", "description", "specifications", "brand_name"]
        parts = [_safe_str(row.get(f, "")) for f in fields]
        text = " ".join([p for p in parts if p])
        return clean_text(text)

    def _get_product_meta(self, product_id: str) -> pd.Series:
        subset = self.products_df[self.products_df["product_id"] == str(product_id)]
        if subset.empty:
            raise ValueError(f"Cannot find product_id={product_id}")
        return subset.iloc[0]

    def _load_local_reviews(self, product_id: str) -> pd.DataFrame:
        subset = self.reviews_df[self.reviews_df["product_id"] == str(product_id)].copy()
        if subset.empty:
            return subset
        
        expected = ["review_id", "product_id", "rating", "title", "content", "helpful_count", "is_verified"]
        for c in expected:
            if c not in subset.columns:
                subset[c] = ""
        
        subset["rating"] = pd.to_numeric(subset.get("rating", 0), errors="coerce").fillna(3)
        subset["helpful_count"] = pd.to_numeric(subset.get("helpful_count", 0), errors="coerce").fillna(0)
        subset["is_verified"] = subset.get("is_verified", "false").apply(to_bool)
        return subset

    def _crawl_reviews_live(self, product_id: str, max_pages: int = 5) -> pd.DataFrame:
        base_url = "https://tiki.vn/api/v2/reviews"
        records: List[Dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            params = {"product_id": product_id, "page": page, "limit": 20, "sort": "score|desc"}
            try:
                response = requests.get(base_url, params=params, timeout=15)
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
                records.append({
                    "review_id": str(rv.get("id", "")),
                    "product_id": str(product_id),
                    "rating": float(rv.get("rating", 3)),
                    "title": rv.get("title", ""),
                    "content": rv.get("content", ""),
                    "helpful_count": int(rv.get("thank_count", 0)),
                    "is_verified": bool(creator.get("purchased_at")),
                })
            time.sleep(0.3)

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def load_reviews(self, product_id: str, reviews_source: str = "auto") -> pd.DataFrame:
        if reviews_source == "local":
            return self._load_local_reviews(product_id)
        if reviews_source == "live":
            live = self._crawl_reviews_live(product_id)
            if not live.empty:
                return live
            return self._load_local_reviews(product_id)
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

        df["noise_score"] = df.apply(
            lambda row: compute_noise_score(
                row["normalized_text"],
                bool(row.get("is_verified", False)),
                float(row.get("helpful_count", 0)),
            ),
            axis=1,
        )
        df["is_noisy"] = df["noise_score"] >= 0.5

        df["review_weight"] = df.apply(
            lambda row: (1.0 + math.log1p(row.get("helpful_count", 0))) *
                        (1.25 if row.get("is_verified", False) else 1.0) *
                        max(0.1, 1.0 - row["noise_score"]),
            axis=1,
        )
        return df

    def compute_absa_score(self, reviews: pd.DataFrame, max_reviews: Optional[int] = None) -> Dict[str, Any]:
        if reviews.empty:
            return {"p_absa": 0.5, "used_reviews": 0, "used_sentences": 0}

        df = reviews.copy()
        if max_reviews and len(df) > max_reviews:
            df = df.sort_values(["helpful_count", "rating"], ascending=[False, False]).head(max_reviews)

        rows = []
        for _, rv in df.iterrows():
            text = _safe_str(rv["normalized_text"])
            sentences = [s for s in split_sentences(text) if len(s) >= self.cfg.min_sentence_chars]
            if not sentences:
                continue

            rating_weight = float(rv.get("rating", 3)) / 5.0

            for sent in sentences:
                absa_items = self.predictor.predict_sentence(sent)
                for item in absa_items:
                    conf = item["confidence"]
                    if conf < self.cfg.confidence_threshold:
                        continue
                    weight = rating_weight * rv["review_weight"] * conf
                    rows.append({"sentiment_score": item["sentiment_score"], "weight": weight})

        if not rows:
            return {"p_absa": 0.5, "used_reviews": len(df), "used_sentences": 0}

        total_weight = sum(r["weight"] for r in rows)
        weighted_sum = sum(r["weight"] * (r["sentiment_score"] + 1) / 2 for r in rows)
        p_absa = weighted_sum / total_weight if total_weight > 0 else 0.5
        p_absa = clip(p_absa, 0.0, 1.0)

        return {"p_absa": p_absa, "used_reviews": len(df), "used_sentences": 0}

    def _get_absa_cached(self, product_id: str) -> float:
        cache_file = self._absa_cache_dir / f"{product_id}.json"
        if cache_file.exists():
            try:
                return float(json.loads(cache_file.read_text())["p_absa"])
            except Exception:
                pass
        reviews = self._load_local_reviews(product_id)
        reviews = self.preprocess_reviews(reviews)
        score = self.compute_absa_score(reviews, max_reviews=self.cfg.max_candidate_reviews)
        try:
            cache_file.write_text(json.dumps({"p_absa": score["p_absa"]}))
        except Exception:
            pass
        return float(score["p_absa"])

    def get_candidates_by_category(self, query_product_id: str) -> pd.DataFrame:
        query_meta = self._get_product_meta(query_product_id)
        
        cat_value = ""
        cat_key = ""
        if self.cfg.category_level == "lv3":
            cat_value = str(query_meta.get('category_lv3', ''))
            cat_key = f"lv3_{cat_value}"
        elif self.cfg.category_level == "lv2":
            cat_value = str(query_meta.get('category_lv2', ''))
            cat_key = f"lv2_{cat_value}"
        else:
            cat_value = str(query_meta.get('category_id', ''))
            cat_key = f"id_{cat_value}"

        if cat_value and cat_value != 'nan' and cat_key in self.category_to_products:
            candidate_ids = self.category_to_products[cat_key]
        else:
            if self.cfg.category_level == "lv3":
                candidate_df = self.products_df[self.products_df['category_lv3'].astype(str) == cat_value]
            elif self.cfg.category_level == "lv2":
                candidate_df = self.products_df[self.products_df['category_lv2'].astype(str) == cat_value]
            else:
                candidate_df = self.products_df[self.products_df['category_id'].astype(str) == cat_value]
            candidate_ids = candidate_df['product_id'].astype(str).tolist()

        candidate_ids = [pid for pid in candidate_ids if pid != str(query_product_id)]
        candidates = self.products_df[self.products_df['product_id'].astype(str).isin(candidate_ids)].copy()
        
        # Quantity sold
        if not self.listing_df.empty and 'quantity_sold' in self.listing_df.columns:
            listing_info = self.listing_df[['product_id', 'quantity_sold']].drop_duplicates(subset=['product_id'])
            listing_info['quantity_sold'] = pd.to_numeric(listing_info['quantity_sold'], errors='coerce').fillna(0)
            candidates = candidates.merge(listing_info, on='product_id', how='left')
            candidates['quantity_sold'] = candidates['quantity_sold'].fillna(0)
        else:
            candidates['quantity_sold'] = 0

        # Review count - FIXED: create column safely
        review_counts = self.reviews_df.groupby('product_id').size().reset_index(name='review_count')
        candidates = candidates.merge(review_counts, on='product_id', how='left')
        
        # Ensure column exists
        if 'review_count' not in candidates.columns:
            candidates['review_count'] = 0
        else:
            candidates['review_count'] = candidates['review_count'].fillna(0)

        max_sold = candidates['quantity_sold'].max() if candidates['quantity_sold'].max() > 0 else 1
        max_review = candidates['review_count'].max() if candidates['review_count'].max() > 0 else 1
        
        if 'rating_average' in candidates.columns:
            rating_values = pd.to_numeric(candidates['rating_average'], errors='coerce').fillna(0)
        else:
            rating_values = pd.Series([0] * len(candidates), index=candidates.index)
        
        candidates['popularity_score'] = (
            0.4 * (candidates['quantity_sold'] / max_sold) +
            0.3 * (candidates['review_count'] / max_review) +
            0.3 * (rating_values / 5)
        )

        candidates = candidates.sort_values('popularity_score', ascending=False).head(self.cfg.candidate_pool_size)
        return candidates

    def compute_cbf_scores(self, query_product_id: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
        pid_list = self._tfidf_pid_list
        idx_map = {pid: i for i, pid in enumerate(pid_list)}
        q_idx = idx_map.get(str(query_product_id))

        if q_idx is None:
            candidates_df['cbf_score'] = 0.0
            return candidates_df

        candidate_ids = candidates_df['product_id'].astype(str).tolist()
        valid_candidates = [(pid, idx_map.get(pid)) for pid in candidate_ids if idx_map.get(pid) is not None]
        
        if not valid_candidates:
            candidates_df['cbf_score'] = 0.0
            return candidates_df

        candidate_indices = [idx for _, idx in valid_candidates]
        candidate_matrix = self._tfidf_matrix[candidate_indices]
        query_vector = self._tfidf_matrix[q_idx]
        sims = cosine_similarity(query_vector, candidate_matrix).flatten()

        score_map = {pid: float(sim) for (pid, _), sim in zip(valid_candidates, sims)}
        candidates_df['cbf_score'] = candidates_df['product_id'].astype(str).map(score_map).fillna(0.0)
        return candidates_df

    def compute_hybrid_scores(self, query_reviews: pd.DataFrame, candidates_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        n_reviews = len(query_reviews)
        noise_ratio = query_reviews['noise_score'].mean() if not query_reviews.empty else 0.5

        if n_reviews >= 20:
            w_absa, w_cbf, w_cat = 0.50, 0.20, 0.30
        elif n_reviews >= 10:
            w_absa, w_cbf, w_cat = 0.40, 0.30, 0.30
        elif n_reviews >= 5:
            w_absa, w_cbf, w_cat = 0.30, 0.40, 0.30
        else:
            w_absa, w_cbf, w_cat = 0.20, 0.50, 0.30

        if noise_ratio > 0.6:
            w_absa *= 0.5
            w_cbf *= 1.2

        total = w_absa + w_cbf + w_cat
        w_absa, w_cbf, w_cat = w_absa/total, w_cbf/total, w_cat/total

        candidates_df['hybrid_score'] = (
            w_absa * candidates_df.get('absa_score', 0.5) +
            w_cbf * candidates_df.get('cbf_score', 0.0) +
            w_cat * candidates_df.get('category_match_score', 1.0)
        )

        return candidates_df, {"absa": w_absa, "cbf": w_cbf, "category": w_cat}

    def run(self, product_url: Optional[str] = None, product_id: Optional[str] = None, reviews_source: str = "auto") -> Dict[str, Any]:
        if not product_id:
            if not product_url:
                raise ValueError("Please provide product_url or product_id")
            product_id = extract_product_id_from_url(product_url)
            if not product_id:
                raise ValueError("Cannot extract product_id from URL")

        product_id = str(product_id)
        print(f"\n🎯 Query product: {product_id}")

        query_meta = self._get_product_meta(product_id)
        query_reviews = self.load_reviews(product_id, reviews_source=reviews_source)
        query_reviews = self.preprocess_reviews(query_reviews)

        print(f"📊 Query reviews: {len(query_reviews)} (noisy: {query_reviews['is_noisy'].sum() if not query_reviews.empty else 0})")

        candidates = self.get_candidates_by_category(product_id)
        print(f"📋 Candidates from category: {len(candidates)}")

        if candidates.empty:
            return {"error": "No candidates found"}

        candidates = self.compute_cbf_scores(product_id, candidates)
        print("⚡ Computing ABSA scores for candidates...")
        candidates['absa_score'] = candidates['product_id'].apply(self._get_absa_cached)
        candidates['category_match_score'] = 1.0

        candidates, weights = self.compute_hybrid_scores(query_reviews, candidates)
        ranked = candidates.sort_values('hybrid_score', ascending=False).reset_index(drop=True)
        top_k_df = ranked.head(self.cfg.top_k).copy()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.cfg.output_root) / f"product_{product_id}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        top_k_df.to_csv(out_dir / "top_k_products.csv", index=False, encoding="utf-8-sig")
        
        summary = {
            "input": {
                "product_id": product_id,
                "product_name": _safe_str(query_meta.get('name', '')),
                "category_lv1": _safe_str(query_meta.get('category_lv1', '')),
                "category_lv2": _safe_str(query_meta.get('category_lv2', '')),
                "category_lv3": _safe_str(query_meta.get('category_lv3', '')),
            },
            "query_stats": {
                "n_reviews": len(query_reviews),
                "n_noisy": int(query_reviews['is_noisy'].sum()) if not query_reviews.empty else 0,
                "noise_ratio": float(query_reviews['noise_score'].mean()) if not query_reviews.empty else 1.0,
            },
            "weights": weights,
            "config": {"category_level": self.cfg.category_level, "candidate_pool_size": self.cfg.candidate_pool_size, "top_k": self.cfg.top_k},
            "top_k": top_k_df[['product_id', 'name', 'hybrid_score', 'absa_score', 'cbf_score']].head(10).to_dict('records'),
            "output_dir": str(out_dir),
        }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Improved Hybrid Recommender")
    parser.add_argument("--product_url", type=str, default=None, help="Tiki product URL")
    parser.add_argument("--product_id", type=str, default=None, help="Product ID")
    parser.add_argument("--reviews_source", type=str, default="auto", choices=["auto", "local", "live"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--candidate_pool", type=int, default=150)
    parser.add_argument("--category_level", type=str, default="lv3", choices=["lv2", "lv3", "id"])
    
    args = parser.parse_args()

    config = ImprovedHybridConfig(
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool,
        category_level=args.category_level,
    )

    predictor = ImprovedPhoBERTPredictor(max_len=config.max_len)
    recommender = ImprovedHybridRecommender(config, predictor)

    result = recommender.run(
        product_url=args.product_url,
        product_id=args.product_id,
        reviews_source=args.reviews_source,
    )

    print("\n" + "="*60)
    print("📊 RECOMMENDATION RESULTS")
    print("="*60)
    print(f"Product: {result['input'].get('product_name', 'N/A')}")
    print(f"Category: {result['input'].get('category_lv3', 'N/A')}")
    print(f"Weights: ABSA={result['weights']['absa']:.2f}, CBF={result['weights']['cbf']:.2f}, Category={result['weights']['category']:.2f}")
    print(f"\nTop {args.top_k} recommendations saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()