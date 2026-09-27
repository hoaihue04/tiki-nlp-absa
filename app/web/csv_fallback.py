from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.web.product_parser import _safe_float, _safe_int, _safe_str

_ROOT_DIR = Path(__file__).resolve().parents[2]
_APP_DIR = _ROOT_DIR / "app"

_detail_df_cache: Optional[pd.DataFrame] = None
_cat_df_cache: Optional[pd.DataFrame] = None
_reviews_df_cache: Optional[pd.DataFrame] = None  # fallback only
_asqp_df_cache: Optional[pd.DataFrame] = None      # asqp_annotated_flat.csv


# HELPER UTILITIES

def _get_data_raw_dir() -> Path:
    candidates = [
        Path("data/raw"),
        _ROOT_DIR / "data" / "raw",
        _APP_DIR / "data" / "raw",
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("data/raw")


def _load_csv_safe(filename: str) -> pd.DataFrame:
    path = _get_data_raw_dir() / filename
    if not path.exists():
        print(f"[CSV] NOT FOUND: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
        print(f"[CSV] Loaded {filename}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[CSV] Error loading {filename}: {e}")
        return pd.DataFrame()


def _get_detail_df() -> pd.DataFrame:
    global _detail_df_cache
    if _detail_df_cache is None:
        _detail_df_cache = _load_csv_safe("Tiki_be_detail.csv")
    return _detail_df_cache


def _get_cat_df() -> pd.DataFrame:
    global _cat_df_cache
    if _cat_df_cache is None:
        _cat_df_cache = _load_csv_safe("Tiki_be_product_id.csv")
    return _cat_df_cache


def _get_reviews_df() -> pd.DataFrame:
    """CSV reviews — chỉ dùng làm fallback."""
    global _reviews_df_cache
    if _reviews_df_cache is None:
        _reviews_df_cache = _load_csv_safe("Tiki_be_reviews.csv")
    return _reviews_df_cache


def _get_asqp_df() -> pd.DataFrame:
    """Load asqp_annotated_flat.csv — nguồn quote đã được LLM gán nhãn chính xác."""
    global _asqp_df_cache
    if _asqp_df_cache is not None:
        return _asqp_df_cache

    candidates = [
        _ROOT_DIR / "data" / "processed" / "asqp_annotated_flat.csv",
        _APP_DIR  / "data" / "processed" / "asqp_annotated_flat.csv",
        Path("data/processed/asqp_annotated_flat.csv"),
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
                print(f"[ASQP] Loaded {path}: {len(df)} rows")
                _asqp_df_cache = df
                return _asqp_df_cache
            except Exception as e:
                print(f"[ASQP] Error loading {path}: {e}")

    print("[ASQP] asqp_annotated_flat.csv not found — quote sẽ lấy từ PhoBERT")
    _asqp_df_cache = pd.DataFrame()
    return _asqp_df_cache


def _pick_quote_from_asqp(aspect: str, sentiment: str, max_len: int = 300) -> str:
    """
    Lấy quote từ asqp_annotated_flat.csv theo aspect + sentiment.
    Trả về text_preview của row khớp, ưu tiên review dài hơn.
    """
    import random
    df = _get_asqp_df()
    if df.empty:
        return ""

    required_cols = {"aspect_category", "sentiment", "text_preview"}
    if not required_cols.issubset(df.columns):
        return ""

    mask = (df["aspect_category"] == aspect) & (df["sentiment"] == sentiment)
    pool = df[mask]
    if pool.empty:
        return ""

    pool = pool.copy()
    pool["_len"] = pool["text_preview"].str.len()
    pool = pool[pool["_len"] >= 15].sort_values("_len", ascending=False)
    if pool.empty:
        return ""

    candidates = pool.head(10)["text_preview"].tolist()
    return random.choice(candidates)[:max_len]


def _get_reviews_from_csv_fallback(product_id: str) -> List[Dict]:
    """Fallback: lấy reviews từ CSV nếu API hoàn toàn thất bại."""
    df = _get_reviews_df()
    if df.empty or "product_id" not in df.columns:
        return []
    rows = df[df["product_id"].astype(str).str.strip() == str(product_id).strip()]
    if rows.empty:
        return []
    result = []
    for _, r in rows.iterrows():
        result.append({
            "review_id": _safe_str(r.get("review_id")),
            "product_id": product_id,
            "customer_name": _safe_str(r.get("customer_name")),
            "rating": _safe_int(r.get("rating"), 0),
            "title": _safe_str(r.get("title")),
            "content": _safe_str(r.get("content")),
            "helpful_count": _safe_int(r.get("helpful_count"), 0),
            "is_verified": _safe_str(r.get("is_verified")),
        })
    print(f"[CSV Fallback] Loaded {len(result)} reviews for {product_id}")
    return result


def _get_product_info_from_csv_fallback(product_id: str) -> Dict[str, Any]:
    """Fallback: lấy product info từ CSV nếu API thất bại."""
    info: Dict[str, Any] = {
        "product_id": product_id,
        "name": f"Sản phẩm #{product_id}",
        "price": 0, "original_price": 0,
        "brand_name": "", "seller_name": "", "seller_is_official": False,
        "rating_average": 0.0, "review_count": 0,
        "thumbnail_url": "", "images": [],
        "product_url": f"https://tiki.vn/p/{product_id}",
        "category_lv1": "", "category_lv2": "", "category_lv3": "",
        "sold_quantity": 0, "short_description": "",
    }
    detail_df = _get_detail_df()
    if not detail_df.empty and "product_id" in detail_df.columns:
        rows = detail_df[detail_df["product_id"].astype(str).str.strip() == str(product_id).strip()]
        if not rows.empty:
            row = rows.iloc[0]
            info["name"] = _safe_str(row.get("name"), info["name"])
            info["price"] = _safe_int(row.get("price"))
            info["original_price"] = _safe_int(row.get("list_price") or row.get("original_price"))
            info["brand_name"] = _safe_str(row.get("brand_name"))
            info["seller_name"] = _safe_str(row.get("seller_name"))
            info["rating_average"] = _safe_float(row.get("rating_average"))
            info["review_count"] = _safe_int(row.get("review_count"))
            seller_official = _safe_str(row.get("seller_is_official", "")).lower()
            info["seller_is_official"] = seller_official in ("true", "1", "yes")
            info["thumbnail_url"] = _safe_str(row.get("thumbnail_url") or row.get("image_url"))
    cat_df = _get_cat_df()
    if not cat_df.empty and "product_id" in cat_df.columns:
        rows = cat_df[cat_df["product_id"].astype(str).str.strip() == str(product_id).strip()]
        if not rows.empty:
            row = rows.iloc[0]
            info["category_lv1"] = _safe_str(row.get("category_lv1"))
            info["category_lv2"] = _safe_str(row.get("category_lv2"))
            info["category_lv3"] = _safe_str(row.get("category_lv3"))
    return info


