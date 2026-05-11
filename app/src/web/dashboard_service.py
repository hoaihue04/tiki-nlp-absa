from __future__ import annotations

import json
import re
import sys
import time
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F

# ─── Path fix ────────────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_APP_SRC   = _THIS_FILE.parent          # app/src/web/
_APP_DIR   = _APP_SRC.parent.parent     # app/
_ROOT_DIR  = _APP_DIR.parent            # TIKI/

if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

ProgressCallback = Callable[[int, str], None]


@dataclass
class DashboardConfig:
    max_reviews_for_dashboard: int = 200
    recommendation_top_k: int = 5
    candidate_pool: int = 100
    phobert_batch_size: int = 32
    category_level: str = "lv3"
    confidence_threshold: float = 0.6
    # Live-crawl settings
    reviews_per_page: int = 20
    max_review_pages: int = 15        # tối đa 300 reviews / sản phẩm
    request_timeout: int = 10
    inter_page_sleep: Tuple[float, float] = (0.5, 1.2)


# ─── Constants ───────────────────────────────────────────────────────────────
CATEGORIES: List[str] = [
    "PRODUCT#QUALITY", "PRODUCT#MATERIAL", "PRODUCT#COMFORT",
    "PRODUCT#SIZE", "PRODUCT#DESIGN", "PRODUCT#SAFETY",
    "PRODUCT#FUNCTION", "PRODUCT#DURABILITY", "PRODUCT#VALUE",
    "PRICE#AFFORDABILITY", "PRICE#DISCOUNT",
    "DELIVERY#SPEED", "DELIVERY#PACKAGING", "DELIVERY#ACCURACY",
    "SELLER#SERVICE", "SELLER#RESPONSIVENESS", "SELLER#AUTHENTICITY",
]

ASPECT_REQUIRED_KEYWORDS = {
    "PRODUCT#QUALITY": ["chất lượng","hàng","sản phẩm","tốt","ngon","dở","tệ","chất","ổn","ok","kém","tuyệt"],
    "PRODUCT#MATERIAL": ["chất liệu","vải","cotton","len","sợi","da","nhựa","lông","nỉ","polyester","mỏng","dày"],
    "PRODUCT#COMFORT": ["thoải mái","dễ chịu","mềm","êm","khó chịu","cứng","cảm giác","thoáng","ấm"],
    "PRODUCT#SIZE": ["size","kích thước","to","nhỏ","vừa","chật","rộng","lớn","bé","cỡ","fit","vừa vặn","vừa người"],
    "PRODUCT#DESIGN": ["thiết kế","mẫu","kiểu dáng","đẹp","xinh","xấu","màu","màu sắc","họa tiết","hoa văn","kiểu"],
    "PRODUCT#SAFETY": ["an toàn","bảo vệ","nguy hiểm","độc hại","chất độc","bpa","an toàn cho bé","không độc"],
    "PRODUCT#FUNCTION": ["chức năng","công dụng","dùng","sử dụng","tiện","hữu ích","tác dụng","tiện lợi","dễ dùng"],
    "PRODUCT#DURABILITY": ["bền","chắc","hỏng","rách","xước","mòn","lâu","chắc chắn","bền bỉ"],
    "PRODUCT#VALUE": ["giá trị","đáng tiền","tiền nào","xứng đáng","hợp lý","đáng mua","xứng"],
    "PRICE#AFFORDABILITY": ["giá","rẻ","đắt","mắc","tiền","bao nhiêu","giá cả","giá tiền","giá bán"],
    "PRICE#DISCOUNT": ["giảm giá","khuyến mãi","sale","ưu đãi","voucher","giảm","freeship","coupon","mã giảm"],
    "DELIVERY#SPEED": ["giao hàng","nhanh","chậm","vận chuyển","ship","giao","giao nhanh","nhận hàng","giao chậm"],
    "DELIVERY#PACKAGING": ["đóng gói","bao bì","hộp","túi","bọc","kỹ","cẩn thận","thùng","đóng hàng","bao bì"],
    "DELIVERY#ACCURACY": ["chính xác","sai","thiếu","thừa","đúng","lộn","giao đúng","đủ hàng","giao sai","giao nhầm"],
    "SELLER#SERVICE": ["shop","người bán","tư vấn","nhiệt tình","hỗ trợ","chăm sóc","thái độ","dịch vụ"],
    "SELLER#RESPONSIVENESS": ["phản hồi","trả lời","chat","liên lạc","nhanh nhẹn","rep","reply","hồi âm"],
    "SELLER#AUTHENTICITY": ["chính hãng","hàng thật","fake","nhái","xịn","giả","tem mác","hàng giả","hàng nhái"],
}

ASPECT_FORBIDDEN_KEYWORDS = {
    "PRODUCT#SIZE": ["giao","ship","vận chuyển","giá","rẻ","đắt","shop","người bán","đóng gói","phản hồi"],
    "PRODUCT#QUALITY": ["giao","ship","vận chuyển","size","kích thước","shop","giảm giá","sale"],
    "PRODUCT#MATERIAL": ["giao","ship","size","giá rẻ","giá đắt","shop","giao hàng"],
    "PRODUCT#COMFORT": ["giao","ship","giá","size","kích thước","đóng gói"],
    "PRODUCT#DESIGN": ["giao","ship","giá","size","kích thước","đóng gói","phản hồi"],
    "PRODUCT#FUNCTION": ["giao","ship","size","giá","đóng gói"],
    "PRODUCT#DURABILITY": ["giao","ship","size","giá","đóng gói"],
    "PRODUCT#VALUE": ["giao","ship","đóng gói","size"],
    "DELIVERY#SPEED": ["chất lượng","chất liệu","size","thiết kế","màu","đẹp","xấu","mềm","cứng","bền"],
    "DELIVERY#PACKAGING": ["chất lượng","size","thiết kế","màu","mềm","cứng"],
    "DELIVERY#ACCURACY": ["chất lượng","size","thiết kế","màu","mềm"],
    "PRICE#AFFORDABILITY": ["giao","ship","đóng gói","chất liệu","size","thiết kế"],
    "PRICE#DISCOUNT": ["giao","ship","chất lượng","size","chất liệu"],
    "SELLER#SERVICE": ["giao","ship","vận chuyển","size","chất liệu","thiết kế"],
    "SELLER#RESPONSIVENESS": ["giao","ship","size","chất liệu","thiết kế"],
    "SELLER#AUTHENTICITY": ["giao","ship","size","chất liệu","thiết kế"],
}

ASPECT_DISPLAY_NAMES = {
    "PRODUCT#SIZE": "Kích thước / độ vừa vặn",
    "PRODUCT#QUALITY": "Chất lượng tổng thể",
    "PRODUCT#MATERIAL": "Chất liệu",
    "PRODUCT#COMFORT": "Cảm giác sử dụng",
    "PRODUCT#DESIGN": "Thiết kế & kiểu dáng",
    "PRODUCT#SAFETY": "Độ an toàn",
    "PRODUCT#FUNCTION": "Tính năng & công dụng",
    "PRODUCT#DURABILITY": "Độ bền",
    "PRODUCT#VALUE": "Giá trị so với giá tiền",
    "PRICE#AFFORDABILITY": "Mức giá",
    "PRICE#DISCOUNT": "Ưu đãi & khuyến mãi",
    "DELIVERY#SPEED": "Tốc độ giao hàng",
    "DELIVERY#PACKAGING": "Chất lượng đóng gói",
    "DELIVERY#ACCURACY": "Độ chính xác đơn hàng",
    "SELLER#SERVICE": "Chất lượng dịch vụ của shop",
    "SELLER#RESPONSIVENESS": "Mức độ phản hồi của shop",
    "SELLER#AUTHENTICITY": "Độ tin cậy / chính hãng",
}

ASPECT_ICONS = {
    "PRODUCT#QUALITY": "🧵", "PRODUCT#MATERIAL": "🪡", "PRODUCT#COMFORT": "😌",
    "PRODUCT#SIZE": "📐", "PRODUCT#DESIGN": "🎨", "PRODUCT#SAFETY": "🛡️",
    "PRODUCT#FUNCTION": "⚙️", "PRODUCT#DURABILITY": "💪", "PRODUCT#VALUE": "💰",
    "PRICE#AFFORDABILITY": "🏷️", "PRICE#DISCOUNT": "🎫",
    "DELIVERY#SPEED": "🚚", "DELIVERY#PACKAGING": "📦", "DELIVERY#ACCURACY": "✅",
    "SELLER#SERVICE": "🛒", "SELLER#RESPONSIVENESS": "💬", "SELLER#AUTHENTICITY": "🔐",
}

SENTIMENT_NAMES = ["none", "positive", "neutral", "negative"]
SENTIMENT_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# ─── HTTP headers giả lập browser ────────────────────────────────────────────
_TIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi,en;q=0.9",
    "Referer": "https://tiki.vn/",
    "Origin": "https://tiki.vn",
}

# ─── In-memory cache để không crawl lại cùng sản phẩm trong 1 session ────────
_product_info_cache: Dict[str, Dict] = {}
_reviews_cache: Dict[str, List[Dict]] = {}

# ─── CSV caches (dùng cho recommendation candidates) ─────────────────────────
_detail_df_cache: Optional[pd.DataFrame] = None
_listing_df_cache: Optional[pd.DataFrame] = None
_cat_df_cache: Optional[pd.DataFrame] = None
_reviews_df_cache: Optional[pd.DataFrame] = None  # fallback only
_asqp_df_cache: Optional[pd.DataFrame] = None      # asqp_annotated_flat.csv


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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


def _get_listing_df() -> pd.DataFrame:
    global _listing_df_cache
    if _listing_df_cache is None:
        _listing_df_cache = _load_csv_safe("Tiki_be_listing.csv")
    return _listing_df_cache


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

    # Ưu tiên text dài hơn (thường chứa nhiều thông tin hơn)
    pool = pool.copy()
    pool["_len"] = pool["text_preview"].str.len()
    pool = pool[pool["_len"] >= 15].sort_values("_len", ascending=False)
    if pool.empty:
        return ""

    # Lấy ngẫu nhiên trong top 10 để có đa dạng
    candidates = pool.head(10)["text_preview"].tolist()
    return random.choice(candidates)[:max_len]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = str(val).strip()
        return float(v) if v and v not in ("nan", "", "None") else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        v = str(val).strip()
        return int(float(v)) if v and v not in ("nan", "", "None") else default
    except (ValueError, TypeError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    v = str(val).strip()
    return v if v and v not in ("nan", "None") else default


def extract_product_id_from_url(product_url: str) -> Optional[str]:
    from urllib.parse import urlparse, parse_qs
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


def _sentiment_label_from_score(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _advice_from_score(score: float) -> Dict[str, str]:
    if score >= 0.6:
        return {"label": "Nên mua", "tone": "positive"}
    if score >= 0.3:
        return {"label": "Cân nhắc", "tone": "neutral"}
    return {"label": "Thận trọng", "tone": "negative"}


# ══════════════════════════════════════════════════════════════════════════════
# LIVE TIKI API CRAWLERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_request(url: str, params: Optional[Dict] = None,
                  timeout: int = 10, max_retry: int = 3) -> Optional[requests.Response]:
    """GET với retry + exponential back-off."""
    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.get(url, headers=_TIKI_HEADERS,
                                params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            print(f"[HTTP] {url} → {resp.status_code} (attempt {attempt})")
        except requests.RequestException as exc:
            print(f"[HTTP] {url} error (attempt {attempt}): {exc}")
        time.sleep(1.5 * attempt)
    return None


def fetch_tiki_product_info_live(product_id: str) -> Dict[str, Any]:
    """
    Lấy thông tin sản phẩm TRỰC TIẾP từ Tiki API /api/v2/products/{id}.
    Trả về dict đồng nhất với cấu trúc cũ để không thay đổi downstream code.
    """
    if product_id in _product_info_cache:
        return _product_info_cache[product_id]

    info: Dict[str, Any] = {
        "product_id": product_id,
        "name": f"Sản phẩm #{product_id}",
        "price": 0,
        "original_price": 0,
        "brand_name": "",
        "seller_name": "",
        "seller_is_official": False,
        "rating_average": 0.0,
        "review_count": 0,
        "thumbnail_url": "",
        "images": [],
        "product_url": f"https://tiki.vn/p/{product_id}",
        "category_lv1": "",
        "category_lv2": "",
        "category_lv3": "",
        "sold_quantity": 0,
        "short_description": "",
    }

    print(f"[LiveAPI] Fetching product info: product_id={product_id}")
    resp = _safe_request(
        f"https://tiki.vn/api/v2/products/{product_id}",
        params={"platform": "web"},
    )
    if resp is None:
        print(f"[LiveAPI] Cannot fetch product {product_id} — using defaults")
        _product_info_cache[product_id] = info
        return info

    try:
        data = resp.json()
    except Exception as e:
        print(f"[LiveAPI] JSON parse error: {e}")
        _product_info_cache[product_id] = info
        return info

    # ── Basic fields ────────────────────────────────────────────────────────
    info["name"] = data.get("name") or info["name"]
    info["price"] = int(data.get("price") or 0)
    info["original_price"] = int(data.get("list_price") or data.get("price") or 0)
    info["rating_average"] = float(data.get("rating_average") or 0.0)
    info["review_count"] = int(data.get("review_count") or 0)
    info["short_description"] = data.get("short_description") or ""

    # ── Brand & Seller ──────────────────────────────────────────────────────
    brand = data.get("brand") or {}
    info["brand_name"] = brand.get("name") or ""

    seller = data.get("current_seller") or {}
    info["seller_name"] = seller.get("name") or ""
    info["seller_is_official"] = bool(seller.get("is_official", False))

    # ── Sold quantity ───────────────────────────────────────────────────────
    qty_sold = data.get("quantity_sold") or {}
    if isinstance(qty_sold, dict):
        info["sold_quantity"] = int(qty_sold.get("value") or 0)
    else:
        info["sold_quantity"] = int(qty_sold or 0)

    # ── Images ──────────────────────────────────────────────────────────────
    thumbnail = data.get("thumbnail_url") or ""
    images_raw = data.get("images") or []
    images = [img.get("base_url", "") for img in images_raw if img.get("base_url")]

    if not thumbnail and images:
        thumbnail = images[0]
    info["thumbnail_url"] = thumbnail
    info["images"] = images[:6]

    # ── Product URL ─────────────────────────────────────────────────────────
    short_url = data.get("short_url") or ""
    if short_url.startswith("https://tiki.vn/"):
        info["product_url"] = short_url

    # ── Categories ──────────────────────────────────────────────────────────
    breadcrumbs = data.get("breadcrumbs") or []
    if len(breadcrumbs) >= 1:
        info["category_lv1"] = breadcrumbs[0].get("name", "")
    if len(breadcrumbs) >= 2:
        info["category_lv2"] = breadcrumbs[1].get("name", "")
    if len(breadcrumbs) >= 3:
        info["category_lv3"] = breadcrumbs[2].get("name", "")

    # Fallback: thử categories field
    if not info["category_lv1"]:
        categories = data.get("categories") or {}
        if isinstance(categories, dict):
            info["category_lv1"] = categories.get("name", "")

    print(
        f"[LiveAPI] Product OK: '{info['name']}' | "
        f"price={info['price']:,}đ | rating={info['rating_average']} | "
        f"reviews={info['review_count']}"
    )

    _product_info_cache[product_id] = info
    return info


def fetch_tiki_reviews_live(
    product_id: str,
    max_pages: int = 15,
    per_page: int = 20,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Dict]:
    """
    Crawl toàn bộ reviews của sản phẩm qua Tiki API /api/v2/reviews.
    Hỗ trợ phân trang và progress reporting.
    """
    if product_id in _reviews_cache:
        print(f"[LiveAPI] Reviews cache hit for {product_id}: {len(_reviews_cache[product_id])} reviews")
        return _reviews_cache[product_id]

    print(f"[LiveAPI] Crawling reviews for product_id={product_id} (max {max_pages} pages)...")
    all_reviews: List[Dict] = []
    base_url = "https://tiki.vn/api/v2/reviews"

    for page in range(1, max_pages + 1):
        params = {
            "product_id": product_id,
            "page": page,
            "limit": per_page,
            "sort": "score|desc",
            "platform": "web",
        }

        resp = _safe_request(base_url, params=params)
        if resp is None:
            print(f"[LiveAPI] Page {page} failed — stopping")
            break

        try:
            body = resp.json()
        except Exception as e:
            print(f"[LiveAPI] JSON error page {page}: {e}")
            break

        items = body.get("data") or []
        if not items:
            print(f"[LiveAPI] No more items at page {page}")
            break

        for rv in items:
            creator = rv.get("created_by") or {}
            created_ts = rv.get("created_at")
            purchased_ts = creator.get("purchased_at")

            all_reviews.append({
                "review_id": str(rv.get("id", "")),
                "product_id": product_id,
                "customer_id": str(creator.get("id", "")),
                "customer_name": creator.get("name") or "Ẩn danh",
                "rating": int(rv.get("rating") or 3),
                "title": rv.get("title") or "",
                "content": rv.get("content") or "",
                "helpful_count": int(rv.get("thank_count") or 0),
                "is_verified": bool(purchased_ts),
                "images_count": len(rv.get("images") or []),
                "created_at_ts": created_ts,
            })

        pct_base = 20 + int(10 * page / max_pages)
        if progress_callback:
            try:
                progress_callback(
                    pct_base,
                    f"Thu thập đánh giá trang {page} — {len(all_reviews)} reviews..."
                )
            except Exception:
                pass

        # Kiểm tra có trang tiếp theo không
        paging = body.get("paging") or {}
        total_pages = paging.get("last_page") or paging.get("total_pages") or page
        if page >= total_pages:
            print(f"[LiveAPI] All pages fetched ({page}/{total_pages})")
            break

        # Throttle để không bị block
        time.sleep(random.uniform(0.5, 1.2))

    print(f"[LiveAPI] Total reviews crawled: {len(all_reviews)}")
    _reviews_cache[product_id] = all_reviews
    return all_reviews


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


# ══════════════════════════════════════════════════════════════════════════════
# PHOBERT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class RealPhoBERTPredictor:
    def __init__(self, max_len: int = 256):
        self.max_len = max_len
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self.cache: Dict[str, List[Dict]] = {}

    def _load_model(self):
        if self._loaded:
            return
        print("[PhoBERT] Loading tokenizer and model...")
        model_paths = [
            _ROOT_DIR / "models" / "phobert" / "best_model.pt",
            Path("models/phobert/best_model.pt"),
        ]
        model_path = next((p for p in model_paths if p.exists()), None)

        try:
            from transformers import AutoTokenizer
            local_tok = _ROOT_DIR / "models" / "phobert" / "base_model"
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(local_tok) if local_tok.exists() else "vinai/phobert-base-v2"
            )
            print("[PhoBERT] Tokenizer loaded")
        except Exception as e:
            print(f"[PhoBERT] Tokenizer load failed: {e}")
            self.tokenizer = None

        if model_path:
            try:
                from src.training.train_phobert import PhoBERTASQP, Config
                cfg = Config()
                cfg.MAX_LEN = self.max_len
                ckpt = torch.load(model_path, map_location=self.device)
                self.model = PhoBERTASQP(cfg).to(self.device)
                state = ckpt.get("model_state", ckpt)
                self.model.load_state_dict(state)
                self.model.eval()
                print(f"[PhoBERT] Model loaded from {model_path}")
            except Exception as e:
                print(f"[PhoBERT] Model load failed: {e}")
                self.model = None

        self._loaded = True

    def predict_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        self._load_model()
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            return []
        if sentence in self.cache:
            return self.cache[sentence]
        if self.model is not None and self.tokenizer is not None:
            try:
                result = self._predict_with_model(sentence)
                self.cache[sentence] = result
                return result
            except Exception as e:
                print(f"[PhoBERT] Inference error: {e}")
        result = self._rule_based_predict(sentence)
        self.cache[sentence] = result
        return result

    def _predict_with_model(self, sentence: str) -> List[Dict[str, Any]]:
        enc = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits_per_cat = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
        outputs = []
        for cat_idx, logits in enumerate(logits_per_cat):
            probs = F.softmax(logits[0], dim=-1)
            pred_idx = int(torch.argmax(probs).item())
            pred_name = SENTIMENT_NAMES[pred_idx]
            conf = float(probs[pred_idx].item())
            if pred_name != "none" and conf >= 0.5:
                outputs.append({
                    "aspect": CATEGORIES[cat_idx],
                    "sentiment": pred_name,
                    "confidence": conf,
                    "sentiment_score": SENTIMENT_TO_SCORE[pred_name],
                })
        return outputs

    def _rule_based_predict(self, sentence: str) -> List[Dict[str, Any]]:
        sl = sentence.lower()
        pos_kws = ["tốt","hay","đẹp","chất lượng","tuyệt","ổn","hài lòng","thích","ưng",
                   "ngon","bền","đáng tiền","mềm","dễ chịu","thoải mái","tiện lợi",
                   "an toàn","chắc chắn","rẻ","hợp lý"]
        neg_kws = ["tệ","hỏng","rách","xấu","kém","không tốt","chán","thất vọng",
                   "mỏng","cứng","khó chịu","đắt","chậm","không hài lòng","lỗi","tồi","bẩn"]
        pos_score = sum(1 for kw in pos_kws if kw in sl)
        neg_score = sum(1 for kw in neg_kws if kw in sl)
        if pos_score > neg_score:
            sentiment, score = "positive", 0.7
        elif neg_score > pos_score:
            sentiment, score = "negative", -0.7
        else:
            sentiment, score = "neutral", 0.0
        confidence = min(0.8, 0.5 + abs(pos_score - neg_score) / 10)
        results = []
        for aspect, keywords in ASPECT_REQUIRED_KEYWORDS.items():
            if not any(kw in sl for kw in keywords):
                continue
            forbidden = ASPECT_FORBIDDEN_KEYWORDS.get(aspect, [])
            if any(kw in sl for kw in forbidden):
                continue
            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": confidence,
                "sentiment_score": score,
            })
            if len(results) >= 4:
                break
        return results


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD SERVICE
# ══════════════════════════════════════════════════════════════════════════════

class DashboardService:
    def __init__(self):
        self.config = DashboardConfig()
        self._predictor: Optional[RealPhoBERTPredictor] = None

    def _get_predictor(self) -> RealPhoBERTPredictor:
        if self._predictor is None:
            self._predictor = RealPhoBERTPredictor(max_len=256)
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
        for i, meta in enumerate(sentence_meta):
            preds = predictor.predict_sentence(meta["sentence"])
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
            if total > 0 and (i + 1) % 30 == 0:
                pct = 45 + int(35 * (i + 1) / total)
                self._safe_progress(
                    progress_callback, pct,
                    f"Đang phân tích câu {i + 1}/{total}..."
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
        import random

        # ── Tầng 1: ASQP CSV — nguồn đáng tin cậy nhất (LLM-labeled) ──────────
        asqp_quote = _pick_quote_from_asqp(aspect, sentiment, max_len)
        if asqp_quote:
            return asqp_quote

        # ── Tầng 2: PhoBERT pool với keyword filter (fallback) ─────────────────
        pool = [t for t in tuples if t.get("aspect") == aspect and t.get("sentiment") == sentiment]
        if not pool:
            return ""
        required_kws = ASPECT_REQUIRED_KEYWORDS.get(aspect, [])
        forbidden_kws = ASPECT_FORBIDDEN_KEYWORDS.get(aspect, [])

        def score_item(t):
            return (t.get("helpful_count", 0), t.get("confidence", 0), len(t.get("sentence", "")))

        # Tier 2a: có required keyword VÀ không có forbidden keyword
        tier1 = [t for t in pool
                 if required_kws and any(kw in t.get("sentence", "").lower() for kw in required_kws)
                 and not any(kw in t.get("sentence", "").lower() for kw in forbidden_kws)]
        if tier1:
            return re.sub(r"\s+", " ", random.choice(sorted(tier1, key=score_item, reverse=True)[:5]).get("sentence", "")).strip()[:max_len]

        # Tier 2b: có required keyword (nới lỏng forbidden)
        tier2 = [t for t in pool
                 if required_kws and any(kw in t.get("sentence", "").lower() for kw in required_kws)]
        if tier2:
            return re.sub(r"\s+", " ", random.choice(sorted(tier2, key=score_item, reverse=True)[:5]).get("sentence", "")).strip()[:max_len]

        # Không có câu nào khớp → trả rỗng, không hiển thị quote lệch aspect
        return 

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
        import random
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

    def _get_candidates_by_category(self, product_id: str, product_info: Dict) -> List[Dict]:
        """
        Tìm sản phẩm tương tự từ CSV (recommendation candidates).
        Nếu CSV trống → trả về list rỗng (không crash).
        """
        category_lv3 = product_info.get("category_lv3", "")
        category_lv2 = product_info.get("category_lv2", "")
        if not category_lv3 and not category_lv2:
            print("[Recommend] No category info — skipping recommendations")
            return []
        cat_df = _get_cat_df()
        if cat_df.empty:
            print("[Recommend] cat_df empty — skipping")
            return []
        cat_products = cat_df[
            (cat_df.get("category_lv3", pd.Series(dtype=str)).astype(str) == category_lv3) |
            (cat_df.get("category_lv2", pd.Series(dtype=str)).astype(str) == category_lv2)
        ]
        candidate_ids = cat_products["product_id"].astype(str).tolist()
        candidate_ids = [pid for pid in candidate_ids if pid != str(product_id)]
        candidates = []
        for pid in candidate_ids[:self.config.candidate_pool]:
            info = _get_product_info_from_csv_fallback(pid)
            if info["name"] and info["name"] != f"Sản phẩm #{pid}":
                # Dùng rating từ CSV (không live-crawl để tránh quá nhiều requests)
                absa_score = info.get("rating_average", 3.0) / 5.0
                candidates.append({
                    "product_id": str(pid),
                    "name": str(info["name"]),
                    "price": int(info["price"]) if info["price"] else None,
                    "rating_average": float(info.get("rating_average", 0)),
                    "thumbnail_url": str(info.get("thumbnail_url", "")),
                    "absa_score": round(absa_score, 4),
                    "sold_quantity": int(info.get("sold_quantity", 0)),
                })
        candidates.sort(key=lambda x: (x.get("rating_average", 0), x.get("sold_quantity", 0)), reverse=True)
        return candidates[:self.config.recommendation_top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_product(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:

        # ── 1. Trích xuất product_id ─────────────────────────────────────────
        self._safe_progress(progress_callback, 3, "Trích xuất product ID từ URL...")
        if not product_id and product_url:
            product_id = extract_product_id_from_url(product_url)
        if not product_id:
            raise ValueError("Không thể xác định product ID từ URL đã nhập")
        product_id = str(product_id).strip()
        print(f"\n[Analysis] ═══ Starting for product_id={product_id} ═══")

        # ── 2. Live crawl thông tin sản phẩm ────────────────────────────────
        self._safe_progress(progress_callback, 8, "Đang lấy thông tin sản phẩm từ Tiki...")
        product_info = fetch_tiki_product_info_live(product_id)

        # Nếu API fail → thử CSV fallback
        if product_info["name"] == f"Sản phẩm #{product_id}":
            print("[Analysis] API returned empty — trying CSV fallback for product info")
            csv_info = _get_product_info_from_csv_fallback(product_id)
            if csv_info["name"] != f"Sản phẩm #{product_id}":
                product_info.update(csv_info)

        self._safe_progress(progress_callback, 15, f"Tìm thấy: {product_info['name'][:40]}...")

        # ── 3. Live crawl reviews ────────────────────────────────────────────
        self._safe_progress(progress_callback, 18, "Đang thu thập đánh giá từ Tiki API...")
        reviews = fetch_tiki_reviews_live(
            product_id,
            max_pages=self.config.max_review_pages,
            per_page=self.config.reviews_per_page,
            progress_callback=progress_callback,
        )

        # Nếu live crawl ra 0 review → fallback CSV
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

        # ── 4. Tiền xử lý ───────────────────────────────────────────────────
        reviews_df = self._preprocess_reviews(reviews)
        if reviews_df.empty:
            raise ValueError("Không có đánh giá hợp lệ sau tiền xử lý văn bản")
        reviews_df = reviews_df.head(self.config.max_reviews_for_dashboard)
        print(f"[Analysis] Using {len(reviews_df)} reviews for ABSA")

        # ── 5. PhoBERT prediction ────────────────────────────────────────────
        self._safe_progress(progress_callback, 40, "Đang phân tích cảm xúc với PhoBERT...")
        tuples, review_scores, sentiment_bucket = self._batch_predict(reviews_df, progress_callback)
        if not tuples:
            raise ValueError("PhoBERT không trả về kết quả — đánh giá quá ngắn hoặc không có nội dung")

        # ── 6. Tổng hợp kết quả ─────────────────────────────────────────────
        self._safe_progress(progress_callback, 82, "Tổng hợp kết quả theo khía cạnh...")
        aspect_payload = self._build_aspect_payload(tuples)
        self._safe_progress(progress_callback, 86, "Trích xuất ý kiến chi tiết...")
        opinion_table = self._build_opinion_table(tuples)
        self._safe_progress(progress_callback, 90, "Tìm sản phẩm gợi ý tương tự...")
        recommendations = self._get_candidates_by_category(product_id, product_info)

        # ── 7. Tính điểm tổng hợp ───────────────────────────────────────────
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
        advice = None
        representative = self._representative_reviews(reviews_df, review_scores)

        # ── 8. Radar so sánh với top1 recommendation ─────────────────────────
        top1_aspect_scores: Dict[str, float] = {}
        if recommendations:
            top1_id = recommendations[0]["product_id"]
            # Dùng CSV fallback cho top1 (không live-crawl để tránh delay dài)
            top1_reviews = _get_reviews_from_csv_fallback(top1_id)
            if top1_reviews:
                top1_df = self._preprocess_reviews(top1_reviews)
                if not top1_df.empty:
                    top1_tuples, _, _ = self._batch_predict(top1_df.head(50), None)
                    if top1_tuples:
                        top1_pred_df = pd.DataFrame(top1_tuples)
                        for asp in top1_pred_df["aspect"].unique():
                            sub = top1_pred_df[top1_pred_df["aspect"] == asp]
                            s = sub["sentiment_score"].mean()
                            top1_aspect_scores[asp] = max(0.0, min(1.0, (s + 1) / 2))

        self._safe_progress(progress_callback, 98, "Đang hoàn thiện báo cáo...")
        print(f"[Analysis] ✅ Complete! ABSA score: {absa_score:.3f} | Reviews: {total_reviews}")

        # ── 9. Build final result dict ────────────────────────────────────────
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
            "recommendations": {
                "top_products": [
                    {
                        "product_id": str(p.get("product_id", "")),
                        "name": str(p.get("name", "")),
                        "price": int(p.get("price", 0)) if p.get("price") else None,
                        "rating_average": float(p.get("rating_average", 0)),
                        "thumbnail_url": str(p.get("thumbnail_url", "")),
                        "absa_score": float(p.get("absa_score", 0)),
                    }
                    for p in recommendations
                ]
            },
            "top1_aspect_scores": top1_aspect_scores if top1_aspect_scores else None,
        }
        return result