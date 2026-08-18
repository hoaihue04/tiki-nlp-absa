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
import torch.nn as nn
import torch.nn.functional as F

from app.src.v2.repository import V2Repository
from app.src.v2.risk import detect_risks
from app.src.v2.vector_store import QdrantOpinionStore, opinion_chunks_from_aspect_items

# â”€â”€â”€ Path fix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    max_review_pages: int = 15        # tá»‘i Ä‘a 300 reviews / sáº£n pháº©m
    request_timeout: int = 10
    inter_page_sleep: Tuple[float, float] = (0.5, 1.2)


# â”€â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

ASPECT_REQUIRED_KEYWORDS = {
    "PRODUCT#QUALITY": ["cháº¥t lÆ°á»£ng","hÃ ng","sáº£n pháº©m","tá»‘t","ngon","dá»Ÿ","tá»‡","cháº¥t","á»•n","ok","kÃ©m","tuyá»‡t"],
    "PRODUCT#MATERIAL": ["cháº¥t liá»‡u","váº£i","cotton","len","sá»£i","da","nhá»±a","lÃ´ng","ná»‰","polyester","má»ng","dÃ y"],
    "PRODUCT#COMFORT": ["thoáº£i mÃ¡i","dá»… chá»‹u","má»m","Ãªm","khÃ³ chá»‹u","cá»©ng","cáº£m giÃ¡c","thoÃ¡ng","áº¥m"],
    "PRODUCT#SIZE": ["size","kÃ­ch thÆ°á»›c","to","nhá»","vá»«a","cháº­t","rá»™ng","lá»›n","bÃ©","cá»¡","fit","vá»«a váº·n","vá»«a ngÆ°á»i"],
    "PRODUCT#DESIGN": ["thiáº¿t káº¿","máº«u","kiá»ƒu dÃ¡ng","Ä‘áº¹p","xinh","xáº¥u","mÃ u","mÃ u sáº¯c","há»a tiáº¿t","hoa vÄƒn","kiá»ƒu"],
    "PRODUCT#SAFETY": ["an toÃ n","báº£o vá»‡","nguy hiá»ƒm","Ä‘á»™c háº¡i","cháº¥t Ä‘á»™c","bpa","an toÃ n cho bÃ©","khÃ´ng Ä‘á»™c"],
    "PRODUCT#FUNCTION": ["chá»©c nÄƒng","cÃ´ng dá»¥ng","dÃ¹ng","sá»­ dá»¥ng","tiá»‡n","há»¯u Ã­ch","tÃ¡c dá»¥ng","tiá»‡n lá»£i","dá»… dÃ¹ng"],
    "PRODUCT#DURABILITY": ["bá»n","cháº¯c","há»ng","rÃ¡ch","xÆ°á»›c","mÃ²n","lÃ¢u","cháº¯c cháº¯n","bá»n bá»‰"],
    "PRODUCT#VALUE": ["giÃ¡ trá»‹","Ä‘Ã¡ng tiá»n","tiá»n nÃ o","xá»©ng Ä‘Ã¡ng","há»£p lÃ½","Ä‘Ã¡ng mua","xá»©ng"],
    "PRICE#AFFORDABILITY": ["giÃ¡","ráº»","Ä‘áº¯t","máº¯c","tiá»n","bao nhiÃªu","giÃ¡ cáº£","giÃ¡ tiá»n","giÃ¡ bÃ¡n"],
    "PRICE#DISCOUNT": ["giáº£m giÃ¡","khuyáº¿n mÃ£i","sale","Æ°u Ä‘Ã£i","voucher","giáº£m","freeship","coupon","mÃ£ giáº£m"],
    "DELIVERY#SPEED": ["giao hÃ ng","nhanh","cháº­m","váº­n chuyá»ƒn","ship","giao","giao nhanh","nháº­n hÃ ng","giao cháº­m"],
    "DELIVERY#PACKAGING": ["Ä‘Ã³ng gÃ³i","bao bÃ¬","há»™p","tÃºi","bá»c","ká»¹","cáº©n tháº­n","thÃ¹ng","Ä‘Ã³ng hÃ ng","bao bÃ¬"],
    "DELIVERY#ACCURACY": ["chÃ­nh xÃ¡c","sai","thiáº¿u","thá»«a","Ä‘Ãºng","lá»™n","giao Ä‘Ãºng","Ä‘á»§ hÃ ng","giao sai","giao nháº§m"],
    "SELLER#SERVICE": ["shop","ngÆ°á»i bÃ¡n","tÆ° váº¥n","nhiá»‡t tÃ¬nh","há»— trá»£","chÄƒm sÃ³c","thÃ¡i Ä‘á»™","dá»‹ch vá»¥"],
    "SELLER#RESPONSIVENESS": ["pháº£n há»“i","tráº£ lá»i","chat","liÃªn láº¡c","nhanh nháº¹n","rep","reply","há»“i Ã¢m"],
    "SELLER#AUTHENTICITY": ["chÃ­nh hÃ£ng","hÃ ng tháº­t","fake","nhÃ¡i","xá»‹n","giáº£","tem mÃ¡c","hÃ ng giáº£","hÃ ng nhÃ¡i"],
}

ASPECT_FORBIDDEN_KEYWORDS = {
    "PRODUCT#SIZE": ["giao","ship","váº­n chuyá»ƒn","giÃ¡","ráº»","Ä‘áº¯t","shop","ngÆ°á»i bÃ¡n","Ä‘Ã³ng gÃ³i","pháº£n há»“i"],
    "PRODUCT#QUALITY": ["giao","ship","váº­n chuyá»ƒn","size","kÃ­ch thÆ°á»›c","shop","giáº£m giÃ¡","sale"],
    "PRODUCT#MATERIAL": ["giao","ship","size","giÃ¡ ráº»","giÃ¡ Ä‘áº¯t","shop","giao hÃ ng"],
    "PRODUCT#COMFORT": ["giao","ship","giÃ¡","size","kÃ­ch thÆ°á»›c","Ä‘Ã³ng gÃ³i"],
    "PRODUCT#DESIGN": ["giao","ship","giÃ¡","size","kÃ­ch thÆ°á»›c","Ä‘Ã³ng gÃ³i","pháº£n há»“i"],
    "PRODUCT#FUNCTION": ["giao","ship","size","giÃ¡","Ä‘Ã³ng gÃ³i"],
    "PRODUCT#DURABILITY": ["giao","ship","size","giÃ¡","Ä‘Ã³ng gÃ³i"],
    "PRODUCT#VALUE": ["giao","ship","Ä‘Ã³ng gÃ³i","size"],
    "DELIVERY#SPEED": ["cháº¥t lÆ°á»£ng","cháº¥t liá»‡u","size","thiáº¿t káº¿","mÃ u","Ä‘áº¹p","xáº¥u","má»m","cá»©ng","bá»n"],
    "DELIVERY#PACKAGING": ["cháº¥t lÆ°á»£ng","size","thiáº¿t káº¿","mÃ u","má»m","cá»©ng"],
    "DELIVERY#ACCURACY": ["cháº¥t lÆ°á»£ng","size","thiáº¿t káº¿","mÃ u","má»m"],
    "PRICE#AFFORDABILITY": ["giao","ship","Ä‘Ã³ng gÃ³i","cháº¥t liá»‡u","size","thiáº¿t káº¿"],
    "PRICE#DISCOUNT": ["giao","ship","cháº¥t lÆ°á»£ng","size","cháº¥t liá»‡u"],
    "SELLER#SERVICE": ["giao","ship","váº­n chuyá»ƒn","size","cháº¥t liá»‡u","thiáº¿t káº¿"],
    "SELLER#RESPONSIVENESS": ["giao","ship","size","cháº¥t liá»‡u","thiáº¿t káº¿"],
    "SELLER#AUTHENTICITY": ["giao","ship","size","cháº¥t liá»‡u","thiáº¿t káº¿"],
}

ASPECT_DISPLAY_NAMES = {
    "PRODUCT#SIZE": "KÃ­ch thÆ°á»›c / Ä‘á»™ vá»«a váº·n",
    "PRODUCT#QUALITY": "Cháº¥t lÆ°á»£ng tá»•ng thá»ƒ",
    "PRODUCT#MATERIAL": "Cháº¥t liá»‡u",
    "PRODUCT#COMFORT": "Cáº£m giÃ¡c sá»­ dá»¥ng",
    "PRODUCT#DESIGN": "Thiáº¿t káº¿ & kiá»ƒu dÃ¡ng",
    "PRODUCT#SAFETY": "Äá»™ an toÃ n",
    "PRODUCT#FUNCTION": "TÃ­nh nÄƒng & cÃ´ng dá»¥ng",
    "PRODUCT#DURABILITY": "Äá»™ bá»n",
    "PRODUCT#VALUE": "GiÃ¡ trá»‹ so vá»›i giÃ¡ tiá»n",
    "PRICE#AFFORDABILITY": "Má»©c giÃ¡",
    "PRICE#DISCOUNT": "Æ¯u Ä‘Ã£i & khuyáº¿n mÃ£i",
    "DELIVERY#SPEED": "Tá»‘c Ä‘á»™ giao hÃ ng",
    "DELIVERY#PACKAGING": "Cháº¥t lÆ°á»£ng Ä‘Ã³ng gÃ³i",
    "DELIVERY#ACCURACY": "Äá»™ chÃ­nh xÃ¡c Ä‘Æ¡n hÃ ng",
    "SELLER#SERVICE": "Cháº¥t lÆ°á»£ng dá»‹ch vá»¥ cá»§a shop",
    "SELLER#RESPONSIVENESS": "Má»©c Ä‘á»™ pháº£n há»“i cá»§a shop",
    "SELLER#AUTHENTICITY": "Äá»™ tin cáº­y / chÃ­nh hÃ£ng",
}

ASPECT_ICONS = {
    "PRODUCT#QUALITY": "ðŸ§µ", "PRODUCT#MATERIAL": "ðŸª¡", "PRODUCT#COMFORT": "ðŸ˜Œ",
    "PRODUCT#SIZE": "ðŸ“", "PRODUCT#DESIGN": "ðŸŽ¨", "PRODUCT#SAFETY": "ðŸ›¡ï¸",
    "PRODUCT#FUNCTION": "âš™ï¸", "PRODUCT#DURABILITY": "ðŸ’ª", "PRODUCT#VALUE": "ðŸ’°",
    "PRICE#AFFORDABILITY": "ðŸ·ï¸", "PRICE#DISCOUNT": "ðŸŽ«",
    "DELIVERY#SPEED": "ðŸšš", "DELIVERY#PACKAGING": "ðŸ“¦", "DELIVERY#ACCURACY": "âœ…",
    "SELLER#SERVICE": "ðŸ›’", "SELLER#RESPONSIVENESS": "ðŸ’¬", "SELLER#AUTHENTICITY": "ðŸ”",
}

SENTIMENT_NAMES = ["none", "positive", "neutral", "negative"]
SENTIMENT_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# â”€â”€â”€ HTTP headers giáº£ láº­p browser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€â”€ In-memory cache Ä‘á»ƒ khÃ´ng crawl láº¡i cÃ¹ng sáº£n pháº©m trong 1 session â”€â”€â”€â”€â”€â”€â”€â”€
_product_info_cache: Dict[str, Dict] = {}
_reviews_cache: Dict[str, List[Dict]] = {}

# â”€â”€â”€ CSV caches (dÃ¹ng cho recommendation candidates) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_detail_df_cache: Optional[pd.DataFrame] = None
_listing_df_cache: Optional[pd.DataFrame] = None
_cat_df_cache: Optional[pd.DataFrame] = None
_reviews_df_cache: Optional[pd.DataFrame] = None  # fallback only
_asqp_df_cache: Optional[pd.DataFrame] = None      # asqp_annotated_flat.csv


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HELPER UTILITIES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
    """CSV reviews â€” chá»‰ dÃ¹ng lÃ m fallback."""
    global _reviews_df_cache
    if _reviews_df_cache is None:
        _reviews_df_cache = _load_csv_safe("Tiki_be_reviews.csv")
    return _reviews_df_cache


def _get_asqp_df() -> pd.DataFrame:
    """Load asqp_annotated_flat.csv â€” nguá»“n quote Ä‘Ã£ Ä‘Æ°á»£c LLM gÃ¡n nhÃ£n chÃ­nh xÃ¡c."""
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

    print("[ASQP] asqp_annotated_flat.csv not found â€” quote sáº½ láº¥y tá»« PhoBERT")
    _asqp_df_cache = pd.DataFrame()
    return _asqp_df_cache


def _pick_quote_from_asqp(aspect: str, sentiment: str, max_len: int = 300) -> str:
    """
    Láº¥y quote tá»« asqp_annotated_flat.csv theo aspect + sentiment.
    Tráº£ vá» text_preview cá»§a row khá»›p, Æ°u tiÃªn review dÃ i hÆ¡n.
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

    # Æ¯u tiÃªn text dÃ i hÆ¡n (thÆ°á»ng chá»©a nhiá»u thÃ´ng tin hÆ¡n)
    pool = pool.copy()
    pool["_len"] = pool["text_preview"].str.len()
    pool = pool[pool["_len"] >= 15].sort_values("_len", ascending=False)
    if pool.empty:
        return ""

    # Láº¥y ngáº«u nhiÃªn trong top 10 Ä‘á»ƒ cÃ³ Ä‘a dáº¡ng
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
        return {"label": "NÃªn mua", "tone": "positive"}
    if score >= 0.3:
        return {"label": "CÃ¢n nháº¯c", "tone": "neutral"}
    return {"label": "Tháº­n trá»ng", "tone": "negative"}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LIVE TIKI API CRAWLERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _safe_request(url: str, params: Optional[Dict] = None,
                  timeout: int = 10, max_retry: int = 3) -> Optional[requests.Response]:
    """GET vá»›i retry + exponential back-off."""
    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.get(url, headers=_TIKI_HEADERS,
                                params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            print(f"[HTTP] {url} â†’ {resp.status_code} (attempt {attempt})")
        except requests.RequestException as exc:
            print(f"[HTTP] {url} error (attempt {attempt}): {exc}")
        time.sleep(1.5 * attempt)
    return None


def fetch_tiki_product_info_live(product_id: str) -> Dict[str, Any]:
    """
    Láº¥y thÃ´ng tin sáº£n pháº©m TRá»°C TIáº¾P tá»« Tiki API /api/v2/products/{id}.
    Tráº£ vá» dict Ä‘á»“ng nháº¥t vá»›i cáº¥u trÃºc cÅ© Ä‘á»ƒ khÃ´ng thay Ä‘á»•i downstream code.
    """
    if product_id in _product_info_cache:
        return _product_info_cache[product_id]

    info: Dict[str, Any] = {
        "product_id": product_id,
        "name": f"Sáº£n pháº©m #{product_id}",
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
        print(f"[LiveAPI] Cannot fetch product {product_id} â€” using defaults")
        _product_info_cache[product_id] = info
        return info

    try:
        data = resp.json()
    except Exception as e:
        print(f"[LiveAPI] JSON parse error: {e}")
        _product_info_cache[product_id] = info
        return info

    # â”€â”€ Basic fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    info["name"] = data.get("name") or info["name"]
    info["price"] = int(data.get("price") or 0)
    info["original_price"] = int(data.get("list_price") or data.get("price") or 0)
    info["rating_average"] = float(data.get("rating_average") or 0.0)
    info["review_count"] = int(data.get("review_count") or 0)
    info["short_description"] = data.get("short_description") or ""

    # â”€â”€ Brand & Seller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    brand = data.get("brand") or {}
    info["brand_name"] = brand.get("name") or ""

    seller = data.get("current_seller") or {}
    info["seller_name"] = seller.get("name") or ""
    info["seller_is_official"] = bool(seller.get("is_official", False))

    # â”€â”€ Sold quantity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    qty_sold = data.get("quantity_sold") or {}
    if isinstance(qty_sold, dict):
        info["sold_quantity"] = int(qty_sold.get("value") or 0)
    else:
        info["sold_quantity"] = int(qty_sold or 0)

    # â”€â”€ Images â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    thumbnail = data.get("thumbnail_url") or ""
    images_raw = data.get("images") or []
    images = [img.get("base_url", "") for img in images_raw if img.get("base_url")]

    if not thumbnail and images:
        thumbnail = images[0]
    info["thumbnail_url"] = thumbnail
    info["images"] = images[:6]

    # â”€â”€ Product URL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    short_url = data.get("short_url") or ""
    if short_url.startswith("https://tiki.vn/"):
        info["product_url"] = short_url

    # â”€â”€ Categories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    breadcrumbs = data.get("breadcrumbs") or []
    if len(breadcrumbs) >= 1:
        info["category_lv1"] = breadcrumbs[0].get("name", "")
    if len(breadcrumbs) >= 2:
        info["category_lv2"] = breadcrumbs[1].get("name", "")
    if len(breadcrumbs) >= 3:
        info["category_lv3"] = breadcrumbs[2].get("name", "")

    # Fallback: thá»­ categories field
    if not info["category_lv1"]:
        categories = data.get("categories") or {}
        if isinstance(categories, dict):
            info["category_lv1"] = categories.get("name", "")

    print(
        f"[LiveAPI] Product OK: '{info['name']}' | "
        f"price={info['price']:,}Ä‘ | rating={info['rating_average']} | "
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
    Crawl toÃ n bá»™ reviews cá»§a sáº£n pháº©m qua Tiki API /api/v2/reviews.
    Há»— trá»£ phÃ¢n trang vÃ  progress reporting.
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
            print(f"[LiveAPI] Page {page} failed â€” stopping")
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
                "customer_name": creator.get("name") or "áº¨n danh",
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
                    f"Thu tháº­p Ä‘Ã¡nh giÃ¡ trang {page} â€” {len(all_reviews)} reviews..."
                )
            except Exception:
                pass

        # Kiá»ƒm tra cÃ³ trang tiáº¿p theo khÃ´ng
        paging = body.get("paging") or {}
        total_pages = paging.get("last_page") or paging.get("total_pages") or page
        if page >= total_pages:
            print(f"[LiveAPI] All pages fetched ({page}/{total_pages})")
            break

        # Throttle Ä‘á»ƒ khÃ´ng bá»‹ block
        time.sleep(random.uniform(0.5, 1.2))

    print(f"[LiveAPI] Total reviews crawled: {len(all_reviews)}")
    _reviews_cache[product_id] = all_reviews
    return all_reviews


def _get_reviews_from_csv_fallback(product_id: str) -> List[Dict]:
    """Fallback: láº¥y reviews tá»« CSV náº¿u API hoÃ n toÃ n tháº¥t báº¡i."""
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
    """Fallback: láº¥y product info tá»« CSV náº¿u API tháº¥t báº¡i."""
    info: Dict[str, Any] = {
        "product_id": product_id,
        "name": f"Sáº£n pháº©m #{product_id}",
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PHOBERT PREDICTOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AppPhoBERTASQP(nn.Module):
    def __init__(self, backbone_config_dir: Path, last_n_layers: int = 4,
                 proj_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        bert_config = AutoConfig.from_pretrained(str(backbone_config_dir))
        bert_config.output_hidden_states = True
        self.bert = AutoModel.from_config(bert_config)
        self.last_n_layers = last_n_layers
        concat_dim = int(bert_config.hidden_size) * last_n_layers

        self.projection = nn.Sequential(
            nn.Linear(concat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, len(SENTIMENT_NAMES)),
            )
            for _ in range(len(CATEGORIES))
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_n = outputs.hidden_states[-self.last_n_layers:]
        cls_vectors = [hidden[:, 0, :] for hidden in last_n]
        projected = self.projection(torch.cat(cls_vectors, dim=-1))
        return [classifier(projected) for classifier in self.classifiers]


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class RealPhoBERTPredictor:
    def __init__(self, max_len: int = 256, confidence_threshold: float = 0.6):
        self.max_len = max_len
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self.model_path: Optional[Path] = None
        self.load_error: Optional[str] = None
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
        if model_path is None:
            self._loaded = True
            self.load_error = "Khong tim thay models/phobert/best_model.pt"
            raise FileNotFoundError(self.load_error)

        try:
            from transformers import AutoTokenizer
            local_tok = _ROOT_DIR / "models" / "phobert" / "base_model"
            if not local_tok.exists():
                local_tok = _ROOT_DIR / "models" / "phobert" / "tokenizer"
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(local_tok) if local_tok.exists() else "vinai/phobert-base-v2"
            )
            print("[PhoBERT] Tokenizer loaded")
        except Exception as e:
            self._loaded = True
            self.load_error = f"Khong load duoc tokenizer PhoBERT: {e}"
            raise RuntimeError(self.load_error) from e

        try:
            backbone_config_dir = _ROOT_DIR / "models" / "phobert" / "base_model"
            if not (backbone_config_dir / "config.json").exists():
                raise FileNotFoundError(f"Khong tim thay {backbone_config_dir / 'config.json'}")

            ckpt = _torch_load_checkpoint(model_path, self.device)
            self.model = AppPhoBERTASQP(backbone_config_dir).to(self.device)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_path = model_path
            print(f"[PhoBERT] Model loaded from {model_path} on {self.device}")
        except Exception as e:
            self.model = None
            self._loaded = True
            self.load_error = f"Khong load duoc best_model PhoBERT: {e}"
            raise RuntimeError(self.load_error) from e

        self._loaded = True

    def predict_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            return []
        return self.predict_sentences([sentence])[0]

    def predict_sentences(self, sentences: List[str]) -> List[List[Dict[str, Any]]]:
        self._load_model()
        normalized = [sentence.strip() for sentence in sentences]
        results: List[Optional[List[Dict[str, Any]]]] = [None] * len(normalized)
        uncached_sentences: List[str] = []
        uncached_indices: List[int] = []
        for idx, sentence in enumerate(normalized):
            if not sentence or len(sentence) < 10:
                results[idx] = []
            elif sentence in self.cache:
                results[idx] = self.cache[sentence]
            else:
                uncached_indices.append(idx)
                uncached_sentences.append(sentence)

        if not uncached_sentences:
            return [item or [] for item in results]

        if self.model is not None and self.tokenizer is not None:
            try:
                batch_results = self._predict_batch_with_model(uncached_sentences)
            except Exception as e:
                if self.device.type == "cuda" and "out of memory" in str(e).lower():
                    print("[PhoBERT] CUDA OOM, switching inference to CPU...")
                    torch.cuda.empty_cache()
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    batch_results = self._predict_batch_with_model(uncached_sentences)
                else:
                    raise RuntimeError(f"PhoBERT inference error: {e}") from e

            for idx, sentence, result in zip(uncached_indices, uncached_sentences, batch_results):
                self.cache[sentence] = result
                results[idx] = result
            return [item or [] for item in results]

        raise RuntimeError(self.load_error or "PhoBERT best_model chua duoc load")

    def _predict_batch_with_model(self, sentences: List[str]) -> List[List[Dict[str, Any]]]:
        if not sentences:
            return []
        enc = self.tokenizer(
            sentences,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits_per_cat = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
        outputs: List[List[Dict[str, Any]]] = [[] for _ in sentences]
        for cat_idx, logits in enumerate(logits_per_cat):
            probs = F.softmax(logits, dim=-1)
            confs, pred_indices = torch.max(probs, dim=-1)
            for row_idx, pred_idx_tensor in enumerate(pred_indices):
                pred_idx = int(pred_idx_tensor.item())
                pred_name = SENTIMENT_NAMES[pred_idx]
                conf = float(confs[row_idx].item())
                if pred_name != "none" and conf >= self.confidence_threshold:
                    outputs[row_idx].append({
                        "aspect": CATEGORIES[cat_idx],
                        "sentiment": pred_name,
                        "confidence": conf,
                        "sentiment_score": SENTIMENT_TO_SCORE[pred_name],
                    })
        return outputs

    def _predict_with_model(self, sentence: str) -> List[Dict[str, Any]]:
        return self._predict_batch_with_model([sentence])[0]

    def _rule_based_predict(self, sentence: str) -> List[Dict[str, Any]]:
        sl = sentence.lower()
        pos_kws = ["tá»‘t","hay","Ä‘áº¹p","cháº¥t lÆ°á»£ng","tuyá»‡t","á»•n","hÃ i lÃ²ng","thÃ­ch","Æ°ng",
                   "ngon","bá»n","Ä‘Ã¡ng tiá»n","má»m","dá»… chá»‹u","thoáº£i mÃ¡i","tiá»‡n lá»£i",
                   "an toÃ n","cháº¯c cháº¯n","ráº»","há»£p lÃ½"]
        neg_kws = ["tá»‡","há»ng","rÃ¡ch","xáº¥u","kÃ©m","khÃ´ng tá»‘t","chÃ¡n","tháº¥t vá»ng",
                   "má»ng","cá»©ng","khÃ³ chá»‹u","Ä‘áº¯t","cháº­m","khÃ´ng hÃ i lÃ²ng","lá»—i","tá»“i","báº©n"]
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DASHBOARD SERVICE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class DashboardService:
    def __init__(self):
        self.config = DashboardConfig()
        self._predictor: Optional[RealPhoBERTPredictor] = None
        self._v2_repository = V2Repository()
        self._v2_vector_store = QdrantOpinionStore()
        if self._v2_repository.configured and not self._v2_repository.enabled:
            print(f"[V2 DB] Disabled: {self._v2_repository.status.get('error')}")

    def _get_predictor(self) -> RealPhoBERTPredictor:
        if self._predictor is None:
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
        self._safe_progress(progress_callback, 45, f"Äang phÃ¢n tÃ­ch {total} cÃ¢u vá»›i PhoBERT...")
        tuples: List[Dict] = []
        review_bucket: Dict[str, List[Dict]] = {}
        batch_size = max(1, int(self.config.phobert_batch_size))
        for start in range(0, total, batch_size):
            batch_meta = sentence_meta[start:start + batch_size]
            batch_sentences = [meta["sentence"] for meta in batch_meta]
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
                f"Ã„Âang phÃƒÂ¢n tÃƒÂ­ch cÃƒÂ¢u {processed}/{total}..."
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

        # â”€â”€ Táº§ng 1: ASQP CSV â€” nguá»“n Ä‘Ã¡ng tin cáº­y nháº¥t (LLM-labeled) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        asqp_quote = _pick_quote_from_asqp(aspect, sentiment, max_len)
        if asqp_quote:
            return asqp_quote

        # â”€â”€ Táº§ng 2: PhoBERT pool vá»›i keyword filter (fallback) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        pool = [t for t in tuples if t.get("aspect") == aspect and t.get("sentiment") == sentiment]
        if not pool:
            return ""
        required_kws = ASPECT_REQUIRED_KEYWORDS.get(aspect, [])
        forbidden_kws = ASPECT_FORBIDDEN_KEYWORDS.get(aspect, [])

        def score_item(t):
            return (t.get("helpful_count", 0), t.get("confidence", 0), len(t.get("sentence", "")))

        # Tier 2a: cÃ³ required keyword VÃ€ khÃ´ng cÃ³ forbidden keyword
        tier1 = [t for t in pool
                 if required_kws and any(kw in t.get("sentence", "").lower() for kw in required_kws)
                 and not any(kw in t.get("sentence", "").lower() for kw in forbidden_kws)]
        if tier1:
            return re.sub(r"\s+", " ", random.choice(sorted(tier1, key=score_item, reverse=True)[:5]).get("sentence", "")).strip()[:max_len]

        # Tier 2b: cÃ³ required keyword (ná»›i lá»ng forbidden)
        tier2 = [t for t in pool
                 if required_kws and any(kw in t.get("sentence", "").lower() for kw in required_kws)]
        if tier2:
            return re.sub(r"\s+", " ", random.choice(sorted(tier2, key=score_item, reverse=True)[:5]).get("sentence", "")).strip()[:max_len]

        # KhÃ´ng cÃ³ cÃ¢u nÃ o khá»›p â†’ tráº£ rá»—ng, khÃ´ng hiá»ƒn thá»‹ quote lá»‡ch aspect
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
        TÃ¬m sáº£n pháº©m tÆ°Æ¡ng tá»± tá»« CSV (recommendation candidates).
        Náº¿u CSV trá»‘ng â†’ tráº£ vá» list rá»—ng (khÃ´ng crash).
        """
        category_lv3 = product_info.get("category_lv3", "")
        category_lv2 = product_info.get("category_lv2", "")
        if not category_lv3 and not category_lv2:
            print("[Recommend] No category info â€” skipping recommendations")
            return []
        cat_df = _get_cat_df()
        if cat_df.empty:
            print("[Recommend] cat_df empty â€” skipping")
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
            if info["name"] and info["name"] != f"Sáº£n pháº©m #{pid}":
                # DÃ¹ng rating tá»« CSV (khÃ´ng live-crawl Ä‘á»ƒ trÃ¡nh quÃ¡ nhiá»u requests)
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

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # MAIN ENTRY POINT
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def analyze_product(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:

        # â”€â”€ 1. TrÃ­ch xuáº¥t product_id â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._safe_progress(progress_callback, 3, "TrÃ­ch xuáº¥t product ID tá»« URL...")
        if not product_id and product_url:
            product_id = extract_product_id_from_url(product_url)
        if not product_id:
            raise ValueError("KhÃ´ng thá»ƒ xÃ¡c Ä‘á»‹nh product ID tá»« URL Ä‘Ã£ nháº­p")
        product_id = str(product_id).strip()
        print(f"\n[Analysis] â•â•â• Starting for product_id={product_id} â•â•â•")

        # â”€â”€ 2. Live crawl thÃ´ng tin sáº£n pháº©m â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._safe_progress(progress_callback, 8, "Äang láº¥y thÃ´ng tin sáº£n pháº©m tá»« Tiki...")
        product_info = fetch_tiki_product_info_live(product_id)

        # Náº¿u API fail â†’ thá»­ CSV fallback
        if product_info["name"] == f"Sáº£n pháº©m #{product_id}":
            print("[Analysis] API returned empty â€” trying CSV fallback for product info")
            csv_info = _get_product_info_from_csv_fallback(product_id)
            if csv_info["name"] != f"Sáº£n pháº©m #{product_id}":
                product_info.update(csv_info)

        self._safe_progress(progress_callback, 15, f"TÃ¬m tháº¥y: {product_info['name'][:40]}...")

        # â”€â”€ 3. Live crawl reviews â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._safe_progress(progress_callback, 18, "Äang thu tháº­p Ä‘Ã¡nh giÃ¡ tá»« Tiki API...")
        reviews = fetch_tiki_reviews_live(
            product_id,
            max_pages=self.config.max_review_pages,
            per_page=self.config.reviews_per_page,
            progress_callback=progress_callback,
        )

        # Náº¿u live crawl ra 0 review â†’ fallback CSV
        if not reviews:
            print(f"[Analysis] Live crawl returned 0 reviews â€” trying CSV fallback")
            self._safe_progress(progress_callback, 28, "KhÃ´ng tÃ¬m tháº¥y review trá»±c tiáº¿p, dÃ¹ng dá»¯ liá»‡u offline...")
            reviews = _get_reviews_from_csv_fallback(product_id)

        if not reviews:
            raise ValueError(
                f"KhÃ´ng tÃ¬m tháº¥y Ä‘Ã¡nh giÃ¡ nÃ o cho sáº£n pháº©m {product_id}. "
                "Sáº£n pháº©m cÃ³ thá»ƒ chÆ°a cÃ³ review hoáº·c URL khÃ´ng há»£p lá»‡."
            )

        self._safe_progress(progress_callback, 30, f"ÄÃ£ thu tháº­p {len(reviews)} Ä‘Ã¡nh giÃ¡. Äang tiá»n xá»­ lÃ½...")

        # â”€â”€ 4. Tiá»n xá»­ lÃ½ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        reviews_df = self._preprocess_reviews(reviews)
        if reviews_df.empty:
            raise ValueError("KhÃ´ng cÃ³ Ä‘Ã¡nh giÃ¡ há»£p lá»‡ sau tiá»n xá»­ lÃ½ vÄƒn báº£n")
        reviews_df = reviews_df.head(self.config.max_reviews_for_dashboard)
        print(f"[Analysis] Using {len(reviews_df)} reviews for ABSA")

        # â”€â”€ 5. PhoBERT prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._safe_progress(progress_callback, 40, "Äang phÃ¢n tÃ­ch cáº£m xÃºc vá»›i PhoBERT...")
        tuples, review_scores, sentiment_bucket = self._batch_predict(reviews_df, progress_callback)
        if not tuples:
            raise ValueError("PhoBERT khÃ´ng tráº£ vá» káº¿t quáº£ â€” Ä‘Ã¡nh giÃ¡ quÃ¡ ngáº¯n hoáº·c khÃ´ng cÃ³ ná»™i dung")

        # â”€â”€ 6. Tá»•ng há»£p káº¿t quáº£ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._safe_progress(progress_callback, 82, "Tá»•ng há»£p káº¿t quáº£ theo khÃ­a cáº¡nh...")
        aspect_payload = self._build_aspect_payload(tuples)
        self._safe_progress(progress_callback, 86, "TrÃ­ch xuáº¥t Ã½ kiáº¿n chi tiáº¿t...")
        opinion_table = self._build_opinion_table(tuples)
        risk_flags = detect_risks(tuples)
        self._safe_progress(progress_callback, 90, "TÃ¬m sáº£n pháº©m gá»£i Ã½ tÆ°Æ¡ng tá»±...")
        recommendations = self._get_candidates_by_category(product_id, product_info)

        # â”€â”€ 7. TÃ­nh Ä‘iá»ƒm tá»•ng há»£p â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€ 8. Radar so sÃ¡nh vá»›i top1 recommendation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        top1_aspect_scores: Dict[str, float] = {}
        if recommendations:
            top1_id = recommendations[0]["product_id"]
            # DÃ¹ng CSV fallback cho top1 (khÃ´ng live-crawl Ä‘á»ƒ trÃ¡nh delay dÃ i)
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

        self._safe_progress(progress_callback, 98, "Äang hoÃ n thiá»‡n bÃ¡o cÃ¡o...")
        print(f"[Analysis] âœ… Complete! ABSA score: {absa_score:.3f} | Reviews: {total_reviews}")

        # â”€â”€ 9. Build final result dict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
            "risk_flags": risk_flags,
            "top1_aspect_scores": top1_aspect_scores if top1_aspect_scores else None,
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
                    print(f"[V2 Qdrant] Indexed {indexed} opinion chunks for product_id={product_id}")
        except Exception as exc:
            print(f"[V2 DB] Save skipped: {exc}")
        return result
