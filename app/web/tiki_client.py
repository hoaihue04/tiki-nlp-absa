from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional

import requests

ProgressCallback = Callable[[int, str], None]

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

_product_info_cache: Dict[str, Dict] = {}
_reviews_cache: Dict[str, List[Dict]] = {}


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

    info["name"] = data.get("name") or info["name"]
    info["price"] = int(data.get("price") or 0)
    info["original_price"] = int(data.get("list_price") or data.get("price") or 0)
    info["rating_average"] = float(data.get("rating_average") or 0.0)
    info["review_count"] = int(data.get("review_count") or 0)
    info["short_description"] = data.get("short_description") or ""

    brand = data.get("brand") or {}
    info["brand_name"] = brand.get("name") or ""

    seller = data.get("current_seller") or {}
    info["seller_name"] = seller.get("name") or ""
    info["seller_is_official"] = bool(seller.get("is_official", False))

    qty_sold = data.get("quantity_sold") or {}
    if isinstance(qty_sold, dict):
        info["sold_quantity"] = int(qty_sold.get("value") or 0)
    else:
        info["sold_quantity"] = int(qty_sold or 0)

    thumbnail = data.get("thumbnail_url") or ""
    images_raw = data.get("images") or []
    images = [img.get("base_url", "") for img in images_raw if img.get("base_url")]

    if not thumbnail and images:
        thumbnail = images[0]
    info["thumbnail_url"] = thumbnail
    info["images"] = images[:6]

    short_url = data.get("short_url") or ""
    if short_url.startswith("https://tiki.vn/"):
        info["product_url"] = short_url

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

        paging = body.get("paging") or {}
        total_pages = paging.get("last_page") or paging.get("total_pages") or page
        if page >= total_pages:
            print(f"[LiveAPI] All pages fetched ({page}/{total_pages})")
            break

        time.sleep(random.uniform(0.5, 1.2))

    print(f"[LiveAPI] Total reviews crawled: {len(all_reviews)}")
    _reviews_cache[product_id] = all_reviews
    return all_reviews
