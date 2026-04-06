import os
import time
import random
import requests
import pandas as pd
from datetime import datetime
from requests.exceptions import RequestException

# =========================
# CONFIG (PRODUCTION MODE)
# =========================
INPUT_FILE = r"C:\Users\HOAI HUE\Desktop\tiki\data\raw\tiki_me_be_products_id.csv"
OUTPUT_FILE = r"C:\Users\HOAI HUE\Desktop\tiki\data\raw\tiki_reviews.csv"
CHECKPOINT_FILE = r"C:\Users\HOAI HUE\Desktop\tiki\checkpoints\done_products.txt"

BATCH_SIZE = 600            # ✅ crawl 600 sản phẩm / lần chạy
WAIT_PER_PAGE = (1.2, 2.2)
WAIT_PER_PRODUCT = (4, 7)
MAX_RETRY = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

BASE_URL = "https://tiki.vn/api/v2/reviews"

# =========================
# UTILS
# =========================
def ts_to_date(ts):
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except:
        return None

def days_between(ts1, ts2):
    if not ts1 or not ts2:
        return None
    return int((ts2 - ts1) / 86400)

# =========================
# CHECKPOINT
# =========================
def load_done_products():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_product_done(product_id):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(str(product_id) + "\n")

def load_existing_reviews():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        df = pd.read_csv(OUTPUT_FILE, usecols=["review_id"])
        return set(df["review_id"].astype(str))
    except:
        return set()

# =========================
# SAFE REQUEST
# =========================
def safe_get(url, params):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=30
            )
            if r.status_code == 200:
                return r
            else:
                print(f"⚠️ HTTP {r.status_code} – retry {attempt}")
        except RequestException as e:
            print(f"⚠️ Network error – retry {attempt}: {e}")

        time.sleep(3 * attempt)

    return None

# =========================
# CRAWL REVIEWS (LẤY HẾT)
# =========================
def crawl_reviews(product_id, category_id):
    page = 1
    reviews = []

    while True:
        params = {
            "product_id": product_id,
            "page": page,
            "limit": 20,
            "sort": "score|desc"
        }

        r = safe_get(BASE_URL, params)
        if r is None:
            print(f"❌ Skip product {product_id}")
            break

        items = r.json().get("data", [])
        if not items:
            break   # ✅ HẾT REVIEW

        for rv in items:
            creator = rv.get("created_by") or {}

            created_at = rv.get("created_at")
            purchased_at = creator.get("purchased_at")

            reviews.append({
                "review_id": rv.get("id"),
                "product_id": product_id,
                "category_id": category_id,

                "customer_id": creator.get("id"),
                "customer_name": creator.get("name"),

                "rating": rv.get("rating"),
                "title": rv.get("title"),
                "content": rv.get("content"),

                "created_at_ts": created_at,
                "created_at_date": ts_to_date(created_at),

                "purchased_at_ts": purchased_at,
                "purchased_at_date": ts_to_date(purchased_at),

                "days_since_purchase": days_between(purchased_at, created_at),

                "helpful_count": rv.get("thank_count", 0),
                "is_verified": bool(purchased_at),
                "images_count": len(rv.get("images", []))
            })

        page += 1
        time.sleep(random.uniform(*WAIT_PER_PAGE))

    return reviews

# =========================
# MAIN
# =========================
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

    df_products = pd.read_csv(INPUT_FILE)
    done_products = load_done_products()
    existing_reviews = load_existing_reviews()

    print(f"📦 Đã crawl trước đó: {len(done_products)} sản phẩm")

    batch_count = 0
    total_saved = 0

    for _, row in df_products.iterrows():
        product_id = str(row["product_id"])
        category_id = row["category_id"]

        if product_id in done_products:
            continue

        print(f"\n🚀 Crawl product {product_id}")
        reviews = crawl_reviews(product_id, category_id)

        if reviews:
            df_new = pd.DataFrame(reviews)
            df_new = df_new[~df_new["review_id"].astype(str).isin(existing_reviews)]

            if not df_new.empty:
                df_new.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    index=False,
                    header=not os.path.exists(OUTPUT_FILE),
                    encoding="utf-8-sig"
                )
                existing_reviews.update(df_new["review_id"].astype(str))
                total_saved += len(df_new)

        mark_product_done(product_id)
        batch_count += 1

        print(f"✅ Xong {product_id} | Tổng review: {total_saved}")

        time.sleep(random.uniform(*WAIT_PER_PRODUCT))

        if batch_count >= BATCH_SIZE:
            print("🛑 Đạt BATCH_SIZE – dừng an toàn")
            break

    print("\n🎉 HOÀN TẤT")
    print(f"Tổng review đã lưu: {total_saved}")

if __name__ == "__main__":
    main()
