import requests
import pandas as pd
import time
import random
from tqdm import tqdm

# ===================== CONFIG =====================
INPUT_CSV = r"C:\Users\HOAI HUE\Desktop\tiki\data\raw\tiki_me_be_products_id.csv"
OUTPUT_CSV = r"C:\Users\HOAI HUE\Desktop\tiki\data\raw\tiki_me_be_products_listing.csv"

BASE_API = "https://tiki.vn/api/personalish/v1/blocks/listings"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "x-guest-token": "8jWSuIDBb2NGVzr6hsUZXpkP1FRin7lY"
}

BASE_PARAMS = {
    "limit": 40,
    "include": "advertisement",
    "aggregations": 2,
    "trackity_id": "dummy"
}

# ===================== LOAD CATEGORY IDS =====================
df_cat = pd.read_csv(INPUT_CSV)

category_ids = (
    df_cat["category_id"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

print(f"🗂️ Total categories: {len(category_ids)}")

# ===================== CRAWL LISTING =====================
rows = []

for cat_id in tqdm(category_ids, desc="Crawling categories"):
    page = 1

    while True:
        params = BASE_PARAMS.copy()
        params["category"] = cat_id
        params["page"] = page

        try:
            res = requests.get(BASE_API, headers=HEADERS, params=params, timeout=10)

            if res.status_code != 200:
                print(f"❌ Category {cat_id} | Page {page} | Status {res.status_code}")
                break

            data = res.json().get("data", [])
            if not data:
                break

            for item in data:
                quantity = item.get("quantity_sold") or {}

                rows.append({
                    "category_id": cat_id,
                    "product_id": item.get("id"),
                    "product_name": item.get("name"),
                    "quantity_sold": quantity.get("value", 0),
                    "quantity_sold_text": quantity.get("text")
                })

            page += 1
            time.sleep(random.uniform(0.4, 0.8))

        except Exception as e:
            print(f"⚠️ Error category {cat_id} page {page}: {e}")
            break

# ===================== SAVE =====================
df_listing = pd.DataFrame(rows)

# remove duplicate product_id
df_listing.drop_duplicates(subset=["product_id"], inplace=True)

df_listing.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ DONE: {len(df_listing)} unique products saved → {OUTPUT_CSV}")
