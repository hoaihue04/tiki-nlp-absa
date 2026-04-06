import pandas as pd
import requests
import time
import random
import json
from tqdm import tqdm
from bs4 import BeautifulSoup

# ===================== HEADERS =====================
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json, text/plain, */*',
    'x-guest-token': '8jWSuIDBb2NGVzr6hsUZXpkP1FRin7lY'
}

PARAMS = {
    'platform': 'web'
}

# ===================== UTILS =====================
def clean_html(html):
    """Remove HTML tags from description"""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def parse_specifications(specs):
    """Convert specifications list to dict"""
    spec_dict = {}
    if not specs:
        return spec_dict

    for group in specs:
        attrs = group.get("attributes") or []
        for attr in attrs:
            key = attr.get("name")
            val = attr.get("value")
            if key and val:
                spec_dict[key] = val

    return spec_dict

# ===================== PARSER PRODUCT =====================
def parser_product(js):
    d = {}

    # -------- PRODUCT BASIC --------
    d['product_id'] = js.get('id')
    d['sku'] = js.get('sku')
    d['name'] = js.get('name')
    d['short_description'] = js.get('short_description')

    d['price'] = js.get('price')
    d['list_price'] = js.get('list_price')
    d['discount'] = js.get('discount')
    d['discount_rate'] = js.get('discount_rate')

    d['rating_average'] = js.get('rating_average')
    d['review_count'] = js.get('review_count')
    d['order_count'] = js.get('order_count')

    stock = js.get('stock_item') or {}
    d['stock_qty'] = stock.get('qty')
    d['max_sale_qty'] = stock.get('max_sale_qty')

    brand = js.get('brand') or {}
    d['brand_id'] = brand.get('id')
    d['brand_name'] = brand.get('name')

    d['inventory_status'] = js.get('inventory_status')
    d['is_visible'] = js.get('is_visible')

    # -------- PRODUCT LINK --------
    d['product_url'] = js.get('short_url') or f"https://tiki.vn/p/{js.get('id')}"

    # -------- DESCRIPTION --------
    d['description'] = clean_html(js.get('description'))

    # -------- SPECIFICATIONS --------
    specs = parse_specifications(js.get('specifications'))
    d['specifications'] = json.dumps(specs, ensure_ascii=False)

    # ===================== SELLER =====================
    seller = js.get('current_seller') or {}

    d['seller_id'] = seller.get('id')
    d['seller_name'] = seller.get('name')
    d['seller_type'] = seller.get('type')
    d['seller_rating'] = seller.get('rating_average')
    d['seller_review_count'] = seller.get('review_count')
    d['seller_product_count'] = seller.get('product_count')
    d['seller_is_official'] = seller.get('is_official', False)
    d['seller_link'] = seller.get('link')

    return d

# ===================== LOAD PRODUCT IDS =====================
CSV_PATH = r"C:\Users\HOAI HUE\Desktop\tiki\data\raw\tiki_me_be_products_id.csv"
df_products = pd.read_csv(CSV_PATH)

product_ids = (
    df_products['product_id']
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

print(f"🧮 Total unique product IDs: {len(product_ids)}")

# ===================== CRAWL PRODUCT DETAIL =====================
results = []

for pid in tqdm(product_ids):
    url = f"https://tiki.vn/api/v2/products/{pid}"

    try:
        res = requests.get(url, headers=headers, params=PARAMS, timeout=10)

        if res.status_code == 200:
            results.append(parser_product(res.json()))
        else:
            print(f"❌ {pid} | Status {res.status_code}")

    except Exception as e:
        print(f"⚠️ Error {pid}: {e}")

    time.sleep(random.uniform(0.5, 1.2))

# ===================== SAVE =====================
df_result = pd.DataFrame(results)

OUTPUT_FILE = r"C:\Users\HOAI HUE\Desktop\tiki\data\raw\tiki_me_be_product_detail_full.csv"
df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"✅ DONE: {len(df_result)} products saved → {OUTPUT_FILE}")
