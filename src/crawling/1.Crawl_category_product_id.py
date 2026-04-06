import requests
import csv
import time

# ===================== CONFIG =====================
BASE_API = "https://tiki.vn/api"
CATEGORY_LV1_ID = 2549   # Mẹ & Bé

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

OUTPUT_FILE = r"C:\Users\HOAI HUE\Desktop\tiki\data\raw\tiki_me_be_products_id.csv"

# ✅ DANH MỤC LV2 ĐƯỢC PHÉP (PHÙ HỢP ĐỀ TÀI)
TARGET_LV2_CATEGORIES = {
    "Tã, Bỉm",
    "Dinh dưỡng cho bé",
    "Thực phẩm ăn dặm",
    "Đồ dùng cho bé"
}

# ===================== GET SUB CATEGORY =====================
def get_sub_categories(parent_id):
    url = f"{BASE_API}/v2/categories"
    params = {"parent_id": parent_id}

    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        print(f"❌ Lỗi lấy category parent_id={parent_id}")
        return []

    return res.json().get("data", [])


# ===================== GET PRODUCT IDS =====================
def get_product_ids(category_id, max_page=3):
    product_ids = []

    for page in range(1, max_page + 1):
        url = f"{BASE_API}/v2/products"
        params = {
            "category": category_id,
            "page": page,
            "limit": 40
        }

        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code != 200:
            break

        data = res.json().get("data", [])
        if not data:
            break

        for item in data:
            product_ids.append(item["id"])

        time.sleep(0.6)

    return product_ids


# ===================== MAIN =====================
def main():
    results = []

    print("🔍 Lấy category level 2 (Mẹ & Bé)...")
    lv2_categories = get_sub_categories(CATEGORY_LV1_ID)

    for lv2 in lv2_categories:
        lv2_id = lv2["id"]
        lv2_name = lv2["name"]

        # ❌ BỎ QUA DANH MỤC KHÔNG PHÙ HỢP
        if lv2_name not in TARGET_LV2_CATEGORIES:
            print(f"⏭️ Bỏ qua LV2: {lv2_name}")
            continue

        print(f"\n✅ LV2 hợp lệ: {lv2_name}")

        lv3_categories = get_sub_categories(lv2_id)

        # ⚠️ CHỈ LẤY KHI CÓ LV3
        if not lv3_categories:
            print("   ⚠️ Không có LV3 → bỏ qua")
            continue

        for lv3 in lv3_categories:
            lv3_id = lv3["id"]
            lv3_name = lv3["name"]

            print(f"   └─ LV3: {lv3_name}")

            product_ids = get_product_ids(lv3_id)

            for pid in product_ids:
                results.append([
                    "Mẹ & Bé",
                    lv2_name,
                    lv3_name,
                    lv3_id,     # ✅ category_id = LV3
                    pid
                ])

    # ===================== WRITE CSV =====================
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category_lv1",
            "category_lv2",
            "category_lv3",
            "category_id",
            "product_id"
        ])
        writer.writerows(results)

    print(f"\n✅ HOÀN TẤT: {len(results)} product IDs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
