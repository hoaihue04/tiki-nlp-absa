import pandas as pd
import numpy as np
import re
import os
import sys


# Add project root vào path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# IMPORT CONSTANTS (đúng theo project của bạn)
from src.utils.constants import (
    TIKI_REVIEWS_FILE,
    PRODUCT_DETAIL_FILE,
    PRODUCT_CATEGORY_FILE,
    CLEANED_REVIEWS_FILE,
    MERGED_DATA_FILE
)

# ========================
# CLEAN TEXT
# ========================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove HTML
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URL
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ========================
# LOAD DATA
# ========================
def load_data():
    try:
        reviews = pd.read_csv(TIKI_REVIEWS_FILE)
        products = pd.read_csv(PRODUCT_DETAIL_FILE)
        categories = pd.read_csv(PRODUCT_CATEGORY_FILE)

        print(f"Loaded Reviews: {len(reviews)}")
        print(f"Loaded Products: {len(products)}")
        print(f"Loaded Categories: {len(categories)}")

        return reviews, products, categories

    except Exception as e:
        print("❌ Lỗi load data:", e)
        return None, None, None


# ========================
# CLEAN REVIEWS
# ========================
def clean_reviews(df):
    print("\n--- CLEANING REVIEWS ---")

    df = df.copy()

    # Drop null quan trọng
    df.dropna(subset=['content', 'title'], inplace=True)

    # Clean text
    df['cleaned_content'] = df['content'].apply(clean_text)

    # Remove text quá ngắn
    df = df[df['cleaned_content'].str.len() >= 5]

    # Remove duplicate
    df.drop_duplicates(subset=['cleaned_content'], inplace=True)


    # Feature engineering
    df['review_length'] = df['cleaned_content'].apply(len)

    print(f"Remaining reviews after cleaning: {len(df)}")

    return df


# ========================
# MERGE DATA
# ========================
def merge_data(df, products, categories):
    print("\n--- MERGING DATA ---")

    # ⚠️ FIX TYPE (QUAN TRỌNG)
    df['product_id'] = df['product_id'].astype(str)
    products['product_id'] = products['product_id'].astype(str)
    categories['product_id'] = categories['product_id'].astype(str)

    # Product info
    products_subset = products[[
        'product_id',
        'name',
        'price',
        'rating_average',
        'brand_name',
        'seller_name'
    ]]

    df = df.merge(products_subset, on='product_id', how='left')

    # Category info
    categories_subset = categories[[
        'product_id',
        'category_lv1',
        'category_lv2',
        'category_lv3'
    ]]

    df = df.merge(categories_subset, on='product_id', how='left')

    print(f"Merged dataset size: {len(df)}")

    return df

# ========================
# SAVE DATA
# ========================
def save_data(cleaned_df, merged_df):
    # Tạo folder nếu chưa có
    os.makedirs(os.path.dirname(CLEANED_REVIEWS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(MERGED_DATA_FILE), exist_ok=True)

    cleaned_df.to_csv(CLEANED_REVIEWS_FILE, index=False)
    print(f"✅ Saved cleaned data: {CLEANED_REVIEWS_FILE}")

    merged_df.to_csv(MERGED_DATA_FILE, index=False)
    print(f"✅ Saved merged data: {MERGED_DATA_FILE}")


# ========================
# MAIN
# ========================
def main():
    reviews, products, categories = load_data()

    if reviews is None:
        return

    cleaned_df = clean_reviews(reviews)
    merged_df = merge_data(cleaned_df, products, categories)

    save_data(cleaned_df, merged_df)

    print("\n🎉 CLEAN DATA COMPLETED!")


if __name__ == "__main__":
    main()