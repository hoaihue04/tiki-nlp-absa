import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# ── Raw data ──────────────────────────────────────────────────────────────────
TIKI_REVIEWS_FILE        = os.path.join(PROJECT_ROOT, "data/raw/Tiki_be_reviews.csv")
PRODUCT_DETAIL_FILE      = os.path.join(PROJECT_ROOT, "data/raw/Tiki_be_detail.csv")
PRODUCT_CATEGORY_FILE    = os.path.join(PROJECT_ROOT, "data/raw/Tiki_be_product_id.csv")

# ── Interim (preprocessing pipeline) ─────────────────────────────────────────
CLEANED_REVIEWS_FILE     = os.path.join(PROJECT_ROOT, "data/interim/cleaned_reviews.csv")
MERGED_DATA_FILE         = os.path.join(PROJECT_ROOT, "data/interim/merged_reviews_products.csv")
NORMALIZED_REVIEWS_FILE  = os.path.join(PROJECT_ROOT, "data/interim/normalized_reviews.csv")
TOKENIZED_REVIEWS_FILE   = os.path.join(PROJECT_ROOT, "data/interim/tokenized_reviews.csv")
