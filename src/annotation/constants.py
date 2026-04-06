# src/annotation/constants.py
"""
Ontology nhãn ASQP cho domain sản phẩm trẻ sơ sinh Tiki
"""

# ── Aspect Categories (Aspect#Attribute) ──────────────────────────────────────
ASPECT_CATEGORIES = [
    "PRODUCT#QUALITY",        # Chất lượng sản phẩm nói chung
    "PRODUCT#MATERIAL",       # Chất liệu, vải, nhựa...
    "PRODUCT#DESIGN",         # Kiểu dáng, màu sắc, thẩm mỹ
    "PRODUCT#SIZE",           # Kích thước, vừa vặn
    "PRODUCT#FUNCTION",       # Công năng, tính năng hoạt động
    "PRODUCT#SAFETY",         # An toàn cho bé
    "PRODUCT#DURABILITY",     # Độ bền
    "PRODUCT#VALUE",          # Giá trị so với tiền bỏ ra
    "PRODUCT#COMFORT",        # Sự thoải mái
    "PRICE#AFFORDABILITY",    # Giá cả phải chăng
    "PRICE#DISCOUNT",         # Khuyến mãi, giảm giá
    "DELIVERY#SPEED",         # Tốc độ giao hàng
    "DELIVERY#PACKAGING",     # Đóng gói, bảo quản khi vận chuyển
    "DELIVERY#ACCURACY",      # Giao đúng hàng, đúng size/màu
    "SELLER#SERVICE",         # Dịch vụ, tư vấn của người bán
    "SELLER#RESPONSIVENESS",  # Phản hồi, giải quyết khiếu nại
    "SELLER#AUTHENTICITY",    # Hàng chính hãng, không giả
]

SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# ── Đường dẫn file ────────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_FILE = os.path.join(BASE_DIR, "data", "interim", "normalized_reviews.csv")

ANNOTATED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(ANNOTATED_DIR, exist_ok=True)

OUTPUT_JSONL = os.path.join(ANNOTATED_DIR, "asqp_annotated.jsonl")   # 1 JSON object / dòng
OUTPUT_CSV   = os.path.join(ANNOTATED_DIR, "asqp_annotated_flat.csv") # flat cho verify

CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "annotation_checkpoint.txt")

# ── Tham số hệ thống ──────────────────────────────────────────────────────────
MAX_REVIEW_LENGTH   = 512    # Cắt ngắn nếu review quá dài (token)
MAX_QUADRUPLES      = 8      # Tối đa quadruples / review
REQUEST_TIMEOUT     = 30     # seconds
RETRY_ATTEMPTS      = 3
RETRY_DELAY         = 2      # seconds

# ── Tham số batch ─────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE  = 100    # Số review mặc định mỗi lần chạy (điều chỉnh qua CLI)
CHECKPOINT_INTERVAL = 50     # Lưu checkpoint mỗi N reviews