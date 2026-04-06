# 🍼 Ứng dụng NLP để Phân tích Phản hồi Khách hàng và Xây dựng Hệ thống Đề xuất Sản phẩm Dành cho Trẻ Sơ sinh trên Sàn Thương mại Điện tử Tiki

---

## 📌 Giới thiệu

Dự án nghiên cứu ứng dụng **Natural Language Processing (NLP)** để phân tích phản hồi (review) của khách hàng trên sàn thương mại điện tử **Tiki**, tập trung vào danh mục **sản phẩm dành cho trẻ sơ sinh** (tã, sữa, đồ chơi, quần áo,...).

Bài toán trọng tâm là **Aspect Sentiment Quad Prediction (ASQP)** — phân tích cảm xúc đa khía cạnh theo 4 chiều:

| Thành phần | Ý nghĩa | Ví dụ |
|---|---|---|
| **Aspect Term** | Cụm từ đề cập khía cạnh | `"chất liệu"` |
| **Aspect Category** | Nhóm khía cạnh chuẩn hóa | `PRODUCT#MATERIAL` |
| **Opinion Term** | Cụm từ thể hiện ý kiến | `"tốt"` |
| **Sentiment** | Cực tính cảm xúc | `positive` |

**Ví dụ:** Review `"chất liệu tốt bé dùng rất thoải mái"` → quadruple: `(chất liệu, PRODUCT#MATERIAL, tốt, positive)`

---

## 🎯 Mục tiêu

- Thu thập dữ liệu review sản phẩm trẻ sơ sinh từ Tiki bằng crawler
- Xây dựng pipeline annotation tự động bằng LLaMA để tạo bộ dữ liệu ASQP tiếng Việt
- Huấn luyện và so sánh 3 mô hình: **BiLSTM-CRF** (baseline), **PhoBERT**, **ViT5**
- Triển khai web demo

---

## 🗂️ Cấu trúc dự án

```
TIKI/
│
├── data/
│   ├── raw/                        # Dữ liệu thô từ Tiki (sản phẩm, reviews)
│   │   ├── Tiki_be_detail.csv
│   │   ├── Tiki_be_listing.csv
│   │   ├── Tiki_be_product_id.csv
│   │   ├── Tiki_be_reviews.csv
│   │   └── ...
│   │
│   ├── interim/                    # Dữ liệu trung gian (đã làm sạch, chuẩn hóa)
│   │   ├── cleaned_reviews.csv
│   │   ├── normalized_reviews.csv
│   │   ├── merged_reviews_products.csv
│   │   ├── human_labels_exported.json
│   │   ├── labelstudio_tasks.json
│   │   └── sampled_ids.json
│   │
│   └── processed/                  # Dữ liệu đã annotation, sẵn sàng train
│       ├── asqp_annotated.jsonl    ← File dữ liệu chính (ASQP quadruples)
│       ├── asqp_annotated_flat.csv
│       ├── annotation.log
│       └── skipped_reviews.jsonl
│
├── models/                         # Model checkpoints sau khi train
│
├── notebooks/                      # Jupyter notebooks phân tích, EDA
│
├── results/                        # Kết quả đánh giá, bảng so sánh models
│
├── src/
│   ├── crawling/                   # Scripts thu thập dữ liệu từ Tiki
│   │   ├── 1.Crawl_category_product_id.py
│   │   ├── 2.Crawl_Product_Detail.py
│   │   ├── 3.Crawl_products_listing.py
│   │   └── 4.Crawl_product_reviews.py
│   │
│   ├── data_preprocessing/         # Làm sạch và chuẩn hóa dữ liệu
│   │   ├── clean_data.py
│   │   └── normalize_text.py
│   │
│   ├── annotation/                 # Pipeline annotation tự động bằng LLaMA
│   │   ├── annotator.py            # Core annotation logic
│   │   ├── run_annotation.py       # Script chạy annotation
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── check_api_status.py
│   │   └── del_NA_review.py
│   │
│   ├── data_labeling/              # Đánh giá chất lượng annotation
│   │   ├── evaluate_human_vs_llm.py
│   │   └── prepare_data.py
│   │
│   ├── training/                   # Train và đánh giá models
│   │   ├── prepare_data.py         # Tiền xử lý cho training (BIO tagging, seq2seq)
│   │   ├── train_bilstm.py         # Train BiLSTM-CRF baseline
│   │   ├── train_phobert.py        # Fine-tune PhoBERT
│   │   ├── train_vit5.py           # Fine-tune ViT5
│   │   ├── compare_models.py       # So sánh kết quả 3 models
│   │   ├── run_all.py              # Chạy toàn bộ pipeline training
│   │   └── colab_utils.py          # Tiện ích cho Google Colab
│   │
│   └── utils/
│       └── constants.py
│
├── .env.example                    # Template biến môi trường 
├── .gitignore
├── requirements.txt
├── README.md
└── Visualize.pbix                  # Dashboard Power BI
```

---

## 🔄 Quy trình thực hiện

### Giai đoạn 1 — Thu thập dữ liệu (Crawling)

```
Tiki Website
    │
    ├─ 1. Lấy danh sách category → product_id
    ├─ 2. Crawl chi tiết từng sản phẩm
    ├─ 3. Crawl listing (tên, giá, rating,...)
    └─ 4. Crawl reviews của từng sản phẩm
```

**Scripts:** `src/crawling/`  
**Output:** `data/raw/*.csv`

Dữ liệu thu thập gồm **~XX sản phẩm** và **~XX.000 reviews** thuộc danh mục trẻ sơ sinh trên Tiki.

---

### Giai đoạn 2 — Tiền xử lý dữ liệu

- Loại bỏ review rác, spam, trùng lặp
- Chuẩn hóa văn bản tiếng Việt (lowercase, bỏ ký tự đặc biệt,...)
- Lọc review quá ngắn (< 5 ký tự) hoặc không có nội dung
- Gộp thông tin review với thông tin sản phẩm

**Scripts:** `src/data_preprocessing/`  
**Output:** `data/interim/cleaned_reviews.csv`, `normalized_reviews.csv`

---

### Giai đoạn 3 — Annotation tự động bằng LLaMA

Sử dụng **LLaMA** (via API) để tự động gán nhãn ASQP quadruples cho từng review:

```
Review text
    │
    ▼
LLaMA API (few-shot prompting)
    │
    ▼
{"aspect_term": "...", "aspect_category": "...",
 "opinion_term": "...", "sentiment": "..."}
```

**Scripts:** `src/annotation/`  
**Output:** `data/processed/asqp_annotated.jsonl`

Định dạng dữ liệu annotation:
```jsonl
{"review_id": "20192566", "text": "mình mua được giá tốt",
 "quadruples": [{"aspect_term": "giá", "aspect_category": "PRICE#AFFORDABILITY",
                 "opinion_term": "tốt", "sentiment": "positive"}]}
```

---

### Giai đoạn 4 — Đánh giá chất lượng Annotation

- So sánh kết quả annotation của LLaMA với nhãn human (Label Studio)
- Tính **Inter-Annotator Agreement** (Cohen's Kappa, F1)
- Lọc và hiệu chỉnh các mẫu chất lượng thấp

**Scripts:** `src/data_labeling/evaluate_human_vs_llm.py`

---

### Giai đoạn 5 — Huấn luyện Models

Ba models được train và so sánh:

| Model | Kiến trúc | Định dạng dữ liệu | Mục đích |
|---|---|---|---|
| **BiLSTM-CRF** | Embedding → BiLSTM → CRF | BIO tagging | Baseline (không dùng LLM) |
| **PhoBERT** | BERT pre-train tiếng Việt + classification head | BIO tagging | Main model |
| **ViT5** | T5 Seq2Seq pre-train tiếng Việt | Text-to-text | Main model |

**Chạy training:**
```bash
# Chuẩn bị dữ liệu
python src/training/prepare_data.py

# Train từng model
python src/training/train_bilstm.py
python src/training/train_phobert.py
python src/training/train_vit5.py

# So sánh kết quả
python src/training/compare_models.py

# Hoặc chạy toàn bộ pipeline
python src/training/run_all.py
```

**Metric đánh giá:** F1 Score (Exact Match Quadruple), Precision, Recall


## 🚀 Cài đặt và chạy

### Yêu cầu hệ thống

- Python 3.10+
- GPU (khuyến nghị, có thể chạy CPU)
- RAM ≥ 8GB

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/tiki-nlp-asqp.git
cd tiki-nlp-asqp
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường

```bash
# Sao chép file template
cp .env.example .env

# Mở file .env và điền API key của bạn
# LLAMA_API_KEY=your_key_here
```


## 📊 Kết quả

> Cập nhật sau khi hoàn thành training.

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| BiLSTM-CRF (Baseline) | - | - | - |
| PhoBERT-base-v2 | - | - | - |
| ViT5-base | - | - | - |

---

## 🛠️ Công nghệ sử dụng

- **Crawling:** `requests`, `BeautifulSoup`, Tiki Public API
- **NLP:** `transformers` (HuggingFace), `torch`, `torchcrf`
- **Models:** [PhoBERT](https://huggingface.co/vinai/phobert-base-v2), [ViT5](https://huggingface.co/VietAI/vit5-base)
- **Annotation:** LLaMA via API
- **Labeling UI:** Label Studio
- **Visualization:** Power BI
- **Web:** FastAPI, Uvicorn
- **Khác:** `underthesea`, `pyvi`, `pandas`, `numpy`

---

## 📁 Dữ liệu

Dữ liệu thô và dữ liệu đã annotation **không được đưa lên GitHub** do kích thước lớn và điều khoản sử dụng của Tiki.

Để tái tạo dữ liệu, chạy theo thứ tự:
```bash
python src/crawling/1.Crawl_category_product_id.py
python src/crawling/2.Crawl_Product_Detail.py
python src/crawling/3.Crawl_products_listing.py
python src/crawling/4.Crawl_product_reviews.py
python src/data_preprocessing/clean_data.py
python src/data_preprocessing/normalize_text.py
python src/annotation/run_annotation.py
```
