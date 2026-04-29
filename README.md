# 🍼 TikiInsight — Phân Tích Cảm Xúc & Hệ Thống Đề Xuất Sản Phẩm Sơ Sinh

> **Ứng dụng NLP để phân tích phản hồi khách hàng và xây dựng hệ thống đề xuất sản phẩm dành cho trẻ sơ sinh trên sàn thương mại điện tử Tiki**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![PhoBERT](https://img.shields.io/badge/Model-PhoBERT-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Giới thiệu

**TikiInsight** là hệ thống phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis - ABSA) và đề xuất sản phẩm thông minh, được xây dựng cho ngành hàng mẹ & bé trên nền tảng Tiki.vn.

Hệ thống cho phép người dùng dán link sản phẩm Tiki và nhận ngay:
- 📊 **Phân tích cảm xúc** theo từng khía cạnh sản phẩm (chất lượng, chất liệu, giá cả, an toàn, giao hàng...)
- 🎯 **Điểm ABSA tổng hợp** và biểu đồ radar so sánh với sản phẩm cùng phân khúc
- 💡 **Gợi ý Top-5 sản phẩm** thay thế theo thuật toán Hybrid Recommender

---

## 🖼️ Demo
![Demo 0](docs/images/demo0.png)
| Tổng quan phân tích | Điểm mạnh & điểm yếu | Opinion Mining chi tiết |
|---|---|---|
| ![Demo 1](docs/images/demo1.png) | ![Demo 2](docs/images/demo2.png) | ![Demo 3](docs/images/demo3.png) |

---

## 🏗️ Kiến trúc hệ thống

```
TIKI/
│
├── app/                         # 🌐 Web Demo (FastAPI)
│   ├── src/                     # Logic xử lý backend
│   ├── static/                  # CSS, JS, assets
│   ├── templates/               # HTML templates (Jinja2)
│   └── app.py                   # Entry point FastAPI
│
├── data/                        # 📊 Dữ liệu
│   ├── raw/                     # Dữ liệu thô từ Tiki
│   ├── interim/                 # Dữ liệu trung gian
│   ├── processed/               # Dữ liệu đã xử lý (20,446 reviews)
│   └── training/                # Tập train/val/test (split 70:15:15)
│
├── models/                      # 🤖 Các mô hình ML/DL
│   ├── bilstm/                  # BiLSTM-CRF (sequence labeling)
│   ├── phobert/                 # PhoBERT fine-tuned (ABSA chính)
│   ├── recommendation/          # Hybrid Recommender
│   └── svm/                     # SVM+TF-IDF (baseline)
│
├── src/                         # 🧠 Core pipeline
│   ├── annotation/              # LLM Annotation (Groq + LLaMA 3.3 70B)
│   ├── crawling/                # Crawl dữ liệu từ Tiki API
│   ├── data_labeling/           # Label Studio integration
│   ├── data_preprocessing/      # Tiền xử lý văn bản tiếng Việt
│   ├── recommendation/          # Thuật toán gợi ý hybrid
│   ├── training/                # Training pipeline
│   └── utils/                   # Hàm tiện ích
│
├── notebooks/                   # Jupyter Notebook (EDA, thử nghiệm)
├── results/                     # Kết quả output (metrics, predict)
├── checkpoints/                 # Model checkpoints
└── requirements.txt
```

---

## 🤖 Các mô hình sử dụng

### Phân tích cảm xúc theo khía cạnh (ABSA)

| Mô hình | AD F1-score | AP F1-score | Ghi chú |
|---|---|---|---|
| **PhoBERT** ✅ | **0.7975** | **0.8519** | Mô hình chính, fine-tuned |
| BiLSTM-CRF | 0.7636 | 0.7951 | BIO sequence labeling |
| SVM + TF-IDF | 0.7735 | 0.8361 | Baseline |

> AD = Aspect Detection (nhận diện khía cạnh) · AP = Aspect Polarity (phân loại cảm xúc)

### Hệ thống gợi ý (Hybrid Recommender)

Công thức tính điểm lai:

```
S_Hybrid = w_ABSA × S_ABSA + w_CBF × S_CBF + w_Category × S_Category
```

| Phương pháp | Precision@5 | Recall@20 |
|---|---|---|
| 2 thành phần (ABSA + CBF) | 0.2769 | 0.5229 |
| **3 thành phần (+ Category)** ✅ | **1.0000** | 0.1844 |

### 17 Khía cạnh được phân tích

`PRODUCT#QUALITY` · `PRODUCT#MATERIAL` · `PRODUCT#DESIGN` · `PRODUCT#SIZE` · `PRODUCT#FUNCTION` · `PRODUCT#COMFORT` · `PRODUCT#SAFETY` · `PRODUCT#VALUE` · `PRODUCT#DURABILITY` · `PRICE#AFFORDABILITY` · `SELLER#SERVICE` · `SELLER#AUTHENTICITY` · `DELIVERY#SPEED` · `DELIVERY#CONDITION` · `PACKAGING#QUALITY` · `PROMOTION#DEALS` · `OVERALL#SATISFACTION`

---

## ⚙️ Cài đặt & Chạy

### Yêu cầu hệ thống

- Python 3.10+
- RAM ≥ 8GB (khuyến nghị 16GB để chạy PhoBERT)
- GPU (tùy chọn, tăng tốc inference)

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/tiki-insight.git
cd tiki-insight
```

### 2. Tạo virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng web

```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

Truy cập: [http://localhost:8000](http://localhost:8000)

---

## 🔄 Pipeline xử lý dữ liệu

```
Thu thập (Crawl Tiki API)
        ↓
Làm sạch văn bản tiếng Việt
(chuẩn hóa Unicode, xóa HTML/URL/emoji, loại trùng lặp)
        ↓
Gán nhãn tự động (Groq + LLaMA 3.3 70B)
→ Trích xuất ABSA quadruples: (aspect_category, aspect_term, opinion_term, sentiment)
        ↓
Kiểm tra chất lượng annotation
→ Cohen's Kappa: 0.86 (Aspect), 0.89 (Sentiment)
        ↓
Chuẩn bị dữ liệu training
→ BiLSTM-CRF: định dạng BIO (.txt)
→ PhoBERT + SVM: định dạng CSV (text + 17 nhãn)
        ↓
Huấn luyện & đánh giá mô hình
        ↓
Deploy FastAPI Web App
```

---

## 📊 Quy trình gán nhãn

Dự án sử dụng **LLM Annotation** (Groq + LLaMA 3.3 70B) để tự động hóa quá trình gán nhãn ABSA trên ~20.000 reviews:

1. Xây dựng structured prompt với danh sách 17 nhãn khía cạnh hợp lệ
2. LLM trả về danh sách quadruples theo định dạng JSONL
3. Retry tự động với các review không sinh được quadruple hợp lệ
4. Kiểm tra chất lượng bằng Label Studio trên 200 mẫu ngẫu nhiên

**Kết quả đánh giá annotation:**
- F1-score (flexible): **81.6%**
- Cohen's Kappa (Aspect): **0.8607** — Rất tốt
- Cohen's Kappa (Sentiment): **0.8899** — Rất tốt

---

## 🚀 Tính năng nổi bật của Web App

- **Nhập URL sản phẩm Tiki** 
- **Biểu đồ cột stacked** phân phối cảm xúc theo 17 khía cạnh
- **Biểu đồ Radar** so sánh với sản phẩm gợi ý #1
- **Bảng Opinion Mining** chi tiết với độ tin cậy từng nhận xét
- **Top 3 đánh giá tiêu biểu** (tích cực nhất / tiêu cực nhất / ngẫu nhiên)
- **Top 5 sản phẩm được gợi ý** 

---

## 📦 Dependencies chính

```
fastapi
uvicorn
transformers
torch
scikit-learn
underthesea
pyvi
pandas
numpy
selenium / requests  # crawling
```

Xem đầy đủ tại [`requirements.txt`](requirements.txt)

---

## 👩‍💻 Tác giả

| | |
|---|---|
| **Sinh viên** | Trần Hoài Huệ |
| **Chuyên ngành** | Khoa học dữ liệu và phân tích kinh doanh |
| **Trường** | Đại học Kinh tế Đà Nẵng |
| **Năm** | 2026 |

---

