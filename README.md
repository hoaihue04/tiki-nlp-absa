# TikiInsight - ABSA Dashboard & AI Shopping Assistant

TikiInsight là hệ thống phân tích đánh giá sản phẩm Tiki bằng NLP, tập trung vào **Aspect-Based Sentiment Analysis (ABSA)** và **AI Shopping Assistant**. Ứng dụng giúp chuyển hàng trăm đánh giá rời rạc của người mua thành dashboard phân tích dễ hiểu và câu trả lời tư vấn mua sắm tự nhiên.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)
![PhoBERT](https://img.shields.io/badge/Model-PhoBERT-orange)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker)

## Bối Cảnh

Trên các sàn thương mại điện tử, người mua thường phải đọc rất nhiều review để trả lời những câu hỏi đơn giản:

- Sản phẩm này có đáng mua không?
- Người mua khen/chê nhiều nhất ở điểm nào?
- Có rủi ro gì về kích thước, chất liệu, an toàn, giao hàng hay bao bì không?
- Sản phẩm có phù hợp với nhu cầu cụ thể như bé da nhạy cảm, cần thấm hút tốt, cần size thoải mái không?

Review thô thường dài, trùng lặp, cảm xúc lẫn lộn và khó tổng hợp. Vì vậy, TikiInsight được xây dựng để phân tích review theo từng khía cạnh, sau đó biến kết quả phân tích thành dashboard và chatbot tư vấn dễ hiểu.


## Cách Giải Quyết

TikiInsight kết hợp 3 lớp xử lý:

1. **PhoBERT ABSA**
   - Nhận diện khía cạnh được nhắc đến trong từng câu review.
   - Phân loại sentiment theo từng khía cạnh: positive, neutral, negative.

2. **Analytics Layer**
   - Tổng hợp thống kê theo PostgreSQL: top praise, top complaint, tỷ lệ positive/negative theo aspect, review statistics.
   - Phát hiện risk flags từ các nhóm phản hồi tiêu cực.

3. **AI Shopping Assistant**

    -Trả lời 1 số câu hỏi cơ bản về sản phẩm 

## Demo

![Demo tổng quan](docs/images/demo0.png)

| User Chatbot | Seller Dashboard | Opinion Mining |
|---|---|---|
| ![Demo 1](docs/images/demo1.png) | ![Demo 2](docs/images/demo2.png) | ![Demo 3](docs/images/demo3.png) |

## Tính Năng Nổi Bật Của Web App

### User

- Chatbot hỏi đáp theo sản phẩm đã phân tích.
- Trả lời bằng tiếng Việt có dấu, thân thiện, giống tư vấn mua sắm.
- Tóm tắt ưu điểm, điểm cần lưu ý, khuyến nghị và độ tin cậy.
- Bằng chứng tham khảo được nén ngắn theo khía cạnh, ví dụ: Kích thước, Chất lượng, Bao bì, Giao hàng.

### Seller Dashboard

- Tổng quan thông tin sản phẩm Tiki: tên, giá, rating, số review.
- Thống kê số review sử dụng, tỷ lệ sentiment và số khía cạnh được nhắc đến.
- Biểu đồ stacked sentiment theo khía cạnh.
- RADAR khía cạnh sản phẩm.
- Danh sách điểm mạnh nổi bật và điểm yếu cần cải thiện.
- Bảng Opinion Mining chi tiết theo aspect, sentiment, confidence và số lượt nhắc.
- Review tiêu biểu theo nhóm tích cực, tiêu cực và trung lập.

## Các Mô Hình Phân Tích Cảm Xúc Theo Khía Cạnh

Dự án có các hướng mô hình phục vụ nghiên cứu và so sánh:

| Mô hình | Vai trò | Ghi chú |
|---|---|---|
| PhoBERT fine-tuned | Mô hình ABSA chính trong web app | Dùng để nhận diện aspect và sentiment |
| BiLSTM-CRF | Baseline sequence labeling | Phục vụ so sánh trong quá trình nghiên cứu |
| SVM + TF-IDF | Baseline truyền thống | Phục vụ so sánh với hướng deep learning |

Kết quả thực nghiệm đã ghi nhận trong project:

| Mô hình | Aspect Detection F1 | Aspect Polarity F1 |
|---|---:|---:|
| PhoBERT | 0.7975 | 0.8519 |
| BiLSTM-CRF | 0.7636 | 0.7951 |
| SVM + TF-IDF | 0.7735 | 0.8361 |

## Nhóm Khía Cạnh Được Phân Tích

Một số khía cạnh tiêu biểu:

- Chất lượng sản phẩm
- Chất liệu
- Kích thước
- Độ an toàn
- Khả năng thấm hút
- Độ thoải mái
- Độ bền
- Giá cả
- Giao hàng
- Bao bì
- Tính chính hãng

Trong backend, các aspect có thể được lưu bằng mã nội bộ như `PRODUCT#QUALITY`, nhưng khi hiển thị cho người dùng hệ thống sẽ map sang nhãn thân thiện như **Chất lượng**, **Kích thước**, **Giá cả**, **Giao hàng**.

## Pipeline Xử Lý Dữ Liệu

```text
Thu thập dữ liệu từ Tiki API
    ↓
Làm sạch review
    ↓
Tách câu và chuẩn hóa tiếng Việt
    ↓
Gán nhãn/chuẩn bị dữ liệu ABSA
    ↓
Huấn luyện và đánh giá mô hình
    ↓
Chạy PhoBERT inference trên review mới
    ↓
Tổng hợp aspect sentiment
    ↓
Lưu PostgreSQL
    ↓
Hiển thị dashboard và phục vụ chatbot
```

## Core Technologies

| Nhóm | Công nghệ |
|---|---|
| Backend | Python, FastAPI, Pydantic, SQLAlchemy |
| Frontend | HTML, CSS, JavaScript, Jinja2, Chart.js |
| NLP / Machine Learning | PhoBERT, PyTorch, Hugging Face Transformers, scikit-learn |
| LLM / RAG | Gemini/Groq API, intent detection, hybrid retrieval, evidence compression, structured JSON output |
| Database / Retrieval | PostgreSQL, Qdrant, Redis |
| Infrastructure | Docker, Docker Compose |
| Data Processing | pandas, NumPy, regex, pyvi, underthesea |



## Cài Đặt Và Chạy

### Cách 1: Docker Compose

Yêu cầu:

- Docker Desktop
- Model PhoBERT tại `models/phobert/best_model.pt`

Chạy:

```powershell
docker compose up -d --build
```

Mở ứng dụng:

```text
http://127.0.0.1:8001
```

Kiểm tra server:

```text
http://127.0.0.1:8001/api/health
```

### Cách 2: Chạy local

Tạo môi trường:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chạy FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.app:app --host 127.0.0.1 --port 8001
```

Mở:

```text
http://127.0.0.1:8001
```
