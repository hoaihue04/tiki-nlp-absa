# Model Artifacts

Các file model nặng không nên commit trực tiếp lên GitHub. Hãy tải hoặc huấn luyện riêng, sau đó đặt vào đúng đường dẫn bên dưới.

## Required for Web App

| File | Vai trò | Ghi chú |
|---|---|---|
| `phobert/best_model.pt` | PhoBERT fine-tuned cho ABSA inference | Cần có để phân tích review trong web app |

## Optional / Research

| File | Vai trò |
|---|---|
| `bilstm/best_model.pt` | Baseline BiLSTM-CRF |
| `svm/svm_model.pkl` | Baseline SVM + TF-IDF |
| `phobert/base_model/` | Tokenizer/base model files nếu chạy local offline |
| `phobert/tokenizer/` | Tokenizer files cho PhoBERT |

## Suggested Setup

```text
models/
├── README.md
├── phobert/
│   ├── best_model.pt
│   ├── base_model/
│   └── tokenizer/
├── bilstm/
│   └── best_model.pt
└── svm/
    └── svm_model.pkl
```

Nếu public repo, nên dùng một trong các cách sau:

- Git LFS cho model artifact lớn.
- Google Drive/Hugging Face để lưu model và ghi link tải trong README.
- Script tải model riêng, ví dụ `scripts/download_models.py`.

File `.gitignore` hiện đã ignore `models/**/*.pt` và `models/**/*.pkl` để tránh commit model nặng.
