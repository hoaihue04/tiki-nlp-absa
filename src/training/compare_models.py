"""
compare_models.py
=================
So sánh kết quả của 3 models trên tập test.
Tạo bảng so sánh và biểu đồ.

Cách chạy (sau khi train xong 3 models):
  cd TIKI
  python src/training/compare_models.py
"""

import os, json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # không cần GUI

MODELS = {
    "BiLSTM+CRF": "models/bilstm/results.json",
    "PhoBERT":    "models/phobert/results.json",
    "ViT5":       "models/vit5/results.json",
}
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_results():
    rows = []
    for model_name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"⚠️ Chưa có kết quả cho {model_name}: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        row = {"Model": model_name}
        # Chuẩn hóa tên metric từ các model khác nhau
        if "f1" in data:          # ViT5
            row["F1 (overall)"]  = round(data["f1"] * 100, 2)
            row["Precision"]     = round(data.get("precision", 0) * 100, 2)
            row["Recall"]        = round(data.get("recall", 0) * 100, 2)
            row["Sentiment F1"]  = "-"
            row["Category F1"]   = "-"
        elif "test_combined_f1" in data:  # PhoBERT
            row["F1 (overall)"]  = round(data["test_combined_f1"] * 100, 2)
            row["Precision"]     = "-"
            row["Recall"]        = "-"
            row["Sentiment F1"]  = round(data["test_sentiment_f1"] * 100, 2)
            row["Category F1"]   = round(data["test_category_f1"] * 100, 2)
        elif "test_f1" in data:   # BiLSTM
            row["F1 (overall)"]  = round(data["test_f1"] * 100, 2)
            row["Precision"]     = "-"
            row["Recall"]        = "-"
            row["Sentiment F1"]  = "-"
            row["Category F1"]   = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(df):
    """Tạo biểu đồ so sánh F1."""
    models = df["Model"].tolist()
    f1s    = [float(v) if v != "-" else 0 for v in df["F1 (overall)"].tolist()]

    colors = ["#6c757d", "#007bff", "#FF6B35"][:len(models)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, f1s, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("So sánh F1 Score giữa 3 Models (ASQP)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"💾 Biểu đồ lưu tại: {path}")
    plt.close()


def main():
    print("=" * 50)
    print("SO SÁNH KẾT QUẢ 3 MODELS")
    print("=" * 50)

    df = load_results()
    if df.empty:
        print("❌ Chưa có kết quả nào. Hãy train các models trước!")
        return

    print("\n📊 Bảng so sánh:")
    print(df.to_string(index=False))

    # Lưu CSV
    csv_path = os.path.join(OUT_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Bảng lưu tại: {csv_path}")

    # Biểu đồ
    plot_comparison(df)

    # In nhận xét
    print("\n📝 Nhận xét:")
    print("  - BiLSTM+CRF: Model baseline đơn giản, train nhanh")
    print("  - PhoBERT:    Fine-tune mạnh, hiểu ngữ nghĩa tiếng Việt tốt")
    print("  - ViT5:       Generative, end-to-end, thường cho kết quả tốt nhất")


if __name__ == "__main__":
    main()
