"""
=============================================================
BƯỚC 1: CHUẨN BỊ DỮ LIỆU CHO LABEL STUDIO
=============================================================
Mục đích : Lấy mẫu ngẫu nhiên từ file JSONL (20.000 dòng)
           và chuyển sang định dạng JSON mà Label Studio đọc được.

Đặt file : TIKI/src/data_labeling/step1_prepare_data.py

Cách chạy (từ thư mục gốc TIKI/):
    python src/data_labeling/step1_prepare_data.py

Kết quả  : TIKI/data/interim/labelstudio_tasks.json
           TIKI/data/interim/sampled_ids.json
=============================================================
"""

import json
import random
import os
from datetime import datetime

# ----------------------------------------------------------
# CẤU HÌNH — chỉnh sửa ở đây nếu cần
# ----------------------------------------------------------
INPUT_FILE  = "data/processed/asqp_annotated.jsonl"   # file JSONL đầu vào
OUTPUT_DIR  = "data/interim"                           # thư mục lưu kết quả
N_SAMPLES   = 200    # số review cần label thủ công
                     # Với 20.000 dòng, 200 mẫu = 1% → đủ để đánh giá tin cậy
RANDOM_SEED = 42     # giữ nguyên để tái lặp kết quả


# ----------------------------------------------------------
# BƯỚC 1A: ĐỌC FILE JSONL
# ----------------------------------------------------------
def load_jsonl(path):
    """Đọc file JSONL — mỗi dòng là 1 JSON object"""
    print(f"  Đang đọc: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:          # bỏ qua dòng trống
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [CẢNH BÁO] Dòng {line_num} lỗi JSON: {e}")
    print(f"  → Đọc được {len(data):,} review")
    return data


# ----------------------------------------------------------
# BƯỚC 1B: LẤY MẪU THÔNG MINH
# ----------------------------------------------------------
def stratified_sample(data, n_samples, seed):
    """
    Lấy mẫu theo tỷ lệ để đảm bảo đánh giá đa dạng:
      - 10% conflict (review mâu thuẫn cảm xúc — LLM hay sai)
      - 20% có nhiều quadruple (≥3 — review phức tạp)
      - 70% bình thường
    """
    random.seed(seed)

    conflict   = [r for r in data if r.get("conflict", False)]
    multi_quad = [r for r in data if not r.get("conflict", False)
                  and len(r.get("quadruples", [])) >= 3]
    normal     = [r for r in data if not r.get("conflict", False)
                  and len(r.get("quadruples", [])) < 3]

    n_conflict = min(int(n_samples * 0.10), len(conflict))
    n_multi    = min(int(n_samples * 0.20), len(multi_quad))
    n_normal   = n_samples - n_conflict - n_multi

    sampled = (
        random.sample(conflict,   n_conflict) +
        random.sample(multi_quad, min(n_multi, len(multi_quad))) +
        random.sample(normal,     min(n_normal, len(normal)))
    )
    random.shuffle(sampled)

    print(f"  → Đã chọn {len(sampled)} review:")
    print(f"     Conflict     : {n_conflict}")
    print(f"     Multi-quad   : {n_multi}")
    print(f"     Bình thường  : {n_normal}")
    return sampled


# ----------------------------------------------------------
# BƯỚC 1C: CHUYỂN SANG ĐỊNH DẠNG LABEL STUDIO
# ----------------------------------------------------------
def to_labelstudio_format(sampled):
    """
    Label Studio yêu cầu list các "task", mỗi task có key "data".
    Ta thêm "llm_display" để annotator thấy nhãn LLM làm tham khảo.
    """
    tasks = []
    for review in sampled:
        quads = review.get("quadruples", [])

        # Tạo chuỗi hiển thị nhãn LLM dễ đọc
        lines = []
        for i, q in enumerate(quads, 1):
            lines.append(
                f"[{i}] {q.get('aspect_category','?')} | "
                f"aspect={q.get('aspect_term','NULL')} | "
                f"opinion={q.get('opinion_term','?')} | "
                f"{q.get('sentiment','?')}"
            )
        llm_display = "\n".join(lines) if lines else "(LLM không tìm thấy quadruple)"

        tasks.append({
            "data": {
                "review_id"     : str(review["review_id"]),
                "text"          : review["text"],
                "llm_quad_count": len(quads),
                "llm_conflict"  : review.get("conflict", False),
                "llm_display"   : llm_display,
            }
        })
    return tasks


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    print("=" * 55)
    print("  BƯỚC 1: CHUẨN BỊ DỮ LIỆU CHO LABEL STUDIO")
    print("=" * 55)

    # Kiểm tra file đầu vào tồn tại
    if not os.path.exists(INPUT_FILE):
        print(f"\n[LỖI] Không tìm thấy file: {INPUT_FILE}")
        print("  Hãy kiểm tra lại đường dẫn trong phần CẤU HÌNH ở đầu file.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/3] Đọc dữ liệu...")
    data = load_jsonl(INPUT_FILE)

    print("\n[2/3] Lấy mẫu...")
    sampled = stratified_sample(data, N_SAMPLES, RANDOM_SEED)

    print("\n[3/3] Chuyển định dạng và lưu file...")
    tasks = to_labelstudio_format(sampled)

    # Lưu file cho Label Studio
    tasks_path = os.path.join(OUTPUT_DIR, "labelstudio_tasks.json")
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"  → Đã lưu: {tasks_path}")

    # Lưu danh sách ID đã chọn (để tra cứu sau)
    ids_path = os.path.join(OUTPUT_DIR, "sampled_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump({
            "sampled_at" : datetime.now().isoformat(),
            "n_samples"  : len(sampled),
            "seed"       : RANDOM_SEED,
            "review_ids" : [r["review_id"] for r in sampled],
        }, f, ensure_ascii=False, indent=2)
    print(f"  → Đã lưu: {ids_path}")

    print("\n✅ XONG! Tiếp theo: import file labelstudio_tasks.json vào Label Studio.")
    print("   Xem hướng dẫn BƯỚC 2 trong README hoặc tài liệu đi kèm.\n")


if __name__ == "__main__":
    main()