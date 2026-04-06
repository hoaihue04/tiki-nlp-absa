import json
import pandas as pd
from pathlib import Path

# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

JSONL_PATH = BASE_DIR / "data" / "processed" / "asqp_annotated.jsonl"
CSV_PATH = BASE_DIR / "data" / "processed" / "asqp_annotated_flat.csv"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "annotation_checkpoint.txt"

# =========================
# STEP 1: LOAD + FILTER JSONL 
# =========================
filtered_lines = []
valid_ids = set()

total = 0
removed = 0

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        line_strip = line.strip()

        if not line_strip:
            continue

        try:
            item = json.loads(line_strip)
        except json.JSONDecodeError:
            print(f"[WARNING] Lỗi parse JSON dòng {total}")
            continue

        # 👉 CHỈ check quadruples có tồn tại và không rỗng
        if item.get("quadruples"):
            filtered_lines.append(line)  # giữ nguyên dòng gốc
            valid_ids.add(str(item["review_id"]))
        else:
            removed += 1

print(f"[INFO] Tổng review: {total}")
print(f"[INFO] Bị xóa (không có quadruples): {removed}")
print(f"[INFO] Giữ lại: {len(filtered_lines)}")

# =========================
# STEP 2: SAVE JSONL (GIỮ NGUYÊN FORMAT)
# =========================
with open(JSONL_PATH, "w", encoding="utf-8") as f:
    f.writelines(filtered_lines)

print("[OK] Updated JSONL (giữ nguyên format)")

# =========================
# STEP 3: FILTER CSV
# =========================
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)

    if "review_id" in df.columns:
        df["review_id"] = df["review_id"].astype(str)
        df_filtered = df[df["review_id"].isin(valid_ids)]

        df_filtered.to_csv(CSV_PATH, index=False, encoding="utf-8")
        print(f"[OK] Updated CSV: {len(df_filtered)} rows")
    else:
        print("[WARNING] CSV không có cột review_id")
else:
    print("[WARNING] Không tìm thấy CSV")

# =========================
# STEP 4: FILTER CHECKPOINT
# =========================
if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered_checkpoint = [line for line in lines if line.strip() in valid_ids]

    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        f.writelines(filtered_checkpoint)

    print(f"[OK] Updated checkpoint: {len(filtered_checkpoint)} IDs")
else:
    print("[WARNING] Không tìm thấy checkpoint")

print("\n[DONE] Cleaned dataset successfully!")