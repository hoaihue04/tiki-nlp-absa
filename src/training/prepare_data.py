"""
prepare_data.py
===============
Đọc asqp_annotated.jsonl → tạo dữ liệu cho 3 models + tính class weights.

Cách chạy:
  cd TIKI
  python src/training/prepare_data.py
"""

import json, os, random, csv
from collections import Counter

# ─── Đường dẫn ───────────────────────────────────────────────────────────────
JSONL_PATH  = "data/processed/asqp_annotated.jsonl"
OUT_DIR     = "data/training"
SEED        = 42
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# 1. ĐỌC DỮ LIỆU
# ════════════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    valid, skipped = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("quadruples"):
                valid.append(obj)
            else:
                skipped.append(obj)
    print(f"✅ Đọc xong: {len(valid)} reviews có quadruple | "
          f"{len(skipped)} NO_ASPECT bỏ qua")
    return valid


def split_data(data):
    random.shuffle(data)
    n       = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train   = data[:n_train]
    val     = data[n_train : n_train + n_val]
    test    = data[n_train + n_val :]
    print(f"📊 Split: train={len(train)} | val={len(val)} | test={len(test)}")
    return train, val, test


# ════════════════════════════════════════════════════════════════════════════
# 2. TÍNH CLASS WEIGHTS
# ════════════════════════════════════════════════════════════════════════════
#
# Công thức balanced:  weight[c] = N_total / (N_classes × N_c)
#
# Với phân phối thực tế:
#   positive: 80.15% →  N_c lớn  → weight nhỏ  ≈ 0.42
#   negative: 17.11% →  N_c vừa  → weight vừa  ≈ 1.95
#   neutral :  2.74% →  N_c nhỏ  → weight lớn  ≈ 12.17
#
# Kết quả: CrossEntropyLoss nhân weight vào mỗi sample
# → model bị "phạt nặng" hơn khi đoán sai neutral

def compute_class_weights(train_data, label_key, all_labels):
    """
    Tính class weight từ tập TRAIN (không dùng val/test để tránh leakage).

    Trả về:
      weights : dict {label_name: float}
      counts  : Counter {label_name: int}
    """
    counts    = Counter(
        q[label_key]
        for item in train_data
        for q in item["quadruples"]
    )
    n_total   = sum(counts.values())
    n_classes = len(all_labels)
    weights   = {}

    for lbl in all_labels:
        n_c = counts.get(lbl, 0)
        if n_c == 0:
            weights[lbl] = 0.0
            print(f"  ⚠️  '{lbl}' không có mẫu train → weight = 0")
        else:
            weights[lbl] = n_total / (n_classes * n_c)

    return weights, counts


def print_weight_table(weights, counts, title):
    total = sum(counts.values())
    print(f"\n  {title}:")
    print(f"  {'Label':<36} {'Count':>7}  {'Ratio':>7}  {'Weight':>8}")
    print(f"  {'-'*64}")
    for lbl, w in sorted(weights.items(), key=lambda x: -x[1]):
        c     = counts.get(lbl, 0)
        ratio = c / total * 100 if total else 0
        bar   = "█" * max(1, int(ratio / 5))
        print(f"  {lbl:<36} {c:>7d}  {ratio:>6.2f}%  {w:>8.4f}  {bar}")


# ════════════════════════════════════════════════════════════════════════════
# 3. ĐỊNH DẠNG ViT5
# ════════════════════════════════════════════════════════════════════════════

SENTIMENT_VN = {
    "positive": "tích cực",
    "negative": "tiêu cực",
    "neutral":  "trung lập",
}

def quads_to_vit5(quadruples):
    return " | ".join(
        f"({q['aspect_term']}, {q['aspect_category']}, "
        f"{q['opinion_term']}, {SENTIMENT_VN.get(q['sentiment'], q['sentiment'])})"
        for q in quadruples
    )


def save_vit5(split_name, data):
    path = os.path.join(OUT_DIR, f"vit5_{split_name}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["review_id", "input_text", "target_text"])
        w.writeheader()
        for item in data:
            w.writerow({
                "review_id":   item["review_id"],
                "input_text":  item["text"],
                "target_text": quads_to_vit5(item["quadruples"]),
            })
    print(f"  💾 vit5_{split_name}.csv  ({len(data)} rows)")


# ════════════════════════════════════════════════════════════════════════════
# 4. ĐỊNH DẠNG PhoBERT
# ════════════════════════════════════════════════════════════════════════════

def save_phobert(split_name, data):
    path    = os.path.join(OUT_DIR, f"phobert_{split_name}.csv")
    missing = 0
    fields  = [
        "review_id", "text",
        "aspect_term",  "aspect_start",  "aspect_end",
        "opinion_term", "opinion_start", "opinion_end",
        "aspect_category", "sentiment",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for item in data:
            text = item["text"]
            for q in item["quadruples"]:
                a_s = text.lower().find(q["aspect_term"].lower())
                o_s = text.lower().find(q["opinion_term"].lower())
                if a_s == -1 or o_s == -1:
                    missing += 1
                w.writerow({
                    "review_id":       item["review_id"],
                    "text":            text,
                    "aspect_term":     q["aspect_term"],
                    "aspect_start":    a_s,
                    "aspect_end":      a_s + len(q["aspect_term"])  if a_s != -1 else -1,
                    "opinion_term":    q["opinion_term"],
                    "opinion_start":   o_s,
                    "opinion_end":     o_s + len(q["opinion_term"]) if o_s != -1 else -1,
                    "aspect_category": q["aspect_category"],
                    "sentiment":       q["sentiment"],
                })
    print(f"  💾 phobert_{split_name}.csv  ({len(data)} reviews"
          + (f", {missing} term không tìm được vị trí" if missing else "") + ")")


# ════════════════════════════════════════════════════════════════════════════
# 5. ĐỊNH DẠNG BiLSTM
# ════════════════════════════════════════════════════════════════════════════

def bio_tag(tokens, term):
    term_toks = term.strip().split()
    n         = len(term_toks)
    labels    = [None] * len(tokens)
    for i in range(len(tokens) - n + 1):
        if [t.lower() for t in tokens[i:i+n]] == [t.lower() for t in term_toks]:
            labels[i] = "B"
            for j in range(1, n):
                labels[i+j] = "I"
    return labels


def quads_to_bio(text, quadruples):
    tokens  = text.strip().split()
    if not tokens:
        return []
    asp_lbl = ["O"] * len(tokens)
    opn_lbl = ["O"] * len(tokens)
    sent_lbl = quadruples[0]["sentiment"]       if quadruples else "neutral"
    cat_lbl  = quadruples[0]["aspect_category"] if quadruples else "GENERAL"

    for q in quadruples:
        for i, l in enumerate(bio_tag(tokens, q["aspect_term"])):
            if l == "B":                asp_lbl[i] = "B-ASP"
            elif l == "I" and asp_lbl[i] == "O": asp_lbl[i] = "I-ASP"
        for i, l in enumerate(bio_tag(tokens, q["opinion_term"])):
            if l == "B":                opn_lbl[i] = "B-OPN"
            elif l == "I" and opn_lbl[i] == "O": opn_lbl[i] = "I-OPN"

    return [
        (tokens[i],
         asp_lbl[i] if asp_lbl[i] != "O" else opn_lbl[i],
         sent_lbl, cat_lbl)
        for i in range(len(tokens))
    ]


def save_bilstm(split_name, data):
    path  = os.path.join(OUT_DIR, f"bilstm_{split_name}.txt")
    n_tok = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("token\tbio_label\tsentiment\tcategory\n")
        for item in data:
            rows = quads_to_bio(item["text"], item["quadruples"])
            if not rows:
                continue
            for tok, bio, sent, cat in rows:
                f.write(f"{tok}\t{bio}\t{sent}\t{cat}\n")
                n_tok += 1
            f.write("\n")
    print(f"  💾 bilstm_{split_name}.txt  ({len(data)} reviews, {n_tok} tokens)")


# ════════════════════════════════════════════════════════════════════════════
# 6. THỐNG KÊ
# ════════════════════════════════════════════════════════════════════════════

def print_stats(data, name=""):
    sents   = Counter(q["sentiment"]       for i in data for q in i["quadruples"])
    cats    = Counter(q["aspect_category"] for i in data for q in i["quadruples"])
    total_q = sum(sents.values())

    print(f"\n{'='*52}")
    print(f"📊 SENTIMENT DISTRIBUTION ({name.upper()})")
    print(f"{'='*52}")
    for s, c in sorted(sents.items(), key=lambda x: -x[1]):
        bar = "█" * int(c / total_q * 30)
        print(f"  {s:<12}: {c:>6} ({c/total_q*100:5.2f}%)  {bar}")
    print(f"\n  📌 Tổng quadruples: {total_q} | Reviews: {len(data)}")

    print(f"\n  Top 10 categories:")
    for cat, cnt in cats.most_common(10):
        print(f"    {cat:<35} {cnt:>5}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("CHUẨN BỊ DỮ LIỆU ASQP + TÍNH CLASS WEIGHTS")
    print("=" * 60)

    # 1. Đọc & thống kê
    data = load_jsonl(JSONL_PATH)
    print_stats(data, "toàn bộ")

    # 2. Chia split
    train, val, test = split_data(data)

    # 3. Thu thập labels từ TOÀN BỘ dữ liệu
    all_categories = sorted({
        q["aspect_category"]
        for item in data
        for q in item["quadruples"]
    })
    all_sentiments = ["positive", "negative", "neutral"]

    # 4. ── TÍNH CLASS WEIGHTS TỪ TẬP TRAIN ──────────────────────────────────
    print(f"\n{'='*52}")
    print("⚖️  TÍNH CLASS WEIGHTS (chỉ từ tập TRAIN)")
    print(f"{'='*52}")

    sent_weights, sent_counts = compute_class_weights(
        train, "sentiment", all_sentiments
    )
    print_weight_table(sent_weights, sent_counts, "Sentiment weights")

    cat_weights, cat_counts = compute_class_weights(
        train, "aspect_category", all_categories
    )
    print_weight_table(cat_weights, cat_counts, "Category weights")

    # 5. Lưu label_map.json (gồm cả weights)
    label_map = {
        "aspect_categories": all_categories,
        "sentiments":        all_sentiments,
        "bio_labels":        ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"],
        # sent_weights_list[i] = weight của sentiments[i]
        "sent_weights_list": [sent_weights[s] for s in all_sentiments],
        # cat_weights_list[i]  = weight của aspect_categories[i]
        "cat_weights_list":  [cat_weights[c]  for c in all_categories],
    }
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n  📋 Sentiment weights đã lưu vào label_map.json:")
    for i, s in enumerate(all_sentiments):
        print(f"     [{i}] {s:<12} → {label_map['sent_weights_list'][i]:.4f}")

    # 6. Lưu file dữ liệu
    print(f"\n{'='*52}")
    print("💾 LƯU FILE DỮ LIỆU")
    print(f"{'='*52}")

    print("\n🔧 ViT5:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        save_vit5(name, split)

    print("\n🔧 PhoBERT:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        save_phobert(name, split)

    print("\n🔧 BiLSTM:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        save_bilstm(name, split)

    print(f"\n{'='*60}")
    print(f"✅ Hoàn tất! Tất cả file lưu tại: {OUT_DIR}/")
    print(f"   label_map.json  — labels + class weights")
    print(f"   phobert_*.csv   — dùng cho train_phobert.py")
    print(f"   vit5_*.csv      — dùng cho train_vit5.py")
    print(f"   bilstm_*.txt    — dùng cho train_bilstm.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()