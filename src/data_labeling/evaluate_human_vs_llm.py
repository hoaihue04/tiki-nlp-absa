"""
=============================================================
evaluate_human_vs_llm.py
=============================================================
Đánh giá chất lượng annotation LLM so với nhãn thủ công (human).

Xử lý đặc biệt:
  - Human label dùng lẫn lộn alias tiếng Việt ("Tốc độ giao hàng")
    và code gốc ("DELIVERY#SPEED") → tự động chuẩn hóa về code gốc
  - LLM label ở dạng chuỗi trong trường llm_display → parse regex

Đặt file : TIKI/src/data_labeling/evaluate_human_vs_llm.py

Cách chạy (từ thư mục gốc TIKI/):
    python src/data_labeling/evaluate_human_vs_llm.py

Đầu vào:
    data/interim/human_labels_exported.json   ← export từ Label Studio
    data/interim/labelstudio_tasks.json       ← file tasks gốc (chứa LLM display)

Đầu ra:
    results/evaluation_report.json
    results/evaluation_report.txt
=============================================================
"""

import json
import re
import os
from collections import defaultdict, Counter
from datetime import datetime

# ── Thư viện tùy chọn ──────────────────────────────────────
try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[CẢNH BÁO] sklearn chưa cài → bỏ qua Cohen's Kappa.")
    print("           Cài bằng: pip install scikit-learn\n")

# ================================================================
#  CẤU HÌNH — chỉnh đường dẫn nếu cần
# ================================================================
HUMAN_FILE = "data/interim/human_labels_exported.json"
LLM_FILE   = "data/interim/labelstudio_tasks.json"
OUTPUT_DIR = "results"

# ================================================================
#  MAPPING: tất cả alias tiếng Việt → code chuẩn
#  (được kiểm tra thực tế từ dữ liệu)
# ================================================================
ALIAS_TO_CODE = {
    # Alias tiếng Việt (từ label studio config XML)
    "Chất lượng sản phẩm"  : "PRODUCT#QUALITY",
    "Chất liệu"             : "PRODUCT#MATERIAL",
    "Kiểu dáng / màu sắc"  : "PRODUCT#DESIGN",
    "Kích thước"            : "PRODUCT#SIZE",
    "Công năng / tính năng" : "PRODUCT#FUNCTION",
    "An toàn cho bé"        : "PRODUCT#SAFETY",
    "Độ bền sản phẩm"       : "PRODUCT#DURABILITY",
    "Giá trị đồng tiền"     : "PRODUCT#VALUE",
    "Sự thoải mái"          : "PRODUCT#COMFORT",
    "Giá cả phải chăng"     : "PRICE#AFFORDABILITY",
    "Khuyến mãi / giảm giá" : "PRICE#DISCOUNT",
    "Tốc độ giao hàng"      : "DELIVERY#SPEED",
    "Đóng gói"              : "DELIVERY#PACKAGING",
    "Giao đúng hàng"        : "DELIVERY#ACCURACY",
    "Dịch vụ người bán"     : "SELLER#SERVICE",
    "Phản hồi người bán"    : "SELLER#RESPONSIVENESS",
    "Hàng chính hãng"       : "SELLER#AUTHENTICITY",
    # Code gốc → giữ nguyên
    "PRODUCT#QUALITY"       : "PRODUCT#QUALITY",
    "PRODUCT#MATERIAL"      : "PRODUCT#MATERIAL",
    "PRODUCT#DESIGN"        : "PRODUCT#DESIGN",
    "PRODUCT#SIZE"          : "PRODUCT#SIZE",
    "PRODUCT#FUNCTION"      : "PRODUCT#FUNCTION",
    "PRODUCT#SAFETY"        : "PRODUCT#SAFETY",
    "PRODUCT#DURABILITY"    : "PRODUCT#DURABILITY",
    "PRODUCT#VALUE"         : "PRODUCT#VALUE",
    "PRODUCT#COMFORT"       : "PRODUCT#COMFORT",
    "PRICE#AFFORDABILITY"   : "PRICE#AFFORDABILITY",
    "PRICE#DISCOUNT"        : "PRICE#DISCOUNT",
    "DELIVERY#SPEED"        : "DELIVERY#SPEED",
    "DELIVERY#PACKAGING"    : "DELIVERY#PACKAGING",
    "DELIVERY#ACCURACY"     : "DELIVERY#ACCURACY",
    "SELLER#SERVICE"        : "SELLER#SERVICE",
    "SELLER#RESPONSIVENESS" : "SELLER#RESPONSIVENESS",
    "SELLER#AUTHENTICITY"   : "SELLER#AUTHENTICITY",
}

SKIP_VALUES      = {"__SKIP__", "(bỏ trống slot này)", "(Bỏ trống slot này)", ""}
VALID_CATEGORIES = list(dict.fromkeys(ALIAS_TO_CODE.values()))  # giữ thứ tự, không trùng
VALID_SENTIMENTS = ["positive", "negative", "neutral"]


# ================================================================
#  BƯỚC 1: PARSE HUMAN LABELS
# ================================================================
def parse_human_labels(path: str):
    """
    Đọc file export Label Studio.
    Xử lý:
      - Chuẩn hóa alias tiếng Việt → code chuẩn
      - Gom slot cat1/asp1/opi1/sent1 ... cat5/asp5/opi5/sent5
      - Bỏ slot được đánh dấu SKIP hoặc không có category
    Trả về:
      human_labels : dict[review_id → list[quadruple]]
      verdicts     : dict[review_id → str]  (đánh giá của annotator)
      warnings     : list[str]
    """
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    human_labels = {}
    verdicts     = {}
    warnings     = []

    for task in tasks:
        rid         = task["data"]["review_id"]
        annotations = task.get("annotations", [])
        if not annotations:
            warnings.append(f"  review {rid}: không có annotation → bỏ qua")
            continue

        ann   = annotations[-1]   # annotation mới nhất
        items = ann.get("result", [])

        slots   = defaultdict(dict)
        verdict = None

        for item in items:
            fname = item.get("from_name", "")
            val   = item.get("value", {})

            # ── Category: cat1 → cat5 ──────────────────────
            if re.match(r"^cat\d+$", fname):
                idx     = fname[3:]
                choices = val.get("choices", [])
                if choices and choices[0] not in SKIP_VALUES:
                    raw  = choices[0]
                    code = ALIAS_TO_CODE.get(raw)
                    if code:
                        slots[idx]["aspect_category"] = code
                    else:
                        warnings.append(
                            f"  review {rid}: category không nhận ra: {repr(raw)}"
                        )

            # ── Sentiment: sent1 → sent5 ───────────────────
            elif re.match(r"^sent\d+$", fname):
                idx     = fname[4:]
                choices = val.get("choices", [])
                if choices:
                    slots[idx]["sentiment"] = choices[0].strip().lower()

            # ── Aspect term: asp1 → asp5 ───────────────────
            elif re.match(r"^asp\d+$", fname):
                idx   = fname[3:]
                texts = val.get("text", [])
                if texts and texts[0].strip():
                    slots[idx]["aspect_term"] = texts[0].strip().lower()

            # ── Opinion term: opi1 → opi5 ──────────────────
            elif re.match(r"^opi\d+$", fname):
                idx   = fname[3:]
                texts = val.get("text", [])
                if texts and texts[0].strip():
                    slots[idx]["opinion_term"] = texts[0].strip().lower()

            # ── Đánh giá của annotator ─────────────────────
            elif fname == "llm_verdict":
                choices = val.get("choices", [])
                if choices:
                    verdict = choices[0]

        # Tổng hợp slots → quadruples
        quadruples = []
        for idx in sorted(slots.keys(), key=lambda x: int(x)):
            s = slots[idx]
            if "aspect_category" not in s:
                continue   # slot không có category → bỏ qua
            quadruples.append({
                "aspect_category": s["aspect_category"],
                "aspect_term"    : s.get("aspect_term",  "null"),
                "opinion_term"   : s.get("opinion_term", "null"),
                "sentiment"      : s.get("sentiment",    ""),
            })

        human_labels[rid] = quadruples
        verdicts[rid]     = verdict

    return human_labels, verdicts, warnings


# ================================================================
#  BƯỚC 2: PARSE LLM LABELS
# ================================================================
def parse_llm_labels(path: str):
    """
    Đọc file labelstudio_tasks.json.
    Parse chuỗi llm_display:
      "[1] CATEGORY | aspect=X | opinion=Y | sentiment"
    Trả về dict[review_id → list[quadruple]]
    """
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    llm_labels = {}
    warnings   = []

    PATTERN = re.compile(
        r"\[\d+\]\s+([A-Z#_]+)\s*\|\s*aspect=(.+?)\s*\|\s*opinion=(.+?)\s*\|\s*(\w+)",
        re.IGNORECASE,
    )

    for task in tasks:
        d       = task["data"]
        rid     = d["review_id"]
        display = d.get("llm_display", "")

        quadruples = []
        for line in display.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = PATTERN.match(line)
            if m:
                cat  = m.group(1).strip().upper()
                asp  = m.group(2).strip().lower()
                opi  = m.group(3).strip().lower()
                sent = m.group(4).strip().lower()
                quadruples.append({
                    "aspect_category": cat,
                    "aspect_term"    : asp,
                    "opinion_term"   : opi,
                    "sentiment"      : sent,
                })
            else:
                warnings.append(f"  review {rid}: không parse được dòng: {repr(line)}")

        llm_labels[rid] = quadruples

    return llm_labels, warnings


# ================================================================
#  BƯỚC 3: CHUẨN HÓA ĐỂ SO SÁNH
# ================================================================
def normalize_quad(q: dict) -> tuple:
    """
    Chuyển quadruple dict → tuple bất biến để so sánh set.
    Chiến lược: so sánh category + sentiment (2 trường ổn định nhất).
    Aspect/opinion term không dùng để khớp chính vì:
      - Human gõ tự do → khó khớp chính xác
      - LLM và human có thể dùng cách diễn đạt khác nhau cho cùng ý
    """
    return (
        q.get("aspect_category", "").upper().strip(),
        q.get("sentiment",       "").lower().strip(),
    )


def normalize_quad_full(q: dict) -> tuple:
    """Khớp đầy đủ 4 trường — dùng cho Exact Match nghiêm ngặt."""
    return (
        q.get("aspect_category", "").upper().strip(),
        q.get("aspect_term",     "").lower().strip(),
        q.get("opinion_term",    "").lower().strip(),
        q.get("sentiment",       "").lower().strip(),
    )


# ================================================================
#  BƯỚC 4: TÍNH CÁC METRIC
# ================================================================

def compute_quadruple_f1(human_labels: dict, llm_labels: dict, match="partial"):
    """
    Tính Precision, Recall, F1 theo quadruple.

    match="partial" : khớp category + sentiment (linh hoạt hơn)
    match="full"    : khớp cả 4 trường (nghiêm ngặt)
    """
    normalize = normalize_quad if match == "partial" else normalize_quad_full
    common_ids = sorted(set(human_labels) & set(llm_labels))

    tp = fp = fn = 0
    exact_match_count = 0
    per_review = []

    for rid in common_ids:
        h_set = set(normalize(q) for q in human_labels[rid])
        l_set = set(normalize(q) for q in llm_labels[rid])

        r_tp = len(h_set & l_set)
        r_fp = len(l_set - h_set)
        r_fn = len(h_set - l_set)

        tp += r_tp
        fp += r_fp
        fn += r_fn
        if h_set == l_set:
            exact_match_count += 1

        per_review.append({
            "review_id": rid,
            "exact_match": h_set == l_set,
            "human_quads": len(h_set),
            "llm_quads"  : len(l_set),
            "tp": r_tp, "fp": r_fp, "fn": r_fn,
        })

    n         = len(common_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "match_type"  : match,
        "n_reviews"   : n,
        "precision"   : round(precision, 4),
        "recall"      : round(recall,    4),
        "f1"          : round(f1,        4),
        "exact_match" : round(exact_match_count / n, 4) if n > 0 else 0.0,
        "per_review"  : per_review,
    }


def compute_per_category_metrics(human_labels: dict, llm_labels: dict):
    """
    F1 cho từng category (binary: review có category đó không?).
    Không phụ thuộc vào aspect/opinion term.
    """
    common_ids = sorted(set(human_labels) & set(llm_labels))
    results    = {}

    for cat in VALID_CATEGORIES:
        tp = fp = fn = tn = 0
        for rid in common_ids:
            h_cats = {q["aspect_category"] for q in human_labels[rid]}
            l_cats = {q["aspect_category"] for q in llm_labels[rid]}
            h_has  = cat in h_cats
            l_has  = cat in l_cats

            if h_has and l_has:     tp += 1
            elif h_has and not l_has: fn += 1
            elif not h_has and l_has: fp += 1
            else:                     tn += 1

        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[cat] = {
            "precision": round(p,  4),
            "recall"   : round(r,  4),
            "f1"       : round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "human_count": tp + fn,   # số review human có category này
            "llm_count"  : tp + fp,   # số review LLM có category này
        }
    return results


def compute_sentiment_metrics(human_labels: dict, llm_labels: dict):
    """
    Accuracy + F1 cho sentiment — chỉ xét quadruple có cùng category.
    """
    common_ids = sorted(set(human_labels) & set(llm_labels))
    h_sents    = []
    l_sents    = []

    for rid in common_ids:
        for hq in human_labels[rid]:
            cat = hq["aspect_category"]
            # Tìm quadruple LLM có cùng category
            match = next(
                (lq for lq in llm_labels[rid]
                 if lq["aspect_category"] == cat), None
            )
            if match and hq.get("sentiment") and match.get("sentiment"):
                h_sents.append(hq["sentiment"])
                l_sents.append(match["sentiment"])

    if not h_sents:
        return {"n_pairs": 0, "accuracy": 0.0}

    correct  = sum(h == l for h, l in zip(h_sents, l_sents))
    accuracy = correct / len(h_sents)

    # Per-sentiment breakdown
    per_sent = {}
    for s in VALID_SENTIMENTS:
        tp = sum(1 for h, l in zip(h_sents, l_sents) if h == s and l == s)
        fp = sum(1 for h, l in zip(h_sents, l_sents) if h != s and l == s)
        fn = sum(1 for h, l in zip(h_sents, l_sents) if h == s and l != s)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_sent[s] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    return {
        "n_pairs"    : len(h_sents),
        "accuracy"   : round(accuracy, 4),
        "per_sentiment": per_sent,
    }


def compute_cohen_kappa(human_labels: dict, llm_labels: dict):
    """
    Cohen's Kappa cho aspect_category.
    Chiến lược: binary vector per category per review.
    """
    if not HAS_SKLEARN:
        return None

    common_ids = sorted(set(human_labels) & set(llm_labels))
    kappa_per_cat = {}

    # Kappa từng category (binary: có/không)
    for cat in VALID_CATEGORIES:
        h_vec = []
        l_vec = []
        for rid in common_ids:
            h_cats = {q["aspect_category"] for q in human_labels[rid]}
            l_cats = {q["aspect_category"] for q in llm_labels[rid]}
            h_vec.append(1 if cat in h_cats else 0)
            l_vec.append(1 if cat in l_cats else 0)

        # Cần ít nhất 1 positive trong cả hai để kappa có nghĩa
        if sum(h_vec) == 0 or sum(l_vec) == 0:
            kappa_per_cat[cat] = None
            continue
        try:
            kappa_per_cat[cat] = round(cohen_kappa_score(h_vec, l_vec), 4)
        except Exception:
            kappa_per_cat[cat] = None

    # Kappa tổng (flatten tất cả)
    h_all, l_all = [], []
    for rid in common_ids:
        for cat in VALID_CATEGORIES:
            h_cats = {q["aspect_category"] for q in human_labels[rid]}
            l_cats = {q["aspect_category"] for q in llm_labels[rid]}
            h_all.append(cat if cat in h_cats else "NONE")
            l_all.append(cat if cat in l_cats else "NONE")

    try:
        overall = round(cohen_kappa_score(h_all, l_all), 4)
    except Exception:
        overall = None

    # Kappa cho sentiment (trên các cặp khớp category)
    h_sents, l_sents = [], []
    for rid in common_ids:
        for hq in human_labels[rid]:
            cat   = hq["aspect_category"]
            match = next(
                (lq for lq in llm_labels[rid] if lq["aspect_category"] == cat), None
            )
            if match and hq.get("sentiment") and match.get("sentiment"):
                h_sents.append(hq["sentiment"])
                l_sents.append(match["sentiment"])

    try:
        sent_kappa = round(cohen_kappa_score(h_sents, l_sents), 4) if len(h_sents) >= 5 else None
    except Exception:
        sent_kappa = None

    return {
        "overall_category_kappa" : overall,
        "sentiment_kappa"        : sent_kappa,
        "per_category_kappa"     : kappa_per_cat,
    }


def analyze_verdicts(verdicts: dict):
    """Thống kê nhận xét của annotator về LLM."""
    counter = Counter(v for v in verdicts.values() if v)
    total   = sum(counter.values())
    return {
        "total_with_verdict": total,
        "distribution": {k: {"count": v, "pct": round(v/total*100, 1)}
                         for k, v in counter.most_common()},
    }


# ================================================================
#  BƯỚC 5: IN VÀ LƯU BÁO CÁO
# ================================================================
def interpret_kappa(k):
    if k is None: return "N/A"
    if k >= 0.80: return "Rất tốt ✅"
    if k >= 0.60: return "Tốt 🟢"
    if k >= 0.40: return "Trung bình ⚠️"
    if k >= 0.20: return "Yếu 🟠"
    return "Rất yếu ❌"

def interpret_f1(f1):
    if f1 >= 0.80: return "LLM rất đáng tin ✅"
    if f1 >= 0.65: return "LLM đủ dùng ⚠️ — nên review category yếu"
    if f1 >= 0.50: return "LLM trung bình — cân nhắc re-prompt 🟠"
    return "LLM chất lượng thấp ❌ — cần re-prompt hoặc tăng manual label"


def build_report(
    partial_f1, full_f1,
    per_cat, sentiment_metrics,
    kappa_results, verdict_stats,
    human_warnings, llm_warnings,
):
    lines = []
    sep   = "─" * 60

    lines += [
        "=" * 60,
        "  ASQP ANNOTATION QUALITY REPORT",
        f"  TIKI Project — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
    ]

    # ── Tổng quan ──────────────────────────────────────────
    lines += [
        "",
        sep,
        "  1. CHỈ SỐ TỔNG QUAN",
        sep,
        f"  Số review đánh giá          : {partial_f1['n_reviews']}",
        "",
        "  [ Khớp linh hoạt: category + sentiment ]",
        f"  Precision  : {partial_f1['precision']*100:6.1f}%",
        f"  Recall     : {partial_f1['recall']*100:6.1f}%",
        f"  F1         : {partial_f1['f1']*100:6.1f}%   → {interpret_f1(partial_f1['f1'])}",
        f"  Exact Match: {partial_f1['exact_match']*100:6.1f}%",
        "",
        "  [ Khớp nghiêm ngặt: tất cả 4 trường ]",
        f"  F1         : {full_f1['f1']*100:6.1f}%",
        f"  Exact Match: {full_f1['exact_match']*100:6.1f}%",
    ]

    # ── Cohen's Kappa ───────────────────────────────────────
    lines += ["", sep, "  2. COHEN'S KAPPA", sep]
    if kappa_results:
        ok = kappa_results["overall_category_kappa"]
        sk = kappa_results["sentiment_kappa"]
        lines += [
            f"  Kappa (Aspect Category) : {ok}   → {interpret_kappa(ok)}",
            f"  Kappa (Sentiment)       : {sk}   → {interpret_kappa(sk)}",
            "",
            "  Thang đánh giá Kappa:",
            "    ≥ 0.80  Very Good  |  0.60–0.79  Good  |  0.40–0.59  Moderate  |  < 0.40  Poor",
        ]
    else:
        lines.append("  (sklearn chưa cài — bỏ qua Cohen's Kappa)")

    # ── Sentiment ───────────────────────────────────────────
    lines += ["", sep, "  3. SENTIMENT", sep]
    sm = sentiment_metrics
    lines += [
        f"  Accuracy (trên {sm['n_pairs']} cặp khớp category): {sm['accuracy']*100:.1f}%",
        "",
        f"  {'Sentiment':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}",
        "  " + "─" * 40,
    ]
    for s, m in sm.get("per_sentiment", {}).items():
        lines.append(
            f"  {s:<12} {m['precision']*100:>9.1f}% {m['recall']*100:>7.1f}% {m['f1']*100:>7.1f}%"
        )

    # ── Per-category ────────────────────────────────────────
    lines += ["", sep, "  4. F1 THEO TỪNG ASPECT CATEGORY", sep]
    lines.append(f"  {'Category':<28} {'F1':>6}  {'P':>6}  {'R':>6}  {'Human':>6}  {'LLM':>5}  Bar")
    lines.append("  " + "─" * 72)
    for cat, m in sorted(per_cat.items(), key=lambda x: -x[1]["f1"]):
        bar = "█" * int(m["f1"] * 20)
        flag = " ⚠️" if m["f1"] < 0.5 and m["human_count"] >= 5 else ""
        lines.append(
            f"  {cat:<28} {m['f1']*100:>5.1f}%  {m['precision']*100:>5.1f}%  "
            f"{m['recall']*100:>5.1f}%  {m['human_count']:>6}  {m['llm_count']:>5}  {bar}{flag}"
        )

    # Kappa per category
    if kappa_results:
        lines += ["", sep, "  5. COHEN'S KAPPA THEO TỪNG CATEGORY", sep]
        lines.append(f"  {'Category':<28} {'Kappa':>7}  {'Mức độ'}")
        lines.append("  " + "─" * 55)
        for cat, k in sorted(
            kappa_results["per_category_kappa"].items(),
            key=lambda x: -(x[1] or -99),
        ):
            kstr = f"{k:.4f}" if k is not None else " N/A  "
            lines.append(f"  {cat:<28} {kstr:>7}  {interpret_kappa(k)}")

    # ── Verdict của annotator ────────────────────────────────
    lines += ["", sep, "  6. NHẬN XÉT CỦA ANNOTATOR VỀ LLM", sep]
    for verdict, info in verdict_stats["distribution"].items():
        bar = "█" * int(info["pct"] / 3)
        lines.append(f"  {verdict:<45} {info['count']:>4} ({info['pct']:>5.1f}%)  {bar}")

    # ── Cảnh báo ────────────────────────────────────────────
    all_warnings = human_warnings + llm_warnings
    if all_warnings:
        lines += ["", sep, f"  7. CẢNH BÁO PARSE ({len(all_warnings)} mục)", sep]
        for w in all_warnings[:20]:
            lines.append(w)
        if len(all_warnings) > 20:
            lines.append(f"  ... và {len(all_warnings)-20} cảnh báo nữa (xem file JSON)")

    # ── Kết luận ────────────────────────────────────────────
    f1 = partial_f1["f1"]
    ok = kappa_results["overall_category_kappa"] if kappa_results else None
    lines += [
        "", "=" * 60, "  KẾT LUẬN", "=" * 60,
        f"  Quadruple F1 (linh hoạt) : {f1*100:.1f}%",
        f"  Cohen's Kappa (category) : {ok if ok else 'N/A'}",
        "",
    ]
    if f1 >= 0.75:
        lines.append("  ✅ LLM annotation đủ tin cậy để dùng làm dữ liệu train.")
    elif f1 >= 0.55:
        lines.append("  ⚠️  LLM annotation chấp nhận được — nên xem lại category có F1 thấp.")
        low_cats = [c for c, m in per_cat.items() if m["f1"] < 0.5 and m["human_count"] >= 5]
        if low_cats:
            lines.append(f"  Các category cần re-prompt: {', '.join(low_cats)}")
    else:
        lines.append("  ❌ LLM annotation chất lượng thấp — cần re-prompt hoặc tăng manual label.")
    lines.append("=" * 60)

    return "\n".join(lines)


# ================================================================
#  MAIN
# ================================================================
def main():
    print("=" * 60)
    print("  ĐÁNH GIÁ CHẤT LƯỢNG ANNOTATION: HUMAN vs LLM")
    print("=" * 60)

    # Kiểm tra file
    for path in [HUMAN_FILE, LLM_FILE]:
        if not os.path.exists(path):
            print(f"\n[LỖI] Không tìm thấy: {path}")
            print("  Hãy kiểm tra lại đường dẫn CẤU HÌNH đầu file.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Parse ─────────────────────────────────────────────────
    print("\n[1/5] Parse human labels...")
    human_labels, verdicts, h_warns = parse_human_labels(HUMAN_FILE)
    print(f"  → {len(human_labels)} reviews | {len(h_warns)} cảnh báo")

    print("\n[2/5] Parse LLM labels...")
    llm_labels, l_warns = parse_llm_labels(LLM_FILE)
    print(f"  → {len(llm_labels)} reviews | {len(l_warns)} cảnh báo")

    common = set(human_labels) & set(llm_labels)
    print(f"\n  Review IDs khớp: {len(common)}/{len(human_labels)}")

    # ── Tính metrics ──────────────────────────────────────────
    print("\n[3/5] Tính chỉ số...")
    partial_f1  = compute_quadruple_f1(human_labels, llm_labels, match="partial")
    full_f1     = compute_quadruple_f1(human_labels, llm_labels, match="full")
    per_cat     = compute_per_category_metrics(human_labels, llm_labels)
    sent_m      = compute_sentiment_metrics(human_labels, llm_labels)
    kappa       = compute_cohen_kappa(human_labels, llm_labels)
    verdict_stats = analyze_verdicts(verdicts)

    # ── Báo cáo text ─────────────────────────────────────────
    print("\n[4/5] Tạo báo cáo...")
    report_text = build_report(
        partial_f1, full_f1, per_cat, sent_m,
        kappa, verdict_stats, h_warns, l_warns,
    )
    print("\n" + report_text)

    txt_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── Báo cáo JSON ─────────────────────────────────────────
    print("\n[5/5] Lưu file...")
    json_out = {
        "generated_at"    : datetime.now().isoformat(),
        "n_reviews"       : len(common),
        "partial_f1"      : partial_f1,
        "full_f1"         : full_f1,
        "per_category_f1" : per_cat,
        "sentiment"       : sent_m,
        "cohen_kappa"     : kappa,
        "annotator_verdicts": verdict_stats,
        "parse_warnings"  : h_warns + l_warns,
    }
    json_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(f"\n  📄 Báo cáo text : {txt_path}")
    print(f"  📊 Báo cáo JSON : {json_path}")
    print("\n✅ XONG!")


if __name__ == "__main__":
    main()