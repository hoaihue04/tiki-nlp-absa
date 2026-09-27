from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ASPECT_LABELS = {
    "PRODUCT#SAFETY": "độ an toàn",
    "PRODUCT#MATERIAL": "chất liệu",
    "PRODUCT#SCENT": "mùi sản phẩm",
    "PRODUCT#SIZE": "kích thước",
    "PRODUCT#ABSORPTION": "khả năng thấm hút",
    "PRODUCT#FUNCTION": "công năng sử dụng",
    "PRODUCT#QUALITY": "chất lượng",
    "PRODUCT#VALUE": "độ đáng tiền",
    "PRODUCT#DURABILITY": "độ bền",
    "PRODUCT#COMFORT": "độ thoải mái",
    "PRICE#AFFORDABILITY": "giá cả",
    "DELIVERY#SPEED": "giao hàng",
    "DELIVERY#PACKAGING": "bao bì",
    "DELIVERY#ACCURACY": "giao đúng/đủ hàng",
    "SELLER#AUTHENTICITY": "tính chính hãng",
}

SENTIMENT_LABELS = {
    "positive": "tích cực",
    "negative": "tiêu cực",
    "neutral": "trung lập",
}

BABY_FIT_ASPECTS = {
    "PRODUCT#SAFETY",
    "PRODUCT#MATERIAL",
    "PRODUCT#SCENT",
    "PRODUCT#COMFORT",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a RAGAS evaluation dataset for TikiInsight.")
    parser.add_argument("--asqp", default="data/processed/asqp_annotated_flat.csv")
    parser.add_argument("--reviews", default="data/interim/merged_reviews_products.csv")
    parser.add_argument("--output", default="data/eval/ragas_chat_eval.jsonl")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    dataset = build_dataset(
        asqp_path=Path(args.asqp),
        reviews_path=Path(args.reviews),
        limit=args.limit,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(dataset)} samples to {out_path}")


def build_dataset(asqp_path: Path, reviews_path: Path, limit: int) -> list[dict[str, Any]]:
    asqp = pd.read_csv(asqp_path, dtype=str).fillna("")
    reviews = pd.read_csv(
        reviews_path,
        dtype=str,
        usecols=["review_id", "product_id", "name", "rating", "cleaned_content"],
    ).fillna("")
    df = asqp.merge(reviews, on="review_id", how="inner")
    df["text"] = df["text_preview"].where(df["text_preview"].str.strip() != "", df["cleaned_content"])
    df["text"] = df["text"].map(_clean_text)
    df = df[
        df["product_id"].str.strip().ne("")
        & df["aspect_category"].isin(ASPECT_LABELS)
        & df["sentiment"].isin(SENTIMENT_LABELS)
        & df["text"].str.len().ge(18)
    ].copy()

    group_rows = _build_group_rows(df)
    samples: list[dict[str, Any]] = []
    per_product: dict[str, int] = defaultdict(int)
    count_quota = max(8, int(limit * 0.45))
    aspect_quota = max(6, int(limit * 0.25))
    product_quota = max(6, limit - count_quota - aspect_quota)

    for group in group_rows:
        if len([sample for sample in samples if "aspect_count" in sample["eval_tags"]]) >= count_quota:
            break
        product_id = group["product_id"]
        if per_product[product_id] >= 3:
            continue
        samples.append(_count_sample(group))
        per_product[product_id] += 1

    for group in group_rows:
        if len([sample for sample in samples if "aspect_qa" in sample["eval_tags"]]) >= aspect_quota:
            break
        product_id = group["product_id"]
        if per_product[product_id] >= 5:
            continue
        samples.append(_aspect_qa_sample(group))
        per_product[product_id] += 1

    product_added = 0
    for sample in _product_level_samples(df):
        if product_added >= product_quota:
            break
        product_id = sample["product_id"]
        if per_product[product_id] >= 7:
            continue
        samples.append(sample)
        per_product[product_id] += 1
        product_added += 1

    for group in group_rows:
        if len(samples) >= limit:
            break
        product_id = group["product_id"]
        if per_product[product_id] >= 8:
            continue
        sample = _count_sample(group)
        sample_key = _sample_key(sample)
        if sample_key in {_sample_key(existing) for existing in samples}:
            continue
        samples.append(sample)
        per_product[product_id] += 1

    return samples[:limit]


def _build_group_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = (
        df.groupby(["product_id", "name", "aspect_category", "sentiment"], dropna=False)
        .agg(total=("review_id", "count"))
        .reset_index()
        .sort_values(["total", "product_id"], ascending=[False, True])
    )
    for _, group in grouped.iterrows():
        total = int(group["total"])
        if total < 3:
            continue
        product_id = str(group["product_id"])
        aspect = str(group["aspect_category"])
        sentiment = str(group["sentiment"])
        contexts = _contexts_for(df, product_id, aspect, sentiment)
        if not contexts:
            continue
        context_ids = _context_ids_for(df, product_id, aspect, sentiment)
        rows.append({
            "product_id": product_id,
            "product_name": str(group["name"]),
            "aspect": aspect,
            "sentiment": sentiment,
            "total": total,
            "contexts": contexts,
            "context_ids": context_ids,
        })
    return rows


def _product_level_samples(df: pd.DataFrame) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    product_counts = df.groupby(["product_id", "name"]).size().sort_values(ascending=False).head(20)
    for (product_id, name), _ in product_counts.items():
        product_df = df[df["product_id"] == product_id]
        negative = (
            product_df[product_df["sentiment"] == "negative"]
            .groupby("aspect_category")
            .size()
            .sort_values(ascending=False)
        )
        positive = (
            product_df[product_df["sentiment"] == "positive"]
            .groupby("aspect_category")
            .size()
            .sort_values(ascending=False)
        )
        if not negative.empty:
            aspect = str(negative.index[0])
            total = int(negative.iloc[0])
            contexts = _contexts_for(product_df, product_id, aspect, "negative")
            context_ids = _context_ids_for(product_df, product_id, aspect, "negative")
            samples.append({
                "product_id": str(product_id),
                "product_name": str(name),
                "question": "Người mua phàn nàn nhiều nhất về điều gì?",
                "reference": (
                    f"Nhóm phàn nàn nổi bật nhất là {ASPECT_LABELS[aspect]} với {total} lượt ý kiến tiêu cực "
                    "trong dữ liệu đã gán nhãn. Câu trả lời cần nêu khía cạnh này và diễn giải dựa trên review."
                ),
                "reference_contexts": contexts,
                "reference_context_ids": context_ids,
                "expected_intent": "COMPLAINT_SUMMARY",
                "expected_aspects": [aspect],
                "expected_metric": "top_aspects_by_sentiment",
                "expected_sentiment": "negative",
                "eval_tags": ["complaint_summary", "statistics", "retrieval"],
            })
        if not positive.empty:
            aspect = str(positive.index[0])
            total = int(positive.iloc[0])
            contexts = _contexts_for(product_df, product_id, aspect, "positive")
            context_ids = _context_ids_for(product_df, product_id, aspect, "positive")
            samples.append({
                "product_id": str(product_id),
                "product_name": str(name),
                "question": "Điểm nào được người mua khen nhiều nhất?",
                "reference": (
                    f"Điểm được khen nổi bật nhất là {ASPECT_LABELS[aspect]} với {total} lượt ý kiến tích cực "
                    "trong dữ liệu đã gán nhãn. Câu trả lời cần nêu số liệu và evidence tiêu biểu."
                ),
                "reference_contexts": contexts,
                "reference_context_ids": context_ids,
                "expected_intent": "COMPLAINT_SUMMARY",
                "expected_aspects": [aspect],
                "expected_metric": "top_aspects_by_sentiment",
                "expected_sentiment": "positive",
                "eval_tags": ["praise_summary", "statistics", "retrieval"],
            })
        fit_contexts = []
        fit_context_ids = []
        fit_aspects = []
        for aspect in BABY_FIT_ASPECTS:
            contexts = _contexts_for(product_df, product_id, aspect, "positive", limit=1)
            context_ids = _context_ids_for(product_df, product_id, aspect, "positive", limit=1)
            if contexts:
                fit_contexts.extend(contexts)
                fit_context_ids.extend(context_ids)
                fit_aspects.append(aspect)
        if len(fit_contexts) >= 2:
            samples.append({
                "product_id": str(product_id),
                "product_name": str(name),
                "question": "Sản phẩm này có phù hợp cho bé da nhạy cảm không?",
                "reference": (
                    "Câu trả lời cần đánh giá mức độ phù hợp cho bé da nhạy cảm dựa trên các evidence về "
                    "độ an toàn, chất liệu, mùi sản phẩm hoặc độ thoải mái; không được suy diễn ngoài review."
                ),
                "reference_contexts": fit_contexts[:3],
                "reference_context_ids": fit_context_ids[:3],
                "expected_intent": "PRODUCT_FIT",
                "expected_aspects": fit_aspects,
                "expected_metric": "",
                "expected_sentiment": "",
                "eval_tags": ["product_fit", "retrieval", "faithfulness"],
            })
    return samples


def _count_sample(group: dict[str, Any]) -> dict[str, Any]:
    aspect_label = ASPECT_LABELS[group["aspect"]]
    sentiment_label = SENTIMENT_LABELS[group["sentiment"]]
    return {
        "product_id": group["product_id"],
        "product_name": group["product_name"],
        "question": f"Có bao nhiêu đánh giá {sentiment_label} về {aspect_label}?",
        "reference": (
            f"Có {group['total']} lượt ý kiến {sentiment_label} về {aspect_label} trong dữ liệu đã gán nhãn. "
            "Câu trả lời cần nêu số lượng này và diễn giải ngắn dựa trên evidence."
        ),
        "reference_contexts": group["contexts"],
        "reference_context_ids": group["context_ids"],
        "expected_intent": "ASPECT_QA",
        "expected_aspects": [group["aspect"]],
        "expected_metric": "top_aspects_by_sentiment",
        "expected_sentiment": group["sentiment"],
        "eval_tags": ["aspect_count", "statistics", "retrieval"],
    }


def _aspect_qa_sample(group: dict[str, Any]) -> dict[str, Any]:
    aspect_label = ASPECT_LABELS[group["aspect"]]
    sentiment_label = SENTIMENT_LABELS[group["sentiment"]]
    question = f"Người mua nói gì về {aspect_label}?"
    if group["sentiment"] == "negative":
        question = f"{aspect_label.capitalize()} có vấn đề gì không?"
    elif group["sentiment"] == "positive":
        question = f"{aspect_label.capitalize()} có được đánh giá tốt không?"
    return {
        "product_id": group["product_id"],
        "product_name": group["product_name"],
        "question": question,
        "reference": (
            f"Câu trả lời cần tập trung vào {aspect_label}. Dữ liệu có {group['total']} lượt ý kiến "
            f"{sentiment_label} liên quan đến khía cạnh này và cần dùng evidence trong reference_contexts."
        ),
        "reference_contexts": group["contexts"],
        "reference_context_ids": group["context_ids"],
        "expected_intent": "ASPECT_QA",
        "expected_aspects": [group["aspect"]],
        "expected_metric": "top_aspects_by_sentiment",
        "expected_sentiment": group["sentiment"],
        "eval_tags": ["aspect_qa", "retrieval", "faithfulness"],
    }


def _contexts_for(
    df: pd.DataFrame,
    product_id: str,
    aspect: str,
    sentiment: str,
    limit: int = 3,
) -> list[str]:
    return [row["text"] for row in _context_records_for(df, product_id, aspect, sentiment, limit)]


def _context_ids_for(
    df: pd.DataFrame,
    product_id: str,
    aspect: str,
    sentiment: str,
    limit: int = 3,
) -> list[str]:
    return [row["review_id"] for row in _context_records_for(df, product_id, aspect, sentiment, limit)]


def _context_records_for(
    df: pd.DataFrame,
    product_id: str,
    aspect: str,
    sentiment: str,
    limit: int = 3,
) -> list[dict[str, str]]:
    rows = df[
        (df["product_id"].astype(str) == str(product_id))
        & (df["aspect_category"] == aspect)
        & (df["sentiment"] == sentiment)
    ].copy()
    if rows.empty:
        return []
    rows["_len"] = rows["text"].str.len()
    rows = rows.sort_values("_len", ascending=False)
    contexts: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in rows.iterrows():
        text = row["text"]
        clean = _clean_text(text)
        key = clean.lower()
        if clean and key not in seen:
            contexts.append({
                "review_id": str(row["review_id"]),
                "text": clean,
            })
            seen.add(key)
        if len(contexts) >= limit:
            break
    return contexts


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _sample_key(sample: dict[str, Any]) -> str:
    return "|".join([
        str(sample.get("product_id") or ""),
        str(sample.get("question") or ""),
    ])


if __name__ == "__main__":
    main()
