from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TikiInsight RAG outputs with RAGAS using an OpenAI-compatible evaluator.")
    parser.add_argument("--input", default="results/ragas_chat_outputs.jsonl")
    parser.add_argument("--report-csv", default="results/ragas_openai_report.csv")
    parser.add_argument("--summary-json", default="results/ragas_openai_summary.json")
    parser.add_argument("--model", default=os.getenv("RAGAS_EVAL_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.getenv("RAGAS_EVAL_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("RAGAS_EVAL_MAX_TOKENS", "8192")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("RAGAS_EVAL_TIMEOUT", "180")))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("RAGAS_MAX_WORKERS", "8")))
    parser.add_argument("--answer-relevancy-strictness", type=int, default=int(os.getenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "3")))
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Exclude samples tagged 'statistics' so SQL questions are not mixed into semantic RAG evaluation.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate only context precision/recall; ignore generated answers and generation metrics.",
    )
    parser.add_argument("--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-ragas", action="store_true", help="Only compute custom metrics, no LLM judge calls.")
    args = parser.parse_args()

    rows = [row for row in _load_jsonl(Path(args.input)) if row.get("ok")]
    if args.semantic_only:
        rows = [row for row in rows if "statistics" not in set(row.get("eval_tags") or [])]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No successful rows found. Run scripts/collect_ragas_outputs.py first.")

    custom_report = [_custom_metrics(row) for row in rows]
    custom_df = pd.DataFrame(custom_report)

    ragas_df = pd.DataFrame()
    model = str(args.model or "gpt-4.1-mini").strip()

    if not args.skip_ragas:
        evaluator_llm = _build_evaluator_llm(model, args.base_url, args.max_tokens, args.timeout)
        evaluator_embeddings = _build_evaluator_embeddings(args.embedding_model)
        dataset = _build_ragas_dataset(rows)
        ragas_result = _run_ragas(
            dataset,
            evaluator_llm,
            evaluator_embeddings,
            timeout=args.timeout,
            max_workers=args.max_workers,
            answer_relevancy_strictness=args.answer_relevancy_strictness,
            include_generation_metrics=not args.retrieval_only,
        )
        ragas_df = ragas_result.to_pandas()

    report = pd.concat([_base_report(rows), custom_df, ragas_df], axis=1)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_csv, index=False, encoding="utf-8-sig")

    summary = _summarize(report)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote report to {args.report_csv}")
    print(f"Wrote summary to {args.summary_json}")


def _build_evaluator_llm(model: str, base_url: str, max_tokens: int, timeout: int) -> Any:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("RAGAS_EVAL_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    _install_ragas_langchain_compat()
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except Exception as exc:
        raise SystemExit(
            "Could not initialize evaluator. Install compatible packages:\n"
            "python -m pip install ragas datasets openai langchain-openai"
        ) from exc

    try:
        return LangchainLLMWrapper(
            ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0,
                max_completion_tokens=max_tokens,
                timeout=timeout,
                max_retries=0,
            )
        )
    except Exception as exc:
        raise SystemExit(f"Could not initialize evaluator: {exc}") from exc


def _build_evaluator_embeddings(model_name: str) -> Any:
    _install_ragas_langchain_compat()
    try:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise SystemExit(
            "Could not initialize RAGAS embeddings. Install sentence-transformers and ragas:\n"
            "python -m pip install sentence-transformers ragas"
        ) from exc

    class SentenceTransformerEmbeddings:
        def __init__(self, model: str) -> None:
            self.model = model
            self._encoder = SentenceTransformer(model)

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vectors = self._encoder.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.astype(float).tolist()

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    return LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(model_name))


def _run_ragas(
    dataset: Dataset,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    timeout: int,
    max_workers: int,
    answer_relevancy_strictness: int,
    include_generation_metrics: bool,
) -> Any:
    _install_ragas_langchain_compat()
    try:
        from ragas import evaluate
        from ragas.metrics import (
            IDBasedContextPrecision,
            IDBasedContextRecall,
            answer_relevancy,
            context_precision as llm_context_precision,
            context_recall as llm_context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig
    except Exception as exc:
        raise SystemExit(
            "RAGAS is not installed or the installed version is incompatible. Try:\n"
            "python -m pip install ragas datasets"
        ) from exc

    run_config = RunConfig(
        timeout=max(1, timeout),
        max_retries=1,
        max_workers=max(1, max_workers),
    )

    if include_generation_metrics:
        metrics = [llm_context_precision, llm_context_recall]
        # OpenAI supports multiple reverse questions for answer_relevancy.
        # Local OpenAI-compatible servers can use strictness=1.
        answer_relevancy.strictness = max(1, answer_relevancy_strictness)
        metrics.extend([faithfulness, answer_relevancy])
    else:
        # ID-based metrics compare the exact review IDs retrieved by the
        # system with the manually selected gold review IDs.
        context_precision = IDBasedContextPrecision()
        context_recall = IDBasedContextRecall()
        context_precision.name = "context_precision"
        context_recall.name = "context_recall"
        metrics = [context_precision, context_recall]

    return evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )


def _install_ragas_langchain_compat() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility shim, not used for DeepSeek
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def _build_ragas_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list([
        {
            "user_input": row.get("question", ""),
            "response": row.get("answer", ""),
            "retrieved_contexts": row.get("retrieved_contexts") or [],
            "reference": row.get("reference", ""),
            "reference_contexts": row.get("reference_contexts") or [],
            "retrieved_context_ids": row.get("retrieved_context_ids") or [],
            "reference_context_ids": row.get("reference_context_ids") or [],
        }
        for row in rows
    ])


def _base_report(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "product_id": row.get("product_id", ""),
            "question": row.get("question", ""),
            "expected_intent": row.get("expected_intent", ""),
            "predicted_intent": (row.get("query_analysis") or {}).get("intent", ""),
            "expected_metric": row.get("expected_metric", ""),
            "predicted_metric": (row.get("statistics_plan") or {}).get("metric", ""),
            "expected_sentiment": row.get("expected_sentiment", ""),
            "predicted_sentiment": (row.get("statistics_plan") or {}).get("sentiment", ""),
            "answer": row.get("answer", ""),
            "retrieved_context_count": len(row.get("retrieved_contexts") or []),
            "eval_tags": ",".join(row.get("eval_tags") or []),
        }
        for row in rows
    ])


def _custom_metrics(row: dict[str, Any]) -> dict[str, Any]:
    expected_aspects = set(row.get("expected_aspects") or [])
    predicted_aspects = set((row.get("query_analysis") or {}).get("target_aspects") or [])

    intersection = expected_aspects & predicted_aspects
    aspect_precision = len(intersection) / len(predicted_aspects) if predicted_aspects else float(not expected_aspects)
    aspect_recall = len(intersection) / len(expected_aspects) if expected_aspects else 1.0
    aspect_f1 = (
        2 * aspect_precision * aspect_recall / (aspect_precision + aspect_recall)
        if aspect_precision + aspect_recall
        else 0.0
    )

    expected_metric = row.get("expected_metric") or ""
    predicted_metric = (row.get("statistics_plan") or {}).get("metric") or ""
    expected_sentiment = row.get("expected_sentiment") or ""
    predicted_sentiment = (row.get("statistics_plan") or {}).get("sentiment") or ""

    return {
        "intent_ok": int(row.get("expected_intent") == (row.get("query_analysis") or {}).get("intent")),
        "aspect_precision": round(aspect_precision, 4),
        "aspect_recall": round(aspect_recall, 4),
        "aspect_f1": round(aspect_f1, 4),
        "metric_ok": int(not expected_metric or expected_metric == predicted_metric),
        "sentiment_ok": int(not expected_sentiment or expected_sentiment == predicted_sentiment),
        "numeric_ok": int(_numeric_answer_matches_statistics(row)),
    }


def _numeric_answer_matches_statistics(row: dict[str, Any]) -> bool:
    tags = set(row.get("eval_tags") or [])
    if "statistics" not in tags:
        return True

    answer = str(row.get("answer") or "")
    numbers = {int(value) for value in re.findall(r"\d+", answer)}
    if not numbers:
        return False

    statistics_result = row.get("statistics_result") or {}
    for item in statistics_result.get("rows") or []:
        total = item.get("total")
        if total is not None and int(total) in numbers:
            return True
    data = statistics_result.get("data") or {}
    for key in ("stored_review_count", "aspect_record_count"):
        value = data.get(key)
        if value is not None and int(value) in numbers:
            return True
    return False


def _summarize(report: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = [
        column
        for column in report.columns
        if pd.api.types.is_numeric_dtype(report[column])
    ]
    summary: dict[str, Any] = {
        "rows": int(len(report)),
    }
    for column in numeric_cols:
        value = float(report[column].mean())
        summary[column] = None if pd.isna(value) else round(value, 4)

    summary["retrieval_metrics"] = _metric_means(
        report,
        ["context_precision", "context_recall"],
    )
    summary["generation_metrics"] = _metric_means(
        report,
        ["faithfulness", "answer_relevancy"],
    )
    return summary


def _metric_means(report: pd.DataFrame, columns: list[str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for column in columns:
        if column not in report:
            values[column] = None
            continue
        value = pd.to_numeric(report[column], errors="coerce").mean()
        values[column] = None if pd.isna(value) else round(float(value), 4)
    return values


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
