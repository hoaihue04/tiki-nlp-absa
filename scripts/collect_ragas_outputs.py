from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect chatbot outputs for RAGAS evaluation.")
    parser.add_argument("--input", default="data/eval/ragas_chat_eval.jsonl")
    parser.add_argument("--output", default="results/ragas_chat_outputs.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples to collect.")
    parser.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-products", action="store_true", help="Run /api/analyze once per product before chat.")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    samples = _load_jsonl(Path(args.input))
    if args.limit > 0:
        samples = samples[: args.limit]

    base_url = args.base_url.rstrip("/")
    if args.prepare_products:
        _prepare_products(base_url, samples, timeout=args.timeout)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(samples, start=1):
            payload = {
                "product_id": sample["product_id"],
                "question": sample["question"],
                "use_llm": args.use_llm,
            }
            result: dict[str, Any]
            try:
                response = requests.post(
                    f"{base_url}/api/assistant/chat",
                    json=payload,
                    timeout=args.timeout,
                )
                chat = _response_json(response)
                if response.ok:
                    result = _build_output_row(sample, chat, ok=True)
                else:
                    result = _build_output_row(
                        sample,
                        {
                            "error": _error_message(response, chat),
                            "status_code": response.status_code,
                            "response_body": chat,
                        },
                        ok=False,
                    )
            except Exception as exc:
                result = _build_output_row(sample, {"error": str(exc)}, ok=False)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{index}/{len(samples)}] {'ok' if result['ok'] else 'error'} product={sample['product_id']} question={sample['question'][:70]}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote {written} rows to {out_path}")


def _prepare_products(base_url: str, samples: list[dict[str, Any]], timeout: int) -> None:
    seen: set[str] = set()
    product_ids = [str(row["product_id"]) for row in samples if str(row.get("product_id") or "")]
    for index, product_id in enumerate(product_ids, start=1):
        if product_id in seen:
            continue
        seen.add(product_id)
        try:
            response = requests.post(
                f"{base_url}/api/analyze",
                json={"product_id": product_id},
                timeout=timeout,
            )
            body = _response_json(response)
            if response.ok:
                print(f"[prepare {index}/{len(product_ids)}] ok product={product_id}")
            else:
                print(f"[prepare {index}/{len(product_ids)}] error product={product_id}: {_error_message(response, body)}")
        except Exception as exc:
            print(f"[prepare {index}/{len(product_ids)}] error product={product_id}: {exc}")


def _build_output_row(sample: dict[str, Any], chat: dict[str, Any], ok: bool) -> dict[str, Any]:
    evidence = chat.get("evidence") or []
    return {
        "ok": ok,
        "product_id": sample.get("product_id", ""),
        "product_name": sample.get("product_name", ""),
        "question": sample.get("question", ""),
        "reference": sample.get("reference", ""),
        "reference_contexts": sample.get("reference_contexts", []),
        "reference_context_ids": [str(value) for value in sample.get("reference_context_ids", [])],
        "expected_intent": sample.get("expected_intent", ""),
        "expected_aspects": sample.get("expected_aspects", []),
        "expected_metric": sample.get("expected_metric", ""),
        "expected_sentiment": sample.get("expected_sentiment", ""),
        "eval_tags": sample.get("eval_tags", []),
        "answer": chat.get("answer", ""),
        "retrieved_contexts": [str(item.get("text") or "") for item in evidence if item.get("text")],
        "retrieved_context_ids": [str(item.get("review_id") or "") for item in evidence if item.get("text") and item.get("review_id")],
        "evidence": evidence,
        "query_analysis": chat.get("query_analysis") or {},
        "statistics_plan": chat.get("statistics_plan") or {},
        "statistics_result": chat.get("statistics_result") or {},
        "retrieval_strategy": chat.get("retrieval_strategy", ""),
        "vector_used": bool(chat.get("vector_used")),
        "cache_hit": bool(chat.get("cache_hit")),
        "llm_used": bool(chat.get("llm_used")),
        "llm_model": chat.get("llm_model", ""),
        "error": chat.get("error", ""),
        "status_code": chat.get("status_code"),
        "response_body": chat.get("response_body"),
    }


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
        return body if isinstance(body, dict) else {"body": body}
    except Exception:
        return {"text": response.text}


def _error_message(response: requests.Response, body: dict[str, Any]) -> str:
    detail = body.get("detail")
    if detail:
        return str(detail)
    text = body.get("text")
    if text:
        return str(text)
    return f"{response.status_code} {response.reason}"


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
