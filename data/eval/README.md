# RAGAS Evaluation Dataset

This folder contains evaluation samples generated from local TikiInsight data.

Main file:

```text
ragas_chat_eval.jsonl
```

Each JSONL row contains:

- `product_id`: product used when calling `/api/assistant/chat`.
- `product_name`: product name for manual inspection.
- `question`: user question to send to the chatbot.
- `reference`: expected answer or answer requirements.
- `reference_contexts`: manually derived review evidence from annotated ASQP data.
- `expected_intent`: expected query intent for custom query-understanding evaluation.
- `expected_aspects`: expected ABSA aspect labels.
- `expected_metric`: expected statistics metric, if applicable.
- `expected_sentiment`: expected sentiment filter, if applicable.
- `eval_tags`: sample category tags.

Suggested mapping for RAGAS:

```python
user_input = row["question"]
reference = row["reference"]
reference_contexts = row["reference_contexts"]
response = api_response["answer"]
retrieved_contexts = [item["text"] for item in api_response["evidence"]]
```

Use RAGAS for retrieval and answer quality metrics, and use the `expected_*`
fields for custom classification/query-plan metrics.
