from app.assistant.assistant_service import _format_structured_answer, _prepend_statistics_summary


def test_format_structured_answer_prefers_natural_summary() -> None:
    answer = _format_structured_answer(
        {
            "summary": "San pham co the dung duoc cho be da nhay cam neu uu tien chat lieu mem va theo doi lan dau su dung.",
            "pros": ["Chat lieu mem"],
            "cons": ["Can kiem tra kich thuoc"],
            "recommendation": "Nen theo doi phan ung cua be trong vai lan dau.",
            "confidence": "medium",
            "confidence_reasons": ["Co du evidence"],
            "citations": [],
        },
        intent="ASPECT_QA",
    )

    assert "Uu diem:" not in answer
    assert "Khuyen nghi:" not in answer
    assert "Do tin cay:" not in answer
    assert answer.startswith("San pham co the dung duoc")


def test_prepend_statistics_summary_adds_clear_count() -> None:
    answer = _prepend_statistics_summary(
        "Chua thay phan hoi tieu cuc noi bat ve do thoai mai.",
        {
            "metric": "top_aspects_by_sentiment",
            "plan": {"sentiment": "positive"},
            "rows": [
                {
                    "aspect": "PRODUCT#COMFORT",
                    "sentiment": "positive",
                    "total": 7,
                }
            ],
        },
    )

    assert answer.startswith("Có 7")
    assert "đánh giá tích cực" in answer
