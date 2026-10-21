from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.serialization import make_json_serializable


def test_make_json_serializable_handles_numpy_and_pandas() -> None:
    payload = {
        "integer": np.int64(7),
        "floating": np.float32(1.25),
        "array": np.array([1, 2, 3]),
        "frame": pd.DataFrame([{"name": "A", "score": np.int32(5)}]),
        "series": pd.Series({"ok": True}),
    }

    result = make_json_serializable(payload)

    assert result["integer"] == 7
    assert result["floating"] == 1.25
    assert result["array"] == [1, 2, 3]
    assert result["frame"] == [{"name": "A", "score": 5}]
    assert result["series"] == {"ok": True}
