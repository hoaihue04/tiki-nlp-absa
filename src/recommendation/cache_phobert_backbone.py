#!/usr/bin/env python3
"""
Download and save PhoBERT backbone locally for offline inference.
"""

from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def main() -> None:
    model_name = "vinai/phobert-base-v2"
    out_dir = Path("models/phobert/base_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"Saved backbone to: {out_dir}")


if __name__ == "__main__":
    main()

