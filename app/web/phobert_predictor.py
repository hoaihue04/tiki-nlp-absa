from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.web.constants import (
    ASPECT_FORBIDDEN_KEYWORDS,
    ASPECT_REQUIRED_KEYWORDS,
    CATEGORIES,
    SENTIMENT_NAMES,
    SENTIMENT_TO_SCORE,
)

_ROOT_DIR = Path(__file__).resolve().parents[2]

class AppPhoBERTASQP(nn.Module):
    def __init__(self, backbone_config_dir: Path, last_n_layers: int = 4,
                 proj_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        bert_config = AutoConfig.from_pretrained(str(backbone_config_dir))
        bert_config.output_hidden_states = True
        self.bert = AutoModel.from_config(bert_config)
        self.last_n_layers = last_n_layers
        concat_dim = int(bert_config.hidden_size) * last_n_layers

        self.projection = nn.Sequential(
            nn.Linear(concat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, len(SENTIMENT_NAMES)),
            )
            for _ in range(len(CATEGORIES))
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_n = outputs.hidden_states[-self.last_n_layers:]
        cls_vectors = [hidden[:, 0, :] for hidden in last_n]
        projected = self.projection(torch.cat(cls_vectors, dim=-1))
        return [classifier(projected) for classifier in self.classifiers]


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class RealPhoBERTPredictor:
    def __init__(self, max_len: int = 256, confidence_threshold: float = 0.6):
        self.max_len = max_len
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self.model_path: Optional[Path] = None
        self.load_error: Optional[str] = None
        self.cache: Dict[str, List[Dict]] = {}

    def _load_model(self):
        if self._loaded:
            return
        print("[PhoBERT] Loading tokenizer and model...")
        model_paths = [
            _ROOT_DIR / "models" / "phobert" / "best_model.pt",
            Path("models/phobert/best_model.pt"),
        ]
        model_path = next((p for p in model_paths if p.exists()), None)
        if model_path is None:
            self._loaded = True
            self.load_error = "Khong tim thay models/phobert/best_model.pt"
            raise FileNotFoundError(self.load_error)

        try:
            from transformers import AutoTokenizer
            local_tok = _ROOT_DIR / "models" / "phobert" / "base_model"
            if not local_tok.exists():
                local_tok = _ROOT_DIR / "models" / "phobert" / "tokenizer"
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(local_tok) if local_tok.exists() else "vinai/phobert-base-v2"
            )
            print("[PhoBERT] Tokenizer loaded")
        except Exception as e:
            self._loaded = True
            self.load_error = f"Khong load duoc tokenizer PhoBERT: {e}"
            raise RuntimeError(self.load_error) from e

        try:
            backbone_config_dir = _ROOT_DIR / "models" / "phobert" / "base_model"
            if not (backbone_config_dir / "config.json").exists():
                raise FileNotFoundError(f"Khong tim thay {backbone_config_dir / 'config.json'}")

            ckpt = _torch_load_checkpoint(model_path, self.device)
            self.model = AppPhoBERTASQP(backbone_config_dir).to(self.device)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_path = model_path
            print(f"[PhoBERT] Model loaded from {model_path} on {self.device}")
        except Exception as e:
            self.model = None
            self._loaded = True
            self.load_error = f"Khong load duoc best_model PhoBERT: {e}"
            raise RuntimeError(self.load_error) from e

        self._loaded = True

    def predict_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            return []
        return self.predict_sentences([sentence])[0]

    def predict_sentences(self, sentences: List[str]) -> List[List[Dict[str, Any]]]:
        self._load_model()
        normalized = [sentence.strip() for sentence in sentences]
        results: List[Optional[List[Dict[str, Any]]]] = [None] * len(normalized)
        uncached_sentences: List[str] = []
        uncached_indices: List[int] = []
        for idx, sentence in enumerate(normalized):
            if not sentence or len(sentence) < 10:
                results[idx] = []
            elif sentence in self.cache:
                results[idx] = self.cache[sentence]
            else:
                uncached_indices.append(idx)
                uncached_sentences.append(sentence)

        if not uncached_sentences:
            return [item or [] for item in results]

        if self.model is not None and self.tokenizer is not None:
            try:
                batch_results = self._predict_batch_with_model(uncached_sentences)
            except Exception as e:
                if self.device.type == "cuda" and "out of memory" in str(e).lower():
                    print("[PhoBERT] CUDA OOM, switching inference to CPU...")
                    torch.cuda.empty_cache()
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    batch_results = self._predict_batch_with_model(uncached_sentences)
                else:
                    raise RuntimeError(f"PhoBERT inference error: {e}") from e

            for idx, sentence, result in zip(uncached_indices, uncached_sentences, batch_results):
                self.cache[sentence] = result
                results[idx] = result
            return [item or [] for item in results]

        raise RuntimeError(self.load_error or "PhoBERT best_model chua duoc load")

    def _predict_batch_with_model(self, sentences: List[str]) -> List[List[Dict[str, Any]]]:
        if not sentences:
            return []
        enc = self.tokenizer(
            sentences,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits_per_cat = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
        outputs: List[List[Dict[str, Any]]] = [[] for _ in sentences]
        for cat_idx, logits in enumerate(logits_per_cat):
            probs = F.softmax(logits, dim=-1)
            confs, pred_indices = torch.max(probs, dim=-1)
            for row_idx, pred_idx_tensor in enumerate(pred_indices):
                pred_idx = int(pred_idx_tensor.item())
                pred_name = SENTIMENT_NAMES[pred_idx]
                conf = float(confs[row_idx].item())
                if pred_name != "none" and conf >= self.confidence_threshold:
                    outputs[row_idx].append({
                        "aspect": CATEGORIES[cat_idx],
                        "sentiment": pred_name,
                        "confidence": conf,
                        "sentiment_score": SENTIMENT_TO_SCORE[pred_name],
                    })
        return outputs

    def _predict_with_model(self, sentence: str) -> List[Dict[str, Any]]:
        return self._predict_batch_with_model([sentence])[0]

    def _rule_based_predict(self, sentence: str) -> List[Dict[str, Any]]:
        sl = sentence.lower()
        pos_kws = ["tốt","hay","đẹp","chất lượng","tuyệt","ổn","hài lòng","thích","ưng",
                   "ngon","bền","đáng tiền","mềm","dễ chịu","thoải mái","tiện lợi",
                   "an toàn","chắc chắn","rẻ","hợp lý"]
        neg_kws = ["tệ","hỏng","rách","xấu","kém","không tốt","chán","thất vọng",
                   "mỏng","cứng","khó chịu","đắt","chậm","không hài lòng","lỗi","tồi","bẩn"]
        pos_score = sum(1 for kw in pos_kws if kw in sl)
        neg_score = sum(1 for kw in neg_kws if kw in sl)
        if pos_score > neg_score:
            sentiment, score = "positive", 0.7
        elif neg_score > pos_score:
            sentiment, score = "negative", -0.7
        else:
            sentiment, score = "neutral", 0.0
        confidence = min(0.8, 0.5 + abs(pos_score - neg_score) / 10)
        results = []
        for aspect, keywords in ASPECT_REQUIRED_KEYWORDS.items():
            if not any(kw in sl for kw in keywords):
                continue
            forbidden = ASPECT_FORBIDDEN_KEYWORDS.get(aspect, [])
            if any(kw in sl for kw in forbidden):
                continue
            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": confidence,
                "sentiment_score": score,
            })
            if len(results) >= 4:
                break
        return results

