from __future__ import annotations

from app.src.v2.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.model_name = settings.embedding_model
        self._model = None
        self._dimension: int | None = None
        self.enabled = bool(self.model_name)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            vector = self.embed_text("dimension probe")
            self._dimension = len(vector)
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean = [str(text or "").strip() for text in texts if str(text or "").strip()]
        if not clean or not self.enabled:
            return []
        try:
            model = self._load_model()
            vectors = model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
            return [vector.astype(float).tolist() for vector in vectors]
        except Exception:
            self.enabled = False
            return []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
