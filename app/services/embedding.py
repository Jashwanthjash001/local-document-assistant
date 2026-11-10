"""
Embedding service — Stage 3.

Wraps the local sentence-transformers model (`all-MiniLM-L6-v2`). The model is
loaded **lazily and exactly once** (first use), then reused — loading it per
request would be slow and waste memory.

The LLM and this embedding model are different components: this one only turns
text into 384-dim vectors for retrieval; it never generates answers.
"""
from __future__ import annotations

from app import config


class EmbeddingService:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None   # loaded on first use

    @property
    def model(self):
        if self._model is None:
            # Imported here so importing this module stays cheap and so a missing
            # heavy dependency only fails when embeddings are actually used.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns normalized vectors as plain lists.

        Normalization is applied consistently (L2) so cosine similarity in the
        vector store behaves well. Empty input returns an empty list without
        touching the model.
        """
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def count_tokens(self, text: str) -> int:
        """Token count using the model's own tokenizer (for accurate chunking)."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# --- module singleton ------------------------------------------------------
_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService(config.EMBEDDING_MODEL, config.EMBEDDING_DEVICE)
    return _service
