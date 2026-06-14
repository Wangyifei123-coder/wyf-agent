"""Embedding 服务 — 封装 text2vec 模型"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name)
        return self._model

    def embed_query(self, query: str) -> list[float]:
        model = self._load_model()
        vec = model.encode(query, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vecs]
