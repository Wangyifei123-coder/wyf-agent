"""Embedding 服务测试"""

import pytest
from src.rag.embeddings import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture(scope="class")
    def service(self):
        return EmbeddingService(model_name="shibing624/text2vec-base-chinese")

    def test_embed_query_returns_vector(self, service):
        vec = service.embed_query("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_texts_returns_batch(self, service):
        texts = ["hello", "world", "test"]
        vecs = service.embed_texts(texts)
        assert len(vecs) == 3
        assert all(isinstance(v, list) for v in vecs)

    def test_embed_query_consistent(self, service):
        vec1 = service.embed_query("test query")
        vec2 = service.embed_query("test query")
        assert vec1 == vec2

    def test_embed_different_texts_differ(self, service):
        vec1 = service.embed_query("hello")
        vec2 = service.embed_query("completely different topic about quantum physics")
        assert vec1 != vec2
