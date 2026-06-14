"""知识图谱检索器测试"""

from unittest.mock import MagicMock, patch
from src.rag.kg_retriever import KnowledgeGraphRetriever
from src.rag.loader import Document


class TestKnowledgeGraphRetriever:
    def test_extract_entities(self):
        entities = KnowledgeGraphRetriever._extract_entities("Python和RAG技术有什么关系？")
        assert any("Python" in e or "RAG" in e for e in entities)

    def test_extract_entities_tech_terms(self):
        entities = KnowledgeGraphRetriever._extract_entities("LangGraph是LangChain团队开发的框架")
        assert any("LangGraph" in e for e in entities)

    def test_search_returns_empty_when_not_connected(self):
        retriever = KnowledgeGraphRetriever()
        results = retriever.search("test", top_k=5)
        assert len(results) == 0

    def test_add_documents_skips_when_not_connected(self):
        retriever = KnowledgeGraphRetriever()
        docs = [Document(content="test", metadata={})]
        retriever.add_documents(docs)

    @patch("neo4j.GraphDatabase")
    def test_search_with_mock(self, mock_graph_db):
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []

        retriever = KnowledgeGraphRetriever()
        retriever._connected = True
        retriever._driver = mock_driver

        results = retriever.search("Python", top_k=5)
        assert isinstance(results, list)
