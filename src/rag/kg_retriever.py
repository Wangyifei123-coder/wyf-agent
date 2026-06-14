"""知识图谱检索器 — 基于 Neo4j 的实体关系检索"""

from __future__ import annotations

from typing import Any

import structlog

from .loader import Document

logger = structlog.get_logger(__name__)


class KnowledgeGraphRetriever:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "wyf-agent-2024",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Any = None
        self._connected = False

    def _connect(self) -> bool:
        if self._connected:
            return True
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            self._connected = True
            logger.info("neo4j_connected", uri=self._uri)
            return True
        except Exception as e:
            logger.warning("neo4j_connection_failed", error=str(e))
            self._connected = False
            return False

    def add_documents(self, documents: list[Document]) -> None:
        if not self._connect():
            logger.warning("neo4j_not_available_skip_index")
            return

        with self._driver.session() as session:
            for doc in documents:
                entities = self._extract_entities(doc.content)
                for entity in entities:
                    session.run(
                        "MERGE (e:Entity {name: $name}) "
                        "SET e.source = $source, e.content = $content",
                        name=entity,
                        source=doc.metadata.get("source", ""),
                        content=doc.content[:500],
                    )

                for i in range(len(entities) - 1):
                    session.run(
                        "MATCH (a:Entity {name: $a}) "
                        "MATCH (b:Entity {name: $b}) "
                        "MERGE (a)-[:RELATED_TO]->(b)",
                        a=entities[i],
                        b=entities[i + 1],
                    )

        logger.info("kg_indexed", documents=len(documents))

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        if not self._connect():
            return []

        entities = self._extract_entities(query)
        if not entities:
            return []

        results: list[Document] = []
        seen: set[str] = set()

        with self._driver.session() as session:
            for entity in entities:
                records = session.run(
                    "MATCH (e:Entity) WHERE e.name CONTAINS $name "
                    "OPTIONAL MATCH (e)-[:RELATED_TO]-(related) "
                    "RETURN e.name AS name, e.content AS content, e.source AS source, "
                    "collect(DISTINCT related.name) AS related_names "
                    "LIMIT $limit",
                    name=entity,
                    limit=top_k,
                )
                for record in records:
                    content = record["content"] or ""
                    if content and content not in seen:
                        seen.add(content)
                        related = record["related_names"] or []
                        results.append(Document(
                            content=content,
                            metadata={
                                "source": record["source"] or "",
                                "kg_entity": record["name"],
                                "kg_related": related,
                                "relevance_score": 1.0,
                            },
                        ))

        logger.info("kg_search", query=query[:50], entities=entities, results=len(results))
        return results[:top_k]

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        import re
        entities: list[str] = []

        patterns = [
            r"[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*",
            r"RAG|LangGraph|Python|ChromaDB|Neo4j|FastAPI|LangChain",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        seen: set[str] = set()
        unique: list[str] = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique[:10]

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._connected = False
