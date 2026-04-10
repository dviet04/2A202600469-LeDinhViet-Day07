from __future__ import annotations

from typing import Any, Callable, List

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document
import math

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0
        self._query_cache: dict[str, list[float]] = {}
        self._query_cache_max = 256

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = _normalize(self._embedding_fn(doc.content))  

        record_id = f"{doc.id}_{self._next_index}"

        record = {
            "id": record_id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": {
                **(doc.metadata or {}),
                "doc_id": doc.id,
                "chunk_id": self._next_index,
            },
        }

        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        if not records:
            return []

        if query in self._query_cache:
            query_vec = self._query_cache[query]
        else:
            query_vec = _normalize(self._embedding_fn(query))
            if len(self._query_cache) >= self._query_cache_max:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query] = query_vec

        scored = []
        for r in records:
            score = _cosine(query_vec, r["embedding"])  
            scored.append({
                "content": r["content"],
                "metadata": r["metadata"],
                "score": score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        if self._use_chroma:
            ids, documents, embeddings, metadatas = [], [], [], []

            for doc in docs:
                if not doc.content:
                    continue
                record = self._make_record(doc)

                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])

            if ids:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
        else:
            seen_ids: set[str] = set()
            for doc in docs:
                if not doc.content or doc.id in seen_ids:
                    continue
                seen_ids.add(doc.id)
                record = self._make_record(doc)
                self._store.append(record)
        # raise NotImplementedError("Implement EmbeddingStore.add_documents")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if self._use_chroma:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )

            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # 🔥 COSINE SIMILARITY
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]

        return self._search_records(query, self._store, top_k)
        # raise NotImplementedError("Implement EmbeddingStore.search")

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)
        # raise NotImplementedError("Implement EmbeddingStore.get_collection_size")

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=metadata_filter,
            )

            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # 🔥 cosine
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]

        filtered = [
            r for r in self._store
            if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]

        return self._search_records(query, filtered, top_k)
        # raise NotImplementedError("Implement EmbeddingStore.search_with_filter")

    def delete_document(self, id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['id'] == id
        if self._use_chroma:
            # delete theo metadata filter
            self._collection.delete(where={"doc_id": id})
            return True

        initial_len = len(self._store)

        self._store = [
            r for r in self._store
            if r["metadata"].get("doc_id") != id
        ]

        return len(self._store) < initial_len
        # raise NotImplementedError("Implement EmbeddingStore.delete_document")
