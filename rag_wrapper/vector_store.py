from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Install required dependencies: pip install chromadb sentence-transformers"
    ) from e


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def insert(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """Insert documents into the vector store.

        Args:
            documents: List of document texts (chunks)
            metadatas: Optional list of metadata dicts per document
            ids: Optional list of unique IDs for documents
        """
        ...

    @abstractmethod
    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        """Query the vector store for relevant documents.

        Args:
            query_text: The search query
            n_results: Number of results to return

        Returns:
            List of matching document texts
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all data from the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        ...


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore.

    Uses Chroma with SentenceTransformer embeddings for local persistence.
    """

    def __init__(
        self,
        db_path: str | Path,
        embedding_model: Any = None,
        collection_name: str = "rag_collection",
    ):
        """Initialize Chroma vector store.

        Args:
            db_path: Path to Chroma database directory
            embedding_model: SentenceTransformer instance (if None, uses default)
            collection_name: Name of the Chroma collection
        """
        self.db_path = str(db_path)
        self.collection_name = collection_name

        self.embedding_model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")

        self.client = chromadb.PersistentClient(path=self.db_path)

        class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
            def __init__(self, model):
                self.model = model

            def __call__(self, input: list[str]) -> list[list[float]]:
                return self.model.encode(input).tolist()

        self.embedding_fn = SentenceTransformerEmbeddingFunction(self.embedding_model)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
        )

    def insert(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """Insert documents into Chroma.

        Args:
            documents: List of document texts (chunks)
            metadatas: Optional list of metadata dicts per document
            ids: Optional list of unique IDs for documents
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("ids length must match documents length")

        if metadatas is not None:
            if len(metadatas) != len(documents):
                raise ValueError("metadatas length must match documents length")
            for m in metadatas:
                if not m:
                    raise ValueError("metadata dicts must be non-empty")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        """Query Chroma for similar documents."""
        n_results = min(n_results, self.count())
        if n_results == 0:
            return []
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []

    def clear(self) -> None:
        """Delete the collection and recreate it."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
        )

    def count(self) -> int:
        """Return document count."""
        return self.collection.count()
