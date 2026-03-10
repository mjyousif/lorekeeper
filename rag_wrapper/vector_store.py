"""Vector storage abstraction for RAG.

Provides a clean black-box interface for embedding storage and retrieval.
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path


class VectorStore:
    """Abstract interface for vector storage backends.

    Implementations must provide methods for inserting documents and querying
    by similarity.
    """

    def __init__(self, db_path: str | Path, embedding_model: Any = None):
        """Initialize the vector store.

        Args:
            db_path: Path to the database/storage directory
            embedding_model: Optional embedding model instance
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        """Query the vector store for relevant documents.

        Args:
            query_text: The search query
            n_results: Number of results to return

        Returns:
            List of matching document texts
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all data from the store."""
        raise NotImplementedError

    def count(self) -> int:
        """Return the number of documents in the store."""
        raise NotImplementedError


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
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.db_path = str(db_path)
        self.collection_name = collection_name

        # Use provided embedding model or create default
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize Chroma client (persistent)
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Get or create collection with embedding function
        # We need to provide a Chroma embedding function that wraps our model
        class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
            def __init__(self, model):
                self.model = model

            def __call__(self, input: list[str]) -> list[list[float]]:
                # Chroma expects a list of embeddings, each is a list of floats
                embeddings = self.model.encode(input)
                return embeddings.tolist()

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
        """Insert documents into Chroma."""
        import uuid

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("ids length must match documents length")

        # Empty metadatas if None
        if metadatas is None:
            metadatas = [{} for _ in documents]
        elif len(metadatas) != len(documents):
            raise ValueError("metadatas length must match documents length")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        """Query Chroma for similar documents."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        # Results structure: {"documents": [[...]]}
        return results["documents"][0] if results["documents"] else []

    def clear(self) -> None:
        """Delete the collection and recreate it."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection may not exist
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
        )

    def count(self) -> int:
        """Return document count."""
        return self.collection.count()
