"""Test the VectorStore abstraction and Chroma implementation."""

import tempfile
import os
import pytest
from pathlib import Path

from rag_wrapper.vector_store import VectorStore, ChromaVectorStore


class TestVectorStoreInterface:
    """Test the VectorStore abstract base class."""

    def test_abstract_methods_raise_not_implemented(self):
        """All VectorStore methods should raise NotImplementedError."""

        class DummyStore(VectorStore):
            def __init__(self, db_path, embedding_model=None):
                pass

            # Not implementing any methods

        store = DummyStore("test_path")
        with pytest.raises(NotImplementedError):
            store.insert(["doc1"])
        with pytest.raises(NotImplementedError):
            store.query("test")
        with pytest.raises(NotImplementedError):
            store.clear()
        with pytest.raises(NotImplementedError):
            store.count()


class TestChromaVectorStore:
    """Test the ChromaVectorStore implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for the test database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create a fresh ChromaVectorStore for each test."""
        db_path = os.path.join(temp_dir, "test_db")
        store = ChromaVectorStore(db_path=db_path)
        return store

    def test_init_creates_db_path(self, temp_dir):
        """Initialization should set up the database path."""
        db_path = os.path.join(temp_dir, "new_db")
        store = ChromaVectorStore(db_path=db_path)
        assert Path(db_path).exists()

    def test_insert_adds_documents(self, vector_store):
        """Insert should add documents to the store."""
        docs = ["doc1 content", "doc2 content", "doc3 content"]
        metadatas = [{"source": "file1"}, {"source": "file2"}, {"source": "file3"}]
        ids = ["id1", "id2", "id3"]

        vector_store.insert(documents=docs, metadatas=metadatas, ids=ids)

        assert vector_store.count() == 3

    def test_insert_without_metadatas_or_ids(self, vector_store):
        """Insert should work without metadatas or ids (auto-generated)."""
        docs = ["doc1", "doc2"]

        vector_store.insert(documents=docs)

        assert vector_store.count() == 2

    def test_insert_validates_lengths(self, vector_store):
        """Insert should validate that lists match documents length."""
        docs = ["doc1", "doc2"]
        ids = ["id1"]  # Wrong length

        with pytest.raises(ValueError, match="ids length must match documents length"):
            vector_store.insert(documents=docs, ids=ids)

    def test_query_returns_relevant_docs(self, vector_store):
        """Query should return relevant documents based on similarity."""
        # Insert documents with known content
        docs = [
            "The sky is blue and the sun shines brightly.",
            "Grass is green and grows in fields.",
            " teaches .",
            "  b .",
        ]
        vector_store.insert(documents=docs)

        # Query for something about sky
        results = vector_store.query("sky color", n_results=2)
        assert len(results) <= 2
        assert any("sky" in doc.lower() for doc in results)

    def test_query_returns_empty_list_if_no_docs(self, vector_store):
        """Query should return empty list if store has no documents."""
        # Ensure store is empty (just created)
        assert vector_store.count() == 0

        results = vector_store.query("any query", n_results=3)
        assert results == []

    def test_query_n_results_respected(self, vector_store):
        """Query should respect n_results parameter."""
        docs = [f"Document number {i}" for i in range(10)]
        vector_store.insert(documents=docs)

        results = vector_store.query("document", n_results=5)
        assert len(results) == 5

    def test_clear_removes_all_data(self, vector_store):
        """Clear should remove all documents from the store."""
        docs = ["doc1", "doc2", "doc3"]
        vector_store.insert(documents=docs)
        assert vector_store.count() == 3

        vector_store.clear()
        assert vector_store.count() == 0

    def test_clear_handles_nonexistent_collection(self, temp_dir):
        """Clear should handle case where collection doesn't exist."""
        db_path = os.path.join(temp_dir, "fresh_db")
        # Create store but don't insert anything
        store = ChromaVectorStore(db_path=db_path)

        # Should not raise an error
        store.clear()
        assert store.count() == 0

    def test_count_returns_correct_number(self, vector_store):
        """Count should return the number of documents."""
        assert vector_store.count() == 0

        docs = ["doc1", "doc2", "doc3"]
        vector_store.insert(documents=docs)
        assert vector_store.count() == 3

        # Insert more
        vector_store.insert(documents=["doc4", "doc5"])
        assert vector_store.count() == 5

    def test_metadata_preserved_on_insert(self, vector_store):
        """Metadata should be stored and retrievable (indirectly via query)."""
        docs = [" promotes  c and  d."]
        metadatas = [{"source": "_lore.txt", "chapter": "1"}]

        vector_store.insert(documents=docs, metadatas=metadatas)

        # Query to retrieve the document
        results = vector_store.query("", n_results=1)
        assert len(results) == 1
        assert "" in results[0]

    def test_multiple_inserts_accumulate(self, vector_store):
        """Multiple insert calls should accumulate documents."""
        vector_store.insert(documents=["first batch"])
        assert vector_store.count() == 1

        vector_store.insert(documents=["second batch", "third batch"])
        assert vector_store.count() == 3

    def test_duplicate_documents_allowed(self, vector_store):
        """Vector stores typically allow duplicate content (different IDs)."""
        docs = ["same content", "same content"]
        vector_store.insert(documents=docs)
        assert vector_store.count() == 2
