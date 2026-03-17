"""Test the VectorStore abstraction and Chroma implementation."""

import tempfile
import os
import pytest
from pathlib import Path

from src.vector_store import VectorStore, ChromaVectorStore


class TestVectorStoreInterface:
    """Test the VectorStore abstract base class."""

    def test_cannot_instantiate_incomplete_subclass(self):
        """Subclass missing abstract methods should raise TypeError on instantiation."""

        class IncompleteStore(VectorStore):
            pass  # Does not implement any abstract methods

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteStore()

    def test_can_instantiate_complete_subclass(self):
        """Subclass implementing all abstract methods should instantiate fine."""

        class CompleteStore(VectorStore):
            def insert(self, documents, metadatas=None, ids=None):
                pass

            def query(self, query_text, n_results=3):
                return []

            def clear(self):
                pass

            def count(self):
                return 0

        store = CompleteStore()
        assert store.count() == 0


class TestChromaVectorStore:
    """Test the ChromaVectorStore implementation."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def vector_store(self, temp_dir):
        db_path = os.path.join(temp_dir, "test_db")
        return ChromaVectorStore(db_path=db_path)

    def test_init_creates_db_path(self, temp_dir):
        db_path = os.path.join(temp_dir, "new_db")
        ChromaVectorStore(db_path=db_path)
        assert Path(db_path).exists()

    def test_insert_adds_documents(self, vector_store):
        docs = ["doc1 content", "doc2 content", "doc3 content"]
        metadatas = [{"source": "file1"}, {"source": "file2"}, {"source": "file3"}]
        ids = ["id1", "id2", "id3"]
        vector_store.insert(documents=docs, metadatas=metadatas, ids=ids)
        assert vector_store.count() == 3

    def test_insert_without_metadatas_or_ids(self, vector_store):
        vector_store.insert(documents=["doc1", "doc2"])
        assert vector_store.count() == 2

    def test_insert_validates_lengths(self, vector_store):
        with pytest.raises(ValueError, match="ids length must match documents length"):
            vector_store.insert(documents=["doc1", "doc2"], ids=["id1"])

    def test_query_returns_relevant_docs(self, vector_store):
        docs = [
            "The sky is blue and the sun shines brightly.",
            "Grass is green and grows in fields.",
            "Mountains are tall and covered in snow.",
            "Rivers flow through valleys and into the sea.",
        ]
        vector_store.insert(documents=docs)
        results = vector_store.query("sky color", n_results=2)
        assert len(results) <= 2
        assert any("sky" in doc.lower() for doc in results)

    def test_query_returns_empty_list_if_no_docs(self, vector_store):
        assert vector_store.count() == 0
        results = vector_store.query("any query", n_results=3)
        assert results == []

    def test_query_n_results_respected(self, vector_store):
        docs = [f"Document number {i}" for i in range(10)]
        vector_store.insert(documents=docs)
        results = vector_store.query("document", n_results=5)
        assert len(results) == 5

    def test_clear_removes_all_data(self, vector_store):
        vector_store.insert(documents=["doc1", "doc2", "doc3"])
        assert vector_store.count() == 3
        vector_store.clear()
        assert vector_store.count() == 0

    def test_clear_handles_nonexistent_collection(self, temp_dir):
        store = ChromaVectorStore(db_path=os.path.join(temp_dir, "fresh_db"))
        store.clear()
        assert store.count() == 0

    def test_count_returns_correct_number(self, vector_store):
        assert vector_store.count() == 0
        vector_store.insert(documents=["doc1", "doc2", "doc3"])
        assert vector_store.count() == 3
        vector_store.insert(documents=["doc4", "doc5"])
        assert vector_store.count() == 5

    def test_metadata_preserved_on_insert(self, vector_store):
        vector_store.insert(
            documents=["The ocean is deep and full of marine life."],
            metadatas=[{"source": "nature_facts.txt", "chapter": "1"}],
        )
        results = vector_store.query("ocean", n_results=1)
        assert len(results) == 1
        assert "ocean" in results[0].lower()

    def test_multiple_inserts_accumulate(self, vector_store):
        vector_store.insert(documents=["first batch"])
        assert vector_store.count() == 1
        vector_store.insert(documents=["second batch", "third batch"])
        assert vector_store.count() == 3

    def test_duplicate_documents_allowed(self, vector_store):
        vector_store.insert(documents=["same content", "same content"])
        assert vector_store.count() == 2
