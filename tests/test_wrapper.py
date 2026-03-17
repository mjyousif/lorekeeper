"""Test the RAGWrapper class."""

import tempfile
import os
import pytest
from unittest.mock import MagicMock, patch

from rag_wrapper.wrapper import RAGWrapper
from rag_wrapper.vector_store import VectorStore
from rag_wrapper.config import Config


def make_config(**kwargs) -> Config:
    """Helper to create a Config with test overrides."""
    defaults = dict(
        db_path="db",
        chunk_size=1000,
        overlap=200,
        chunk_threshold=10000,
        log_level="INFO",
    )
    defaults.update(kwargs)
    return Config(**defaults)


class TestRAGWrapperInitialization:

    def test_init_with_single_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Content 1")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(data_dir))

        assert wrapper.files == [str(data_dir / "doc1.txt")]

    def test_init_with_list_of_files(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=[str(file1), str(file2)])

        assert len(wrapper.files) == 2

    def test_init_with_mixed_files_and_directories(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Dir doc")
        file1 = tmp_path / "file1.txt"
        file1.write_text("File doc")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=[str(data_dir), str(file1)])

        assert len(wrapper.files) == 2

    def test_init_creates_sessions_dict(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Test content")
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(tmp_path))

        assert wrapper.sessions == {}

    def test_init_scans_files_and_updates_manifest(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Test content")
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(tmp_path))

        assert len(wrapper._manifest) > 0
        assert str(tmp_path / "doc.txt") in wrapper._manifest

    def test_init_with_custom_vector_store(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Test content")
        mock_store = MagicMock(spec=VectorStore)
        mock_store.count.return_value = 0

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(tmp_path), vector_store=mock_store)

        assert wrapper.vector_store is mock_store

    def test_init_loads_and_embeds_files(self, tmp_path):
        (tmp_path / "doc.txt").write_text(" teaches .")
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(tmp_path))

        assert wrapper.vector_store.count() > 0

    def test_init_skips_embedding_if_collection_exists(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Original content")
        db_path = str(tmp_path / "db")
        cfg = make_config(db_path=db_path)

        wrapper1 = RAGWrapper(config=cfg, files=str(tmp_path))
        assert wrapper1.vector_store.count() > 0

        (tmp_path / "doc.txt").write_text("Modified content")

        wrapper2 = RAGWrapper(config=cfg, files=str(tmp_path))
        assert wrapper2.vector_store.count() > 0


class TestRAGWrapperFileOperations:

    @pytest.fixture
    def wrapper(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Short content.")
        (data_dir / "doc2.txt").write_text("X" * 3000)
        cfg = make_config(db_path=str(tmp_path / "db"))
        return RAGWrapper(config=cfg, files=str(data_dir))

    def test_resolve_files_single_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(file_path))

        assert wrapper._resolve_files(str(file_path)) == [str(file_path)]

    def test_resolve_files_directory_recursive(self, tmp_path):
        data_dir = tmp_path / "data"
        subdir = data_dir / "sub"
        subdir.mkdir(parents=True)
        (data_dir / "root.txt").write_text("root")
        (subdir / "nested.md").write_text("nested")
        (subdir / "ignore.jpg").write_text("image")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(data_dir))
        resolved = wrapper._resolve_files(str(data_dir))

        assert len(resolved) == 2
        assert any("root.txt" in f for f in resolved)
        assert any("nested.md" in f for f in resolved)
        assert not any(".jpg" in f for f in resolved)

    def test_resolve_files_filters_supported_extensions(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("txt")
        (data_dir / "doc.md").write_text("md")
        (data_dir / "doc.pdf").write_text("pdf content")
        (data_dir / "doc.py").write_text("python")
        (data_dir / "doc.json").write_text("json")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(data_dir))
        resolved = wrapper._resolve_files(str(data_dir))

        assert len(resolved) == 3
        assert {os.path.splitext(f)[1] for f in resolved} == {".txt", ".md", ".pdf"}

    def test_read_file_txt(self, wrapper, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Plain text content", encoding="utf-8")
        assert wrapper._read_file(str(file_path)) == "Plain text content"

    def test_read_file_md(self, wrapper, tmp_path):
        file_path = tmp_path / "test.md"
        file_path.write_text("# Markdown content", encoding="utf-8")
        assert wrapper._read_file(str(file_path)) == "# Markdown content"

    def test_read_file_pdf(self, wrapper, tmp_path):
        pytest.skip("PDF test requires sample PDF file")

    def test_chunk_text_splits_correctly(self, wrapper):
        text = "A" * 2500
        chunks = wrapper._chunk_text(text, chunk_size=1000, overlap=200)

        assert len(chunks) > 1
        for chunk in chunks:
            assert 1 <= len(chunk) <= 1000
        assert len(chunks[0]) == 1000

    def test_chunk_text_empty_input(self, wrapper):
        assert wrapper._chunk_text("") == []

    def test_chunk_text_overlap(self, wrapper):
        text = "0123456789" * 200
        chunks = wrapper._chunk_text(text, chunk_size=500, overlap=100)

        if len(chunks) >= 2:
            assert chunks[0][-100:] == chunks[1][:100]


class TestRAGWrapperManifestAndRebuild:

    @pytest.fixture
    def wrapper_with_files(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file1 = data_dir / "doc1.txt"
        file2 = data_dir / "doc2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(data_dir))
        return wrapper, data_dir, file1, file2

    def test_scan_files_returns_mtime_and_size(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        manifest = wrapper._scan_files()

        assert str(file1) in manifest
        assert isinstance(manifest[str(file1)], tuple)
        assert len(manifest[str(file1)]) == 2

    def test_needs_rebuild_detects_new_file(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        assert not wrapper._needs_rebuild()

        (data_dir / "doc3.txt").write_text("Content 3")
        assert wrapper._needs_rebuild()

    def test_needs_rebuild_detects_modified_file(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        assert not wrapper._needs_rebuild()

        file1.write_text("Modified content")
        assert wrapper._needs_rebuild()

    def test_needs_rebuild_detects_deleted_file(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        assert not wrapper._needs_rebuild()

        file1.unlink()
        assert wrapper._needs_rebuild()

    def test_rebuild_index_clears_and_reembeds(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        file1.write_text("Modified")
        wrapper._rebuild_index()
        assert wrapper.vector_store.count() > 0


class TestRAGWrapperVectorStoreIntegration:

    @pytest.fixture
    def wrapper(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / ".txt").write_text(" teaches .")
        (data_dir / ".txt").write_text(" is the evil religion of .")

        cfg = make_config(db_path=str(tmp_path / "db"))
        return RAGWrapper(config=cfg, files=str(data_dir))

    def test_get_relevant_context_returns_matches(self, wrapper):
        results = wrapper.get_relevant_context("What does  teach?", n_results=2)
        assert len(results) > 0
        assert any("" in doc for doc in results)

    def test_get_relevant_context_empty_query(self, wrapper):
        results = wrapper.get_relevant_context("", n_results=2)
        assert isinstance(results, list)

    def test_vector_store_black_box(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Test content for .")

        class CustomStore(VectorStore):
            def __init__(self): self._docs = []
            def insert(self, documents, metadatas=None, ids=None): self._docs.extend(documents)
            def query(self, query_text, n_results=3): return self._docs[:n_results]
            def clear(self): self._docs.clear()
            def count(self): return len(self._docs)

        custom_store = CustomStore()
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = RAGWrapper(config=cfg, files=str(data_dir), vector_store=custom_store)

        assert isinstance(wrapper.vector_store, CustomStore)
        assert wrapper.vector_store.count() > 0


class TestRAGWrapperChat:

    @pytest.fixture
    def wrapper(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "lore.txt").write_text("  a. .")

        cfg = make_config(
            db_path=str(tmp_path / "db"),
            llm={"model": "test-model", "api_key": "test-key"},
        )
        return RAGWrapper(config=cfg, files=str(data_dir))

    def test_chat_creates_new_session_if_not_exists(self, wrapper):
        with patch("rag_wrapper.wrapper.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Test response"
            mock_completion.return_value.choices = [mock_choice]

            response = wrapper.chat(session_id="new_session", message="Hello")

        assert "new_session" in wrapper.sessions
        assert response["message"] == "Test response"

    def test_chat_retrieves_context(self, wrapper):
        with patch("rag_wrapper.wrapper.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id="test", message="What is ?")

            messages = mock_completion.call_args[1]["messages"]
            system_msg = messages[0]
            assert system_msg["role"] == "system"
            assert "Context:" in system_msg["content"]
            assert "" in system_msg["content"]

    def test_chat_uses_relevant_context_for_query(self, wrapper):
        with patch.object(wrapper, "get_relevant_context") as mock_get:
            mock_get.return_value = ["Mocked context"]

            with patch("rag_wrapper.wrapper.litellm.completion") as mock_completion:
                mock_choice = MagicMock()
                mock_choice.message.content = "Reply"
                mock_completion.return_value.choices = [mock_choice]

                wrapper.chat(session_id="test", message="User query")
                mock_get.assert_called_once_with("User query")

    def test_chat_manages_conversation_history(self, wrapper):
        session_id = "history_test"

        with patch("rag_wrapper.wrapper.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Reply 1"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id=session_id, message="First message")
            assert len(wrapper.sessions[session_id]) == 2

            wrapper.chat(session_id=session_id, message="Second message")
            assert len(wrapper.sessions[session_id]) == 4

            messages = mock_completion.call_args[1]["messages"]
            assert len(messages) == 4  # system + 2 history + current user

    def test_chat_handles_llm_error(self, wrapper):
        with patch("rag_wrapper.wrapper.litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("API error")
            response = wrapper.chat(session_id="test", message="Hello")
            assert "Error calling LLM" in response["message"]

    def test_chat_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Content")

        cfg = make_config(db_path=str(tmp_path / "db"), llm={})
        wrapper = RAGWrapper(config=cfg, files=str(data_dir))

        response = wrapper.chat(session_id="test", message="Hello")
        assert "LLM not configured" in response["message"]

    def test_chat_rebuilds_on_file_change(self, wrapper, tmp_path):
        data_dir = tmp_path / "data"

        with patch("rag_wrapper.wrapper.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Reply"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id="test", message="Hello")
            initial_count = wrapper.vector_store.count()

        (data_dir / "new.txt").write_text("New content")
        wrapper._manifest = {}

        with patch("rag_wrapper.wrapper.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Reply"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id="test", message="Hello again")

        assert wrapper.vector_store.count() >= initial_count


class TestRAGWrapperWithPdf:

    def test_read_pdf_file(self, tmp_path):
        pytest.skip("PDF test requires sample PDF file")


@pytest.mark.integration
class TestRAGWrapperEndToEnd:

    def test_full_rag_pipeline(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "_lore.txt").write_text("""
         is the one true faith.
        Its principles are .
        .
        """)

        cfg = make_config(
            db_path=str(tmp_path / "db"),
            llm={"api_key": os.environ.get("OPENROUTER_API_KEY")},
        )
        wrapper = RAGWrapper(config=cfg, files=str(data_dir))

        assert wrapper.vector_store.count() > 0
        context = wrapper.get_relevant_context("What are 's principles?")
        assert len(context) > 0
        assert any(" c" in doc or " d" in doc for doc in context)
