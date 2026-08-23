"""Test the LoreKeeper class."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Config
from src.core.wrapper import LoreKeeper
from src.rag.vector_store import VectorStore


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


class TestLoreKeeperInitialization:

    def test_init_with_single_directory(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Content 1")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(data_dir))

        assert wrapper.files == [str(data_dir / "doc1.txt")]

    def test_init_with_list_of_files(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=[str(file1), str(file2)])

        assert len(wrapper.files) == 2

    def test_init_with_mixed_files_and_directories(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Dir doc")
        file1 = tmp_path / "file1.txt"
        file1.write_text("File doc")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=[str(data_dir), str(file1)])

        assert len(wrapper.files) == 2

    def test_init_creates_sessions_dict(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Test content")
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(tmp_path))

        assert wrapper.sessions == {}

    def test_init_scans_files_and_updates_manifest(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Test content")
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(tmp_path))

        assert len(wrapper._manifest) > 0
        assert str(tmp_path / "doc.txt") in wrapper._manifest

    def test_init_with_custom_vector_store(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Test content")
        mock_store = MagicMock(spec=VectorStore)
        mock_store.count.return_value = 0

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(tmp_path), vector_store=mock_store)

        assert wrapper.vector_store is mock_store

    def test_init_loads_and_embeds_files(self, tmp_path):
        (tmp_path / "doc.txt").write_text(
            "The water cycle involves evaporation and precipitation."
        )
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(tmp_path))

        assert wrapper.vector_store.count() > 0

    def test_init_skips_embedding_if_collection_exists(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Original content")
        db_path = str(tmp_path / "db")
        cfg = make_config(db_path=db_path)

        wrapper1 = LoreKeeper(config=cfg, files=str(tmp_path))
        assert wrapper1.vector_store.count() > 0

        (tmp_path / "doc.txt").write_text("Modified content")

        wrapper2 = LoreKeeper(config=cfg, files=str(tmp_path))
        assert wrapper2.vector_store.count() > 0

    def test_init_handles_missing_context_file(self, tmp_path, caplog):
        import logging

        caplog.set_level(logging.ERROR)

        missing_file = tmp_path / "missing_context.txt"
        cfg = make_config(db_path=str(tmp_path / "db"), context_file=str(missing_file))

        # Initialize LoreKeeper with missing context file
        wrapper = LoreKeeper(config=cfg, files=str(tmp_path))

        # Verify the context is empty and the error was logged
        assert wrapper.context == ""
        assert f"Failed to read context file {missing_file}" in caplog.text


class TestLoreKeeperFileOperations:

    @pytest.fixture
    def wrapper(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Short content.")
        (data_dir / "doc2.txt").write_text("X" * 3000)
        cfg = make_config(db_path=str(tmp_path / "db"))
        return LoreKeeper(config=cfg, files=str(data_dir))

    def test_resolve_files_single_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(file_path))

        assert wrapper.document_loader.resolve_files(str(file_path)) == [str(file_path)]

    def test_resolve_files_directory_recursive(self, tmp_path):
        data_dir = tmp_path / "data"
        subdir = data_dir / "sub"
        subdir.mkdir(parents=True)
        (data_dir / "root.txt").write_text("root")
        (subdir / "nested.md").write_text("nested")
        (subdir / "ignore.jpg").write_text("image")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(data_dir))
        resolved = wrapper.document_loader.resolve_files(str(data_dir))

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
        wrapper = LoreKeeper(config=cfg, files=str(data_dir))
        resolved = wrapper.document_loader.resolve_files(str(data_dir))

        assert len(resolved) == 3
        assert {os.path.splitext(f)[1] for f in resolved} == {".txt", ".md", ".pdf"}

    def test_read_file_txt(self, wrapper, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Plain text content", encoding="utf-8")
        assert wrapper.document_loader.read_file(str(file_path)) == "Plain text content"

    def test_read_file_md(self, wrapper, tmp_path):
        file_path = tmp_path / "test.md"
        file_path.write_text("# Markdown content", encoding="utf-8")
        assert wrapper.document_loader.read_file(str(file_path)) == "# Markdown content"

    def test_read_file_pdf(self, wrapper, tmp_path):
        file_path = tmp_path / "test2.pdf"
        file_path.write_bytes(open("tests/sample.pdf", "rb").read())
        assert "Hello, World!" in wrapper.document_loader.read_file(str(file_path))

    def test_read_file_not_found(self, wrapper, tmp_path):
        import re

        non_existent_file = str(tmp_path / "does_not_exist.txt")
        with pytest.raises(
            FileNotFoundError, match=re.escape(f"File not found: {non_existent_file}")
        ):
            wrapper.document_loader.read_file(non_existent_file)

    def test_chunk_text_splits_correctly(self, wrapper):
        text = "A" * 25000
        from src.rag.text_chunker import TextChunker

        chunker = TextChunker(chunk_size=1000, overlap=200, chunk_threshold=0)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        assert len(chunks[0]) == 1000
        assert chunks[0] == "A" * 1000

    def test_chunk_text_empty_input(self, wrapper):
        from src.rag.text_chunker import TextChunker

        chunker = TextChunker()
        assert chunker.chunk_text("") == []

    def test_chunk_text_overlap(self, wrapper):
        text = "0123456789" * 2000
        from src.rag.text_chunker import TextChunker

        chunker = TextChunker(chunk_size=500, overlap=100, chunk_threshold=0)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

        if len(chunks) >= 2:
            assert chunks[0][-100:] == chunks[1][:100]


class TestLoreKeeperManifestAndRebuild:

    @pytest.fixture
    def wrapper_with_files(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file1 = data_dir / "doc1.txt"
        file2 = data_dir / "doc2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(data_dir))
        return wrapper, data_dir, file1, file2

    def test_scan_files_returns_mtime_and_size(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        manifest = wrapper.document_loader.scan_files()

        assert str(file1) in manifest
        assert isinstance(manifest[str(file1)], tuple)
        assert len(manifest[str(file1)]) == 2

    def test_needs_rebuild_detects_new_file(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        assert not wrapper.document_loader.needs_rebuild()

        (data_dir / "doc3.txt").write_text("Content 3")
        assert wrapper.document_loader.needs_rebuild()

    def test_needs_rebuild_detects_modified_file(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        assert not wrapper.document_loader.needs_rebuild()

        file1.write_text("Modified content")
        assert wrapper.document_loader.needs_rebuild()

    def test_needs_rebuild_detects_deleted_file(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        assert not wrapper.document_loader.needs_rebuild()

        file1.unlink()
        assert wrapper.document_loader.needs_rebuild()

    def test_rebuild_index_clears_and_reembeds(self, wrapper_with_files):
        wrapper, data_dir, file1, file2 = wrapper_with_files
        file1.write_text("Modified")
        wrapper._rebuild_index()
        assert wrapper.vector_store.count() > 0


class TestLoreKeeperVectorStoreIntegration:

    @pytest.fixture
    def wrapper(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "philosophy.txt").write_text(
            "Stoicism teaches self-control, resilience, and rational thinking."
        )
        (data_dir / "ethics.txt").write_text(
            "Utilitarianism holds that the best action maximizes overall well-being."
        )

        cfg = make_config(db_path=str(tmp_path / "db"))
        return LoreKeeper(config=cfg, files=str(data_dir))

    def test_get_relevant_context_returns_matches(self, wrapper):
        results = wrapper.get_relevant_context("What does Stoicism teach?", n_results=2)
        assert len(results) > 0
        assert any("Stoicism" in doc for doc in results)

    def test_get_relevant_context_empty_query(self, wrapper):
        results = wrapper.get_relevant_context("", n_results=2)
        assert isinstance(results, list)

    def test_vector_store_black_box(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Test content for search.")

        class CustomStore(VectorStore):
            def __init__(self):
                self._docs = []

            def insert(self, documents, metadatas=None, ids=None):
                self._docs.extend(documents)

            def query(self, query_text, n_results=3):
                return self._docs[:n_results]

            def clear(self):
                self._docs.clear()

            def count(self):
                return len(self._docs)

        custom_store = CustomStore()
        cfg = make_config(db_path=str(tmp_path / "db"))
        wrapper = LoreKeeper(config=cfg, files=str(data_dir), vector_store=custom_store)

        assert isinstance(wrapper.vector_store, CustomStore)
        assert wrapper.vector_store.count() > 0


class TestLoreKeeperChat:

    @pytest.fixture
    def wrapper(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "science.txt").write_text(
            "Photosynthesis is the process by which plants convert sunlight into energy."
        )

        cfg = make_config(
            db_path=str(tmp_path / "db"),
            llm={"model": "test-model", "api_key": "test-key"},
        )
        return LoreKeeper(config=cfg, files=str(data_dir))

    def test_chat_creates_new_session_if_not_exists(self, wrapper):
        with patch("src.core.chat_manager.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Test response"
            mock_choice.message.tool_calls = None
            mock_completion.return_value.choices = [mock_choice]

            response = wrapper.chat(session_id="new_session", message="Hello")

        assert "new_session" in wrapper.sessions
        assert response["message"] == "Test response"

    def test_chat_retrieves_context_via_tool(self, wrapper):
        # We simulate the LLM returning a tool call for memory_search, then returning a normal response
        with patch("src.core.chat_manager.litellm.completion") as mock_completion:
            tool_call = MagicMock()
            tool_call.id = "call_123"
            tool_call.function.name = "memory_search"
            import json
            tool_call.function.arguments = json.dumps({"query": "What is photosynthesis?"})
            
            tool_choice = MagicMock()
            tool_choice.message.content = None
            tool_choice.message.tool_calls = [tool_call]
            tool_choice.message.model_dump.return_value = {
                "role": "assistant",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "memory_search", "arguments": tool_call.function.arguments}}]
            }
            
            final_choice = MagicMock()
            final_choice.message.content = "Photosynthesis is the process."
            final_choice.message.tool_calls = None
            
            mock_completion.side_effect = [
                MagicMock(choices=[tool_choice]),
                MagicMock(choices=[final_choice])
            ]

            wrapper.chat(session_id="test", message="What is photosynthesis?")

            assert mock_completion.call_count == 2
            messages_second_call = mock_completion.call_args_list[1][1]["messages"]
            
            # Check that the tool result was added to the history
            tool_msg = messages_second_call[-1]
            assert tool_msg["role"] == "tool"
            assert tool_msg["name"] == "memory_search"
            assert "Photosynthesis is the process by which plants convert sunlight into energy." in tool_msg["content"]

    def test_chat_manages_conversation_history(self, wrapper):
        session_id = "history_test"

        with patch("src.core.chat_manager.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Reply 1"
            mock_choice.message.tool_calls = None
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id=session_id, message="First message")
            assert len(wrapper.sessions[session_id]) == 2

            wrapper.chat(session_id=session_id, message="Second message")
            assert len(wrapper.sessions[session_id]) == 4

            messages = mock_completion.call_args[1]["messages"]
            assert len(messages) == 4  # system + 2 history + current user

    def test_chat_handles_llm_error(self, wrapper):
        with patch("src.core.chat_manager.litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("API error")
            response = wrapper.chat(session_id="test", message="Hello")
            assert "Error calling LLM" in response["message"]

    def test_chat_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Content")

        cfg = make_config(db_path=str(tmp_path / "db"), llm={})
        wrapper = LoreKeeper(config=cfg, files=str(data_dir))

        response = wrapper.chat(session_id="test", message="Hello")
        assert "LLM not configured" in response["message"]

    def test_chat_rebuilds_on_file_change(self, wrapper, tmp_path):
        data_dir = tmp_path / "data"

        with patch("src.core.chat_manager.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Reply"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id="test", message="Hello")
            initial_count = wrapper.vector_store.count()

        (data_dir / "new.txt").write_text("New content")
        wrapper._manifest = {}

        with patch("src.core.chat_manager.litellm.completion") as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Reply"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id="test", message="Hello again")

        assert wrapper.vector_store.count() >= initial_count


class TestLoreKeeperWithPdf:

    def test_read_pdf_file(self, tmp_path):
        from src.core.wrapper import LoreKeeper

        pdf_dir = tmp_path / "pdf_data"
        pdf_dir.mkdir()
        pdf_path = pdf_dir / "test.pdf"
        pdf_path.write_bytes(open("tests/sample.pdf", "rb").read())

        from src.core.config import Config

        keeper = LoreKeeper(config=Config(), files=[str(pdf_dir)])
        assert "Hello, World!" in keeper.document_loader.read_file(str(pdf_path))


@pytest.mark.integration
class TestLoreKeeperEndToEnd:

    def test_full_rag_pipeline(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "science_facts.txt").write_text("""
        Photosynthesis is the process plants use to convert sunlight into energy.
        Its key inputs are carbon dioxide, water, and light.
        Chlorophyll is the primary pigment responsible for absorbing light.
        """)

        cfg = make_config(
            db_path=str(tmp_path / "db"),
            llm={"api_key": os.environ.get("OPENROUTER_API_KEY")},
        )
        wrapper = LoreKeeper(config=cfg, files=str(data_dir))

        assert wrapper.vector_store.count() > 0
        context = wrapper.get_relevant_context("What are the inputs to photosynthesis?")
        assert len(context) > 0
        assert any(
            "carbon dioxide" in doc.lower() or "sunlight" in doc.lower()
            for doc in context
        )
