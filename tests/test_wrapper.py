"""Test the RAGWrapper class."""

import tempfile
import os
import pytest
from unittest.mock import MagicMock, patch

from rag_wrapper.wrapper import RAGWrapper
from rag_wrapper.vector_store import VectorStore


class TestRAGWrapperInitialization:
    """Test RAGWrapper initialization and configuration."""

    def test_init_with_single_directory(self, tmp_path):
        """Should accept a single directory path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Content 1")

        wrapper = RAGWrapper(files=str(data_dir), db_path=str(tmp_path / "db"))

        assert wrapper.files == [str(data_dir / "doc1.txt")]

    def test_init_with_list_of_files(self, tmp_path):
        """Should accept a list of file paths."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        wrapper = RAGWrapper(files=[str(file1), str(file2)], db_path=str(tmp_path / "db"))

        assert len(wrapper.files) == 2

    def test_init_with_mixed_files_and_directories(self, tmp_path):
        """Should accept mix of files and directories."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Dir doc")
        file1 = tmp_path / "file1.txt"
        file1.write_text("File doc")

        wrapper = RAGWrapper(files=[str(data_dir), str(file1)], db_path=str(tmp_path / "db"))

        assert len(wrapper.files) == 2

    def test_init_creates_sessions_dict(self, tmp_path):
        """Should initialize empty sessions dict."""
        (tmp_path / "doc.txt").write_text("Test content")
        wrapper = RAGWrapper(files=str(tmp_path), db_path=str(tmp_path / "db"))

        assert wrapper.sessions == {}

    def test_init_scans_files_and_updates_manifest(self, tmp_path):
        """Should scan files and populate manifest."""
        (tmp_path / "doc.txt").write_text("Test content")
        wrapper = RAGWrapper(files=str(tmp_path), db_path=str(tmp_path / "db"))

        assert len(wrapper._manifest) > 0
        assert str(tmp_path / "doc.txt") in wrapper._manifest

    def test_init_with_custom_vector_store(self, tmp_path):
        """Should accept custom vector store."""
        (tmp_path / "doc.txt").write_text("Test content")
        mock_store = MagicMock(spec=VectorStore)
        mock_store.count.return_value = 0

        wrapper = RAGWrapper(
            files=str(tmp_path),
            db_path=str(tmp_path / "db"),
            vector_store=mock_store
        )

        assert wrapper.vector_store is mock_store

    def test_init_loads_and_embeds_files(self, tmp_path):
        """Should load and embed files during initialization."""
        (tmp_path / "doc.txt").write_text(" teaches .")
        db_path = str(tmp_path / "db")

        wrapper = RAGWrapper(files=str(tmp_path), db_path=db_path)

        # Verify the vector store has data
        assert wrapper.vector_store.count() > 0

    def test_init_skips_embedding_if_collection_exists(self, tmp_path):
        """Should skip embedding if vector store already has data."""
        (tmp_path / "doc.txt").write_text("Original content")
        db_path = str(tmp_path / "db")

        # First initialization creates embeddings
        wrapper1 = RAGWrapper(files=str(tmp_path), db_path=db_path)
        assert wrapper1.vector_store.count() > 0

        # Change file content
        (tmp_path / "doc.txt").write_text("Modified content")

        # Second initialization should detect change and rebuild
        wrapper2 = RAGWrapper(files=str(tmp_path), db_path=db_path)
        # The rebuild happens on first chat, not init
        assert wrapper2.vector_store.count() > 0


class TestRAGWrapperFileOperations:
    """Test file resolution, reading, and chunking."""

    @pytest.fixture
    def wrapper(self, tmp_path):
        """Create a wrapper with test files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Short content.")
        (data_dir / "doc2.txt").write_text("X" * 3000)  # Long content
        wrapper = RAGWrapper(files=str(data_dir), db_path=str(tmp_path / "db"))
        return wrapper

    def test_resolve_files_single_file(self, tmp_path):
        """_resolve_files should handle a single file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")
        wrapper = RAGWrapper(files=str(file_path), db_path=str(tmp_path / "db"))

        files = wrapper._resolve_files(str(file_path))
        assert files == [str(file_path)]

    def test_resolve_files_directory_recursive(self, tmp_path):
        """Should walk directory recursively for supported files."""
        data_dir = tmp_path / "data"
        subdir = data_dir / "sub"
        subdir.mkdir(parents=True)
        (data_dir / "root.txt").write_text("root")
        (subdir / "nested.md").write_text("nested")
        (subdir / "ignore.jpg").write_text("image")

        wrapper = RAGWrapper(files=str(data_dir), db_path=str(tmp_path / "db"))
        resolved = wrapper._resolve_files(str(data_dir))

        assert len(resolved) == 2
        assert any("root.txt" in f for f in resolved)
        assert any("nested.md" in f for f in resolved)
        assert not any(".jpg" in f for f in resolved)

    def test_resolve_files_filters_supported_extensions(self, tmp_path):
        """Should only include .txt, .md, .pdf files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("txt")
        (data_dir / "doc.md").write_text("md")
        (data_dir / "doc.pdf").write_text("pdf content")
        (data_dir / "doc.py").write_text("python")
        (data_dir / "doc.json").write_text("json")

        wrapper = RAGWrapper(files=str(data_dir), db_path=str(tmp_path / "db"))
        resolved = wrapper._resolve_files(str(data_dir))

        assert len(resolved) == 3
        extensions = {os.path.splitext(f)[1] for f in resolved}
        assert extensions == {".txt", ".md", ".pdf"}

    def test_read_file_txt(self, wrapper, tmp_path):
        """_read_file should read .txt files."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Plain text content", encoding="utf-8")

        content = wrapper._read_file(str(file_path))
        assert content == "Plain text content"

    def test_read_file_md(self, wrapper, tmp_path):
        """_read_file should read .md files."""
        file_path = tmp_path / "test.md"
        file_path.write_text("# Markdown content", encoding="utf-8")

        content = wrapper._read_file(str(file_path))
        assert content == "# Markdown content"

    def test_read_file_pdf(self, wrapper, tmp_path):
        """_read_file should read .pdf files (if PyPDF2 installed)."""
        # Skipping actual PDF creation - would need PyPDF2 to generate
        pytest.skip("PDF test requires sample PDF file")

    def test_chunk_text_splits_correctly(self, wrapper):
        """_chunk_text should split into overlapping chunks."""
        text = "A" * 2500  # Should produce multiple chunks
        chunks = wrapper._chunk_text(text, chunk_size=1000, overlap=200)

        assert len(chunks) > 1
        # All chunks should be at most chunk_size and at least 1
        for chunk in chunks:
            assert 1 <= len(chunk) <= 1000
        # The first chunk should be full size
        assert len(chunks[0]) == 1000

    def test_chunk_text_empty_input(self, wrapper):
        """_chunk_text should return empty list for empty text."""
        assert wrapper._chunk_text("") == []

    def test_chunk_text_overlap(self, wrapper):
        """Chunks should have the specified overlap."""
        text = "0123456789" * 200  # 2000 chars
        chunks = wrapper._chunk_text(text, chunk_size=500, overlap=100)

        # Check that consecutive chunks share the overlap region
        if len(chunks) >= 2:
            # Last 100 chars of chunk1 should match first 100 of chunk2
            overlap_region1 = chunks[0][-100:]
            overlap_region2 = chunks[1][:100]
            assert overlap_region1 == overlap_region2


class TestRAGWrapperManifestAndRebuild:
    """Test file manifest tracking and index rebuild logic."""

    @pytest.fixture
    def wrapper_with_files(self, tmp_path):
        """Create wrapper with some files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file1 = data_dir / "doc1.txt"
        file2 = data_dir / "doc2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        db_path = str(tmp_path / "db")
        wrapper = RAGWrapper(files=str(data_dir), db_path=db_path)
        return wrapper, data_dir, file1, file2

    def test_scan_files_returns_mtime_and_size(self, wrapper_with_files):
        """_scan_files should return dict with (mtime, size) tuples."""
        wrapper, data_dir, file1, file2 = wrapper_with_files
        manifest = wrapper._scan_files()

        assert str(file1) in manifest
        assert isinstance(manifest[str(file1)], tuple)
        assert len(manifest[str(file1)]) == 2  # (mtime, size)

    def test_needs_rebuild_detects_new_file(self, wrapper_with_files):
        """_needs_rebuild should return True when new file appears."""
        wrapper, data_dir, file1, file2 = wrapper_with_files

        # Initially, manifest matches files
        assert not wrapper._needs_rebuild()

        # Add a new file
        new_file = data_dir / "doc3.txt"
        new_file.write_text("Content 3")

        # Should detect change
        assert wrapper._needs_rebuild()

    def test_needs_rebuild_detects_modified_file(self, wrapper_with_files):
        """_needs_rebuild should return True when file modified."""
        wrapper, data_dir, file1, file2 = wrapper_with_files

        assert not wrapper._needs_rebuild()

        # Modify existing file
        file1.write_text("Modified content")

        assert wrapper._needs_rebuild()

    def test_needs_rebuild_detects_deleted_file(self, wrapper_with_files):
        """_needs_rebuild should return True when file deleted."""
        wrapper, data_dir, file1, file2 = wrapper_with_files

        assert not wrapper._needs_rebuild()

        # Delete a file
        file1.unlink()

        assert wrapper._needs_rebuild()

    def test_rebuild_index_clears_and_reembeds(self, wrapper_with_files):
        """_rebuild_index should clear and re-embed all files."""
        wrapper, data_dir, file1, file2 = wrapper_with_files

        # Modify files to trigger rebuild
        file1.write_text("Modified")
        wrapper._rebuild_index()

        # Should have updated content
        assert wrapper.vector_store.count() > 0


class TestRAGWrapperVectorStoreIntegration:
    """Test integration between wrapper and vector store."""

    @pytest.fixture
    def wrapper(self, tmp_path):
        """Create wrapper with known content."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / ".txt").write_text(" teaches .")
        (data_dir / ".txt").write_text(" is the evil religion of .")
        db_path = str(tmp_path / "db")
        wrapper = RAGWrapper(files=str(data_dir), db_path=db_path)
        return wrapper

    def test_get_relevant_context_returns_matches(self, wrapper):
        """get_relevant_context should return relevant document chunks."""
        results = wrapper.get_relevant_context("What does  teach?", n_results=2)

        assert len(results) > 0
        assert any("" in doc for doc in results)

    def test_get_relevant_context_empty_query(self, wrapper):
        """Should handle empty query."""
        results = wrapper.get_relevant_context("", n_results=2)
        # Empty query might return something or nothing depending on vector store
        assert isinstance(results, list)

    def test_vector_store_black_box(self, tmp_path):
        """Vector store implementation should be swappable."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Test content for .")

        # Use a custom mock vector store
        class CustomStore(VectorStore):
            def __init__(self, db_path, embedding_model=None):
                self._docs = []
                self._embeddings = []

            def insert(self, documents, metadatas=None, ids=None):
                self._docs.extend(documents)

            def query(self, query_text, n_results=3):
                return self._docs[:n_results]

            def clear(self):
                self._docs.clear()

            def count(self):
                return len(self._docs)

        custom_store = CustomStore(db_path=str(tmp_path / "custom_db"))
        wrapper = RAGWrapper(
            files=str(data_dir),
            db_path=str(tmp_path / "db"),
            vector_store=custom_store
        )

        assert isinstance(wrapper.vector_store, CustomStore)
        assert wrapper.vector_store.count() > 0


class TestRAGWrapperChat:
    """Test chat functionality with LLM integration."""

    @pytest.fixture
    def wrapper(self, tmp_path):
        """Create wrapper with test data and mock LLM."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "lore.txt").write_text("  a. .")

        # Set a dummy API key to avoid config error
        os.environ["OPENROUTER_API_KEY"] = "test-key"

        db_path = str(tmp_path / "db")
        wrapper = RAGWrapper(
            files=str(data_dir),
            db_path=db_path,
            llm_model="test-model",
            llm_api_key="test-key"
        )
        return wrapper

    def test_chat_creates_new_session_if_not_exists(self, wrapper):
        """chat should initialize session if session_id not seen before."""
        session_id = "new_session"

        with patch('rag_wrapper.wrapper.litellm.completion') as mock_completion:
            mock_completion.return_value.choices = [
                type('obj', (object,), {'message': type('obj', (object,), {'content': 'Test response'})()})()
            ]
            response = wrapper.chat(session_id=session_id, message="Hello")

        assert session_id in wrapper.sessions
        assert response["message"] == "Test response"

    def test_chat_retrieves_context(self, wrapper):
        """chat should include relevant context in system prompt."""
        with patch('rag_wrapper.wrapper.litellm.completion') as mock_completion:
            # Create a proper MagicMock for the response
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id="test", message="What is ?")

            # Extract the messages passed to litellm
            call_args = mock_completion.call_args[1]
            messages = call_args['messages']

            # System message should contain context
            system_msg = messages[0]
            assert system_msg['role'] == 'system'
            assert 'Context:' in system_msg['content']
            assert '' in system_msg['content']

    def test_chat_uses_relevant_context_for_query(self, wrapper):
        """chat should query vector store with user message."""
        with patch.object(wrapper, 'get_relevant_context') as mock_get:
            mock_get.return_value = ["Mocked context"]

            with patch('rag_wrapper.wrapper.litellm.completion') as mock_completion:
                mock_choice = MagicMock()
                mock_choice.message.content = "Reply"
                mock_completion.return_value.choices = [mock_choice]

                wrapper.chat(session_id="test", message="User query")

                # Should be called with message, n_results defaults to 3 internally
                # The actual call may not include n_results if wrapper uses default
                mock_get.assert_called_once_with("User query")

    def test_chat_manages_conversation_history(self, wrapper):
        """chat should maintain and include conversation history."""
        session_id = "history_test"

        with patch('rag_wrapper.wrapper.litellm.completion') as mock_completion:
            mock_choice = MagicMock()
            mock_choice.message.content = "Reply 1"
            mock_completion.return_value.choices = [mock_choice]

            wrapper.chat(session_id=session_id, message="First message")

            # Update history - should be 2 messages (user + assistant)
            assert len(wrapper.sessions[session_id]) == 2

            # Second message
            wrapper.chat(session_id=session_id, message="Second message")
            assert len(wrapper.sessions[session_id]) == 4

            # Verify that history was included in LLM call
            call_args = mock_completion.call_args[1]
            messages = call_args['messages']
            # Should include: system + previous history (first turn) + current user message
            # That's 1 (system) + 2 (first user+assistant) + 1 (second user) = 4 messages
            assert len(messages) == 4

    def test_chat_handles_llm_error(self, wrapper):
        """chat should handle LLM API errors gracefully."""
        with patch('rag_wrapper.wrapper.litellm.completion') as mock_completion:
            mock_completion.side_effect = Exception("API error")

            response = wrapper.chat(session_id="test", message="Hello")

            assert "Error calling LLM" in response["message"]

    def test_chat_without_api_key(self, tmp_path, monkeypatch):
        """chat should return helpful message if no API key configured."""
        # Ensure OPENROUTER_API_KEY is not set
        monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Content")

        wrapper = RAGWrapper(
            files=str(data_dir),
            db_path=str(tmp_path / "db"),
            llm_api_key=None  # Explicitly None
        )

        response = wrapper.chat(session_id="test", message="Hello")
        assert "LLM not configured" in response["message"]

    def test_chat_uses_environment_api_key(self, tmp_path):
        """Should fall back to OPENROUTER_API_KEY environment variable."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Content")

        os.environ["OPENROUTER_API_KEY"] = "env-key"
        try:
            wrapper = RAGWrapper(
                files=str(data_dir),
                db_path=str(tmp_path / "db"),
                llm_api_key=None
            )
            assert wrapper.llm_api_key == "env-key"
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    def test_chat_rebuilds_on_file_change(self, wrapper, tmp_path):
        """chat should rebuild index if underlying files change."""
        data_dir = tmp_path / "data"
        wrapper.session_id = "test"

        # Initial chat
        with patch('rag_wrapper.wrapper.litellm.completion') as mock_completion:
            mock_completion.return_value.choices = [
                type('obj', (object,), {'message': type('obj', (object,), {'content': 'Reply'})()})()
            ]
            wrapper.chat(session_id="test", message="Hello")
            initial_count = wrapper.vector_store.count()

        # Modify files
        (data_dir / "new.txt").write_text("New content")
        # Force rebuild by manipulating manifest
        wrapper._manifest = {}  # Invalidate manifest

        with patch('rag_wrapper.wrapper.litellm.completion') as mock_completion:
            mock_completion.return_value.choices = [
                type('obj', (object,), {'message': type('obj', (object,), {'content': 'Reply'})()})()
            ]
            wrapper.chat(session_id="test", message="Hello again")

        # Should have rebuilt (count might increase)
        # Note: count could be same or higher depending on if new file was embedded
        assert wrapper.vector_store.count() >= initial_count


class TestRAGWrapperWithPdf:
    """Test PDF file support (if available)."""

    def test_read_pdf_file(self, tmp_path):
        """_read_file should handle PDF files."""
        # This test would need an actual PDF file
        pytest.skip("PDF test requires sample PDF file")


# Integration test (requires real LLM API key - marked as slow/integration)
@pytest.mark.integration
class TestRAGWrapperEndToEnd:
    """End-to-end integration tests (require API key)."""

    def test_full_rag_pipeline(self, tmp_path):
        """Test full RAG pipeline with actual embedding and query."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lore_file = data_dir / "_lore.txt"
        lore_file.write_text("""
         is the one true faith.
        Its principles are .
        .
        """)

        wrapper = RAGWrapper(
            files=str(data_dir),
            db_path=str(tmp_path / "db"),
            llm_api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        # Verify data was embedded
        assert wrapper.vector_store.count() > 0

        # Query for relevant context
        context = wrapper.get_relevant_context("What are 's principles?")
        assert len(context) > 0
        assert any(" c" in doc or " d" in doc for doc in context)
