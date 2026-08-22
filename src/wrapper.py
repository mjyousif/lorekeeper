from __future__ import annotations

import json
import logging
import os
import time
import uuid

from .chat_manager import ChatManager
from .config import Config, get_config
from .document_loader import DocumentLoader
from .text_chunker import TextChunker
from .vector_store import ChromaVectorStore, VectorStore

logger = logging.getLogger(__name__)


class LoreKeeper:
    """Main LoreKeeper logic class."""

    def __init__(
        self,
        config: Config,
        vector_store: VectorStore | None = None,
        files: list[str] | str | None = None,
    ):
        """Initialize the LoreKeeper wrapper."""
        init_start = time.perf_counter()
        self.config = config
        logger.info("Initializing LoreKeeper...")

        # Configure logging
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper(), logging.INFO),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        else:
            logging.getLogger().setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))

        # Core components
        raw_files = files if files is not None else self.config.files
        exclude_paths = []
        if self.config.character_file:
            exclude_paths.append(self.config.character_file)
        if self.config.context_file:
            exclude_paths.append(self.config.context_file)
        logger.debug("Excluding paths from indexing: %s", exclude_paths)

        self.document_loader = DocumentLoader(raw_files, exclude_paths=exclude_paths)
        logger.info("DocumentLoader resolved %d files", len(self.document_loader.files))
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            chunk_threshold=self.config.chunk_threshold,
        )
        logger.debug(
            "TextChunker configured: chunk_size=%d overlap=%d threshold=%d",
            self.config.chunk_size, self.config.overlap, self.config.chunk_threshold,
        )

        self.db_path = self.config.db_path
        self.vector_store = vector_store or ChromaVectorStore(db_path=self.db_path)
        logger.info("VectorStore initialized at %s (%d existing documents)", self.db_path, self.vector_store.count())

        # Context and character
        self.context = ""
        self.character = ""
        self._load_context_character()

        # Chat and history management
        llm_cfg = self.config.llm or {}
        self.chat_manager = ChatManager(
            llm_model=llm_cfg.get("model"),
            llm_api_key=llm_cfg.get("api_key"),
            llm_api_base=llm_cfg.get("api_base"),
            max_context_size=int(llm_cfg.get("max_context_size", 64000)),
            context=self.context,
            character=self.character,
        )

        # Initialize vector store
        self._load_and_embed_files()

        # Sessions dictionary mapping ID to history list
        self.sessions: dict[str, list[dict]] = {}

        init_elapsed = time.perf_counter() - init_start
        logger.info("LoreKeeper fully initialized in %.2fs", init_elapsed)

    @property
    def files(self) -> list[str]:
        """Provides backwards compatibility for accessing loaded file paths."""
        return self.document_loader.files

    @property
    def _manifest(self) -> dict:
        """Provides backwards compatibility for manifest access."""
        return self.document_loader._manifest

    @_manifest.setter
    def _manifest(self, val):
        """Set the manifest value."""
        self.document_loader._manifest = val

    def _load_context_character(self):
        """Load context and character files from config paths."""
        if self.config.context_file:
            try:
                with open(self.config.context_file, "r", encoding="utf-8") as f:
                    self.context = f.read().strip()
                logger.info(
                    "Loaded context from %s (%d chars)",
                    self.config.context_file, len(self.context),
                )
            except Exception as e:
                logger.error(
                    "Failed to read context file %s: %s", self.config.context_file, e
                )
        else:
            logger.debug("No context_file configured")

        if self.config.character_file:
            try:
                with open(self.config.character_file, "r", encoding="utf-8") as f:
                    self.character = f.read().strip()
                logger.info(
                    "Loaded character from %s (%d chars)",
                    self.config.character_file, len(self.character),
                )
            except Exception as e:
                logger.error(
                    "Failed to read character file %s: %s",
                    self.config.character_file,
                    e,
                )
        else:
            logger.debug("No character_file configured")

    def _rebuild_index(self):
        """Delete the collection and re-embed all files from scratch."""
        logger.info("Data changes detected. Rebuilding index...")
        self.vector_store.clear()
        self.document_loader.update_files()
        self._load_and_embed_files(force=True)
        logger.info("Index rebuild complete.")

    def _load_and_embed_files(self, force: bool = False):
        """Load files, chunk them, and store them in the vector DB.

        Args:
            force: If True, embed regardless of whether the collection is non-empty.
        """
        manifest_path = os.path.join(self.db_path, "manifest.json")
        persisted_manifest = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    persisted_manifest = json.load(f)
            except Exception as e:
                logger.warning("Failed to load persisted manifest: %s", e)

        current_manifest = self.document_loader.scan_files()
        serializable_manifest = {k: list(v) for k, v in current_manifest.items()}

        needs_rebuild = (
            force
            or self.vector_store.count() == 0
            or persisted_manifest is None
            or persisted_manifest != serializable_manifest
        )

        if not needs_rebuild:
            logger.info(
                "Collection is up-to-date (%d documents). Skipping embedding.",
                self.vector_store.count(),
            )
            # Sync manifest in loader so needs_rebuild() uses correct state
            self.document_loader._manifest = current_manifest
            return

        logger.info("Loading and embedding %d files...", len(self.document_loader.files))
        embed_start = time.perf_counter()
        self.vector_store.clear()
        total_chunks = 0
        for file_path in self.document_loader.files:
            try:
                content = self.document_loader.read_file(file_path)
                chunks = self.text_chunker.chunk_text(content)

                if chunks:
                    ids = [str(uuid.uuid4()) for _ in chunks]
                    self.vector_store.insert(
                        documents=chunks,
                        metadatas=[{"source": file_path} for _ in chunks],
                        ids=ids,
                    )
                    total_chunks += len(chunks)
                    logger.debug("Embedded %d chunks from %s (%d chars)", len(chunks), file_path, len(content))
                else:
                    logger.debug("No chunks produced from %s (%d chars)", file_path, len(content))
            except Exception as e:
                logger.error("Error processing file %s: %s", file_path, e)
        embed_elapsed = time.perf_counter() - embed_start
        logger.info(
            "Finished embedding: %d total chunks from %d files in %.2fs",
            total_chunks, len(self.document_loader.files), embed_elapsed,
        )

        try:
            os.makedirs(self.db_path, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(serializable_manifest, f, indent=2)
        except Exception as e:
            logger.error("Failed to save manifest to %s: %s", manifest_path, e)

    def get_relevant_context(self, message: str, n_results: int = 3) -> list[str]:
        """Query the vector store to get context relevant to the message."""
        logger.debug("Querying vector store for: %s (n_results=%d)", message[:80], n_results)
        query_start = time.perf_counter()
        results = self.vector_store.query(message, n_results=n_results)
        query_elapsed = time.perf_counter() - query_start
        logger.info(
            "Vector query returned %d results in %.3fs",
            len(results), query_elapsed,
        )
        return results

    def chat(self, session_id: str, message: str) -> dict:
        """Handle chat: retrieve context, manage history, call LLM, return response."""
        logger.info("[session=%s] chat() called with message (%d chars)", session_id, len(message))
        chat_start = time.perf_counter()

        # 0. Rebuild index if data files changed
        if self.document_loader.needs_rebuild():
            logger.info("[session=%s] Data changes detected, rebuilding index...", session_id)
            self._rebuild_index()

        # 1. Retrieve relevant context
        context = self.get_relevant_context(message)

        # 2. Manage conversation history
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            logger.debug("[session=%s] Created new session", session_id)
        history = self.sessions[session_id]
        logger.debug("[session=%s] Current history: %d messages", session_id, len(history))

        # 3. Ask ChatManager for response
        assistant_message = self.chat_manager.generate_response(
            message=message,
            retrieved_context=context,
            history=history,
        )

        if not assistant_message.startswith("Error calling LLM"):
            # 4. Update history (only if no error)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": assistant_message})
            logger.debug("[session=%s] History updated to %d messages", session_id, len(history))
        else:
            logger.warning("[session=%s] LLM returned error, history not updated", session_id)

        chat_elapsed = time.perf_counter() - chat_start
        logger.info(
            "[session=%s] chat() completed in %.2fs (response=%d chars)",
            session_id, chat_elapsed, len(assistant_message),
        )
        return {"message": assistant_message, "context": context}


if __name__ == "__main__":
    print("Starting LoreKeeper example...")

    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
    with open("test_docs/doc1.txt", "w") as f:
        f.write("The first rule of Fight Club is: you do not talk about Fight Club.")
    with open("test_docs/doc2.txt", "w") as f:
        f.write("The sky is blue and the grass is green. The sun is a star.")

    config = get_config()
    lorekeeper = LoreKeeper(config, files="test_docs")

    session_id = "test_session_123"

    print("\n--- Query 1 ---")
    response1 = lorekeeper.chat(session_id, "What is the primary rule of the club?")
    print("\nWrapper Response:", response1)

    print("\n--- Query 2 ---")
    response2 = lorekeeper.chat(session_id, "What color is the sky?")
    print("\nWrapper Response:", response2)
