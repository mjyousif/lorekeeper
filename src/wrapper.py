from __future__ import annotations

import os
import logging
import uuid

from .vector_store import VectorStore, ChromaVectorStore
from .config import Config, get_config
from .document_loader import DocumentLoader
from .text_chunker import TextChunker
from .chat_manager import ChatManager

logger = logging.getLogger(__name__)


class LoreKeeper:
    def __init__(
        self,
        config: Config,
        vector_store: VectorStore | None = None,
        files: list[str] | str | None = None,
    ):
        self.config = config

        # Configure logging
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper(), logging.INFO),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Core components
        raw_files = files if files is not None else self.config.files
        self.document_loader = DocumentLoader(raw_files)
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            chunk_threshold=self.config.chunk_threshold,
        )

        self.db_path = self.config.db_path
        self.vector_store = vector_store or ChromaVectorStore(db_path=self.db_path)

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
        self.document_loader._manifest = val

    def _load_context_character(self):
        """Load context and character files from config paths."""
        if self.config.context_file:
            try:
                with open(self.config.context_file, "r", encoding="utf-8") as f:
                    self.context = f.read().strip()
                logger.info("Loaded context from %s", self.config.context_file)
            except Exception as e:
                logger.error(
                    "Failed to read context file %s: %s", self.config.context_file, e
                )

        if self.config.character_file:
            try:
                with open(self.config.character_file, "r", encoding="utf-8") as f:
                    self.character = f.read().strip()
                logger.info("Loaded character from %s", self.config.character_file)
            except Exception as e:
                logger.error(
                    "Failed to read character file %s: %s",
                    self.config.character_file,
                    e,
                )

    def _resolve_files(self, input_paths: list[str] | str) -> list[str]:
        """Deprecated: Handled by DocumentLoader."""
        return self.document_loader.resolve_files(input_paths)

    def _scan_files(self) -> dict[str, tuple[float, int]]:
        """Deprecated: Handled by DocumentLoader."""
        return self.document_loader.scan_files()

    def _needs_rebuild(self) -> bool:
        """Deprecated: Handled by DocumentLoader."""
        return self.document_loader.needs_rebuild()

    def _rebuild_index(self):
        """Delete the collection and re-embed all files from scratch."""
        logger.info("Data changes detected. Rebuilding index...")
        self.vector_store.clear()
        self.document_loader.update_files()
        self._load_and_embed_files(force=True)
        logger.info("Index rebuild complete.")

    def _read_file(self, file_path: str) -> str:
        """Deprecated: Handled by DocumentLoader."""
        return self.document_loader.read_file(file_path)

    def _chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """Deprecated: Handled by TextChunker."""
        # Using a temporary chunker to respect backward compatibility
        # if anyone was relying on passing varying chunk_sizes.
        temp_chunker = TextChunker(chunk_size, overlap, 0)
        return temp_chunker.chunk_text(text)

    def _load_and_embed_files(self, force: bool = False):
        """Load files, chunk them, and store them in the vector DB.

        Args:
            force: If True, embed regardless of whether the collection is non-empty.
        """
        if not force and self.vector_store.count() > 0:
            logger.info("Collection already contains documents. Skipping embedding.")
            return

        logger.info("Loading and embedding files...")
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
                    logger.debug("Embedded %d chunks from %s", len(chunks), file_path)
            except Exception as e:
                logger.error("Error processing file %s: %s", file_path, e)
        logger.info("Finished loading files.")

    def get_relevant_context(self, message: str, n_results: int = 3) -> list[str]:
        """Query the vector store to get context relevant to the message."""
        logger.debug("Querying vector store for: %s (n_results=%d)", message, n_results)
        return self.vector_store.query(message, n_results=n_results)

    def chat(self, session_id: str, message: str) -> dict:
        """Handle chat: retrieve context, manage history, call LLM, return response."""
        # 0. Rebuild index if data files changed
        if self.document_loader.needs_rebuild():
            logger.info("Data changes detected, rebuilding index...")
            self._rebuild_index()

        # 1. Retrieve relevant context
        context = self.get_relevant_context(message)

        # 2. Manage conversation history
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            logger.debug("Created new session: %s", session_id)
        history = self.sessions[session_id]

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
