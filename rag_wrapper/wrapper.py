from __future__ import annotations

import os
import logging
import uuid

from pypdf import PdfReader
import litellm

from .vector_store import VectorStore, ChromaVectorStore
from .config import Config, get_config

logger = logging.getLogger(__name__)


class RAGWrapper:
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

        # Files with override support
        raw_files = files if files is not None else self.config.files
        self._file_spec = raw_files
        self.files = self._resolve_files(raw_files)
        self.db_path = self.config.db_path

        # Vector store
        self.vector_store = vector_store or ChromaVectorStore(db_path=self.db_path)

        # Chunking
        self.chunk_size = self.config.chunk_size
        self.overlap = self.config.overlap
        self.chunk_threshold = self.config.chunk_threshold

        self._load_and_embed_files()

        # LLM
        llm_cfg = self.config.llm or {}
        self.llm_model = llm_cfg.get("model")
        self.llm_api_key = llm_cfg.get("api_key")
        self.llm_api_base = llm_cfg.get("api_base")

        # Context and character
        self._load_context_character()

        # Sessions and manifest
        self.sessions: dict[str, list[dict]] = {}
        self._manifest = self._scan_files()

    def _load_context_character(self):
        """Load context and character files from config paths."""
        self.context = ""
        if self.config.context_file:
            try:
                with open(self.config.context_file, "r", encoding="utf-8") as f:
                    self.context = f.read().strip()
                logger.info("Loaded context from %s", self.config.context_file)
            except Exception as e:
                logger.error("Failed to read context file %s: %s", self.config.context_file, e)

        self.character = ""
        if self.config.character_file:
            try:
                with open(self.config.character_file, "r", encoding="utf-8") as f:
                    self.character = f.read().strip()
                logger.info("Loaded character from %s", self.config.character_file)
            except Exception as e:
                logger.error("Failed to read character file %s: %s", self.config.character_file, e)

    def _resolve_files(self, input_paths: list[str] | str) -> list[str]:
        """Return a flat list of readable files.

        * If a directory is provided, walk it recursively and include any
          `.txt`, `.md` or `.pdf` files.
        * If a list is provided it may contain files or directories.
        * Nonexistent paths are skipped with a warning.
        """
        allowed = (".txt", ".md", ".pdf")
        results: list[str] = []

        if isinstance(input_paths, str):
            input_paths = [input_paths]

        for path in input_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path, followlinks=True):
                    for fname in files:
                        if fname.lower().endswith(allowed):
                            results.append(os.path.join(root, fname))
            elif os.path.isfile(path):
                results.append(path)
            else:
                logger.warning("Path does not exist or is not a file/dir: %s", path)

        return results

    def _scan_files(self) -> dict[str, tuple[float, int]]:
        """Scan the current set of data files and return a dict of path → (mtime, size)."""
        manifest: dict[str, tuple[float, int]] = {}
        current_files = self._resolve_files(self._file_spec)
        for path in current_files:
            try:
                stat = os.stat(path)
                manifest[path] = (stat.st_mtime, stat.st_size)
            except Exception as e:
                logger.warning("Cannot stat %s: %s", path, e)
        return manifest

    def _needs_rebuild(self) -> bool:
        """Compare current files against the stored manifest."""
        current = self._scan_files()
        if set(current.keys()) != set(self._manifest.keys()):
            return True
        for path, info in current.items():
            if info != self._manifest.get(path):
                return True
        return False

    def _rebuild_index(self):
        """Delete the collection and re-embed all files from scratch."""
        logger.info("Data changes detected. Rebuilding index...")
        self.vector_store.clear()
        self.files = self._resolve_files(self._file_spec)
        self._load_and_embed_files(force=True)
        self._manifest = self._scan_files()
        logger.info("Index rebuild complete.")

    def _load_and_embed_files(self, force: bool = False):
        """Load files, chunk them, and store them in the vector DB.

        Args:
            force: If True, embed regardless of whether the collection is non-empty.
        """
        if not force and self.vector_store.count() > 0:
            logger.info("Collection already contains documents. Skipping embedding.")
            return

        logger.info("Loading and embedding files...")
        for file_path in self.files:
            try:
                content = self._read_file(file_path)
                if len(content) > self.chunk_threshold:
                    chunks = self._chunk_text(
                        content, chunk_size=self.chunk_size, overlap=self.overlap
                    )
                else:
                    chunks = [content]

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

    def _read_file(self, file_path: str) -> str:
        """Read content from a file (supports .txt, .md, .pdf)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def get_relevant_context(self, message: str, n_results: int = 3) -> list[str]:
        """Query the vector store to get context relevant to the message."""
        logger.debug("Querying vector store for: %s (n_results=%d)", message, n_results)
        return self.vector_store.query(message, n_results=n_results)

    def chat(self, session_id: str, message: str) -> dict:
        """Handle chat: retrieve context, manage history, call LLM, return response."""
        # 0. Rebuild index if data files changed
        if self._needs_rebuild():
            logger.info("Data changes detected, rebuilding index...")
            self._rebuild_index()

        # 1. Retrieve relevant context
        context = self.get_relevant_context(message)
        context_str = "\n---\n".join(context) if context else "No relevant context found."
        logger.debug("Retrieved %d context chunks", len(context))

        # 2. Manage conversation history
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            logger.debug("Created new session: %s", session_id)
        history = self.sessions[session_id]

        # 3. Build messages list
        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
                "If the context does not contain the answer, say so. Keep responses concise.\n\n"
                f"Context:\n{context_str}\n\n---\n\nKey Context:\n{self.context}\n\n---\n\nCharacter:\n{self.character}"
            ),
        }
        messages = [system_msg] + history + [{"role": "user", "content": message}]

        # 4. Call LLM
        if not self.llm_api_key:
            assistant_message = "LLM not configured: set OPENROUTER_API_KEY environment variable or provide llm.api_key in config."
            logger.warning("LLM API key not configured; returning placeholder message")
        else:
            try:
                logger.debug("Calling LLM model: %s", self.llm_model)
                response = litellm.completion(
                    model=self.llm_model,
                    messages=messages,
                    api_key=self.llm_api_key,
                    api_base=self.llm_api_base,
                )
                assistant_message = response.choices[0].message.content
                logger.info("LLM call successful")
            except Exception as e:
                logger.exception("Error calling LLM")
                assistant_message = f"Error calling LLM: {e}"

        # 5. Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": assistant_message})

        return {"message": assistant_message, "context": context}


if __name__ == "__main__":
    print("Starting RAG Wrapper example...")

    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
    with open("test_docs/doc1.txt", "w") as f:
        f.write("The first rule of Fight Club is: you do not talk about Fight Club.")
    with open("test_docs/doc2.txt", "w") as f:
        f.write("The sky is blue and the grass is green. The sun is a star.")

    config = get_config()
    rag_wrapper = RAGWrapper(config, files="test_docs")

    session_id = "test_session_123"

    print("\n--- Query 1 ---")
    response1 = rag_wrapper.chat(session_id, "What is the primary rule of the club?")
    print("\nWrapper Response:", response1)

    print("\n--- Query 2 ---")
    response2 = rag_wrapper.chat(session_id, "What color is the sky?")
    print("\nWrapper Response:", response2)
