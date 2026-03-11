import os
from pypdf import PdfReader
import uuid
import litellm
from .vector_store import VectorStore, ChromaVectorStore


class RAGWrapper:
    def __init__(
        self,
        files: list[str] | str,
        db_path: str = "db",
        vector_store: VectorStore | None = None,
        llm_model: str = None,
        llm_api_key: str = None,
        llm_api_base: str = None,
    ):
        # support passing a single directory or list of files/dirs; build a
        # concrete list of files to embed
        self.files = self._resolve_files(files)
        self.db_path = db_path

        # Initialize vector store (use provided or create default Chroma one)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = ChromaVectorStore(db_path=db_path)

        self._load_and_embed_files()

        # LLM configuration: prefer explicit args, else environment
        self.llm_model = llm_model or os.getenv(
            "OPENROUTER_MODEL", "openrouter/stepfun/step-3.5-flash:free"
        )
        self.llm_api_key = (
            llm_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
        )
        self.llm_api_base = (
            llm_api_base
            or os.getenv("OPENROUTER_API_BASE")
            or os.getenv("LLM_API_BASE")
        )

        # Load context and character files from environment variables
        self.context = ""
        self.character = ""
        context_file = os.getenv("CONTEXT_FILE")
        character_file = os.getenv("CHARACTER_FILE")
        if context_file:
            try:
                with open(context_file, "r", encoding="utf-8") as f:
                    self.context = f.read()
            except Exception as e:
                print(f"Warning: could not load context file: {e}")
        if character_file:
            try:
                with open(character_file, "r", encoding="utf-8") as f:
                    self.character = f.read()
            except Exception as e:
                print(f"Warning: could not load character file: {e}")

        # Track data file manifest for auto‑rebuild
        self._file_spec = files  # keep original spec to re‑discover files later
        self._manifest: dict[str, tuple[float, int]] = {}  # path → (mtime, size)
        self.sessions: dict[str, list[dict]] = {}  # session_id → message history
        # After initial embedding, record the current state
        self._manifest = self._scan_files()

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
                print(f"Warning: path does not exist or is not a file/dir: {path}")

        return results

    def _scan_files(self) -> dict[str, tuple[float, int]]:
        """Scan the current set of data files (resolving the original spec) and return a dict of path → (mtime, size)."""
        manifest: dict[str, tuple[float, int]] = {}
        current_files = self._resolve_files(self._file_spec)
        for path in current_files:
            try:
                stat = os.stat(path)
                manifest[path] = (stat.st_mtime, stat.st_size)
            except Exception as e:
                print(f"Warning: cannot stat {path}: {e}")
        return manifest

    def _needs_rebuild(self) -> bool:
        """Compare current files against the stored manifest."""
        current = self._scan_files()
        # Simple comparison: if set of paths or their (mtime,size) differ => rebuild
        if set(current.keys()) != set(self._manifest.keys()):
            return True
        for path, info in current.items():
            if info != self._manifest.get(path):
                return True
        return False

    def _rebuild_index(self):
        """Delete the collection and re‑embed all files from scratch."""
        print("Data changes detected. Rebuilding index...")
        self.vector_store.clear()
        # Refresh the file list to include any new/removed files
        self.files = self._resolve_files(self._file_spec)
        self._load_and_embed_files(force=True)
        self._manifest = self._scan_files()
        print("Index rebuild complete.")

    def _load_and_embed_files(self, force: bool = False):
        """Loads files, chunks them, and stores them in the vector DB.

        Args:
            force: If True, embed regardless of whether the collection is non‑empty.
        """
        if not force and self.vector_store.count() > 0:
            print("Collection already contains documents. Skipping embedding process.")
            return

        print("Loading and embedding files...")
        for file_path in self.files:
            try:
                content = self._read_file(file_path)
                # Only chunk files large
                if len(content) > 10000:
                    chunks = self._chunk_text(content)
                else:
                    chunks = [content]  # Store small files as single chunk

                # Create unique IDs for each chunk
                ids = [str(uuid.uuid4()) for _ in chunks]

                self.vector_store.insert(
                    documents=chunks,
                    metadatas=[{"source": file_path} for _ in chunks],
                    ids=ids,
                )
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        print("Finished loading files.")

    def _read_file(self, file_path: str) -> str:
        """Reads content from a file (supports .txt, .md, .pdf)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:  # Assumes .txt, .md, or other plain text formats
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    def _chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """Splits text into overlapping chunks."""
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
        """Queries the vector store to get context relevant to the message."""
        return self.vector_store.query(message, n_results=n_results)

    def chat(self, session_id: str, message: str) -> dict:
        """
        Handles the chat logic: retrieves context, manages history,
        calls the LLM via LiteLLM, and returns the response.
        """
        # 0. Rebuild index if underlying data files have changed
        if self._needs_rebuild():
            self._rebuild_index()

        # 1. Retrieve relevant context
        context = self.get_relevant_context(message)
        context_str = (
            "\n---\n".join(context) if context else "No relevant context found."
        )

        # 2. Manage conversation history
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        history = self.sessions[session_id]

        # 3. Build LiteLLM messages list
        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
                "If the context does not contain the answer, say so. Keep responses concise.\n\n"
                f"Context:\n{context_str}\n\n---\n\nKey Context:\n{self.context}\n\n---\n\nCharacter:\n{self.character}"
            ),
        }
        messages = [system_msg] + history + [{"role": "user", "content": message}]

        # 4. Call LLM if configured; otherwise return placeholder
        if not self.llm_api_key:
            assistant_message = "LLM not configured: set OPENROUTER_API_KEY environment variable or pass llm_api_key to RAGWrapper."
        else:
            try:
                response = litellm.completion(
                    model=self.llm_model,
                    messages=messages,
                    api_key=self.llm_api_key,
                    api_base=self.llm_api_base,
                )
                assistant_message = response.choices[0].message.content
            except Exception as e:
                assistant_message = f"Error calling LLM: {e}"

        # 5. Update history (store both user and assistant)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": assistant_message})

        # Optional: trim history to last N exchanges to avoid excessive token usage
        # e.g., keep last 10 user+assistant pairs (20 messages)
        # if len(history) > 20: self.sessions[session_id] = history[-20:]

        return {"message": assistant_message, "context": context}


if __name__ == "__main__":
    # Example Usage
    print("Starting RAG Wrapper example...")

    # Create dummy files for testing
    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
    with open("test_docs/doc1.txt", "w") as f:
        f.write("The first rule of Fight Club is: you do not talk about Fight Club.")
    with open("test_docs/doc2.txt", "w") as f:
        f.write("The sky is blue and the grass is green. The sun is a star.")

    # show that passing the directory will automatically pick up both files
    rag_wrapper = RAGWrapper(files="test_docs", db_path="local_db")

    # --- Test Chat ---
    session_id = "test_session_123"

    print("\n--- Query 1 ---")
    message1 = "What is the primary rule of the club?"
    response1 = rag_wrapper.chat(session_id, message1)
    print("\nWrapper Response:", response1)

    print("\n--- Query 2 ---")
    message2 = "What color is the sky?"
    response2 = rag_wrapper.chat(session_id, message2)
    print("\nWrapper Response:", response2)
