import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import uuid


class RAGWrapper:
    def __init__(self, files: list[str] | str, db_path: str = "db"):
        # support passing a single directory or list of files/dirs; build a
        # concrete list of files to embed
        self.files = self._resolve_files(files)
        self.db_path = db_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name="rag_collection")
        self._load_and_embed_files()

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
                for root, _, files in os.walk(path):
                    for fname in files:
                        if fname.lower().endswith(allowed):
                            results.append(os.path.join(root, fname))
            elif os.path.isfile(path):
                results.append(path)
            else:
                print(f"Warning: path does not exist or is not a file/dir: {path}")

        return results

    def _load_and_embed_files(self):
        """Loads files, chunks them, and stores them in the vector DB."""
        if self.collection.count() > 0:
            print("Collection already contains documents. Skipping embedding process.")
            # In a real application, you might want to check if the files have changed
            # and update the collection accordingly.
            return

        print("Loading and embedding files...")
        for file_path in self.files:
            try:
                content = self._read_file(file_path)
                chunks = self._chunk_text(content)

                # Create unique IDs for each chunk
                ids = [str(uuid.uuid4()) for _ in chunks]

                self.collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=[{"source": file_path} for _ in chunks],
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
        results = self.collection.query(query_texts=[message], n_results=n_results)
        return results["documents"][0] if results["documents"] else []

    def chat(self, session_id: str, message: str):
        """
        Handles the chat logic: retrieves context, gets LLM response.
        (LLM call is not implemented yet).
        """
        # 1. Retrieve relevant context from the vector DB
        context = self.get_relevant_context(message)

        # 2. (TODO) Manage conversation history for the session_id

        # 3. (TODO) Build the prompt for the LLM
        prompt = f"""
        Conversation History:
        [History will go here]

        Relevant Information from Documents:
        ---
        {" ".join(context)}
        ---
        
        User's Message:
        {message}
        
        Response:
        """

        # 4. (TODO) Call the LLM with the prompt and history
        # For now, we'll just return the prompt and context
        print(f"--- Generated Prompt for Session {session_id} ---")
        print(prompt)

        return {
            "message": "This is a placeholder response. LLM integration is not complete.",
            "context": context,
        }


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
