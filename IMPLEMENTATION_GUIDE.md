# Vector Storage Abstraction - Implementation Guide

## Overview

The RAG wrapper now uses a clean abstraction layer for vector storage. The vector database implementation is a **black box** to the rest of the application, making it easy to swap backends (Chroma, FAISS, Qdrant, etc.) without changing wrapper logic.

## Architecture

### VectorStore (Abstract Base Class)

Located in `rag_wrapper/vector_store.py`:

```python
class VectorStore(ABC):
    def __init__(self, db_path: str | Path, embedding_model: Any = None): ...
    def insert(self, documents: list[str], metadatas: Optional[list[dict]] = None, ids: Optional[list[str]] = None) -> None: ...
    def query(self, query_text: str, n_results: int = 3) -> list[str]: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...
```

### ChromaVectorStore (Default Implementation)

The current implementation uses ChromaDB with SentenceTransformer embeddings.

- `__init__(db_path, embedding_model=None, collection_name="rag_collection")`
- Automatically wraps SentenceTransformer in a Chroma embedding function
- Creates persistent client at `db_path`
- Raises `ValueError` if metadata dicts are empty (Chroma requirement)

### RAGWrapper Refactoring

**Before:** Direct ChromaDB usage (tightly coupled)
**After:** Accepts `vector_store` parameter (dependency injection)

```python
wrapper = RAGWrapper(
    files="data",
    db_path="db",
    vector_store=ChromaVectorStore(db_path="db")  # optional, created automatically if None
)
```

**Key changes:**
- Removed: `import chromadb`, `import SentenceTransformer` from `wrapper.py`
- Uses: `self.vector_store.insert()`, `self.vector_store.query()`, `self.vector_store.count()`, `self.vector_store.clear()`
- All file loading, chunking, manifest tracking, session management remain in wrapper

## Usage Examples

### Default (Chroma backend)

```python
from rag_wrapper.wrapper import RAGWrapper

wrapper = RAGWrapper(files="data", db_path="my_db")
# Automatically uses ChromaVectorStore
```

### Custom VectorStore Implementation

```python
from rag_wrapper.vector_store import VectorStore

class MyCustomStore(VectorStore):
    def __init__(self, db_path, embedding_model=None):
        # Initialize your backend
        pass

    def insert(self, documents, metadatas=None, ids=None):
        # Your logic
        pass

    def query(self, query_text, n_results=3):
        # Your logic
        return [...]
    def clear(self):
        pass
    def count(self):
        return 0

custom_store = MyCustomStore(db_path="custom_db")
wrapper = RAGWrapper(files="data", vector_store=custom_store)
```

## Testing

### Running Tests

```bash
cd /home/mjyousif/.openclaw/workspace/src/rag-wrapper
./run_tests.sh
# or: pytest tests/ -v
```

**Test coverage:** 47 passing tests, 2 skipped.

### Test Files

- `tests/test_vector_store.py` - Tests for VectorStore interface and ChromaVectorStore
- `tests/test_wrapper.py` - Tests for RAGWrapper (file ops, chunking, manifest, chat, sessions)

Key test scenarios:
- Insert with/without metadata, ID generation
- Query returns relevant documents
- Clear and count operations
- File resolution (recursive, extension filtering)
- PDF, TXT, MD reading
- Chunking with overlap
- Manifest rebuild detection (new, modified, deleted files)
- Chat builds correct context and history
- LLM error handling
- API key configuration (direct, env var, missing)
- Environment variable isolation (`monkeypatch`)

## Gradio UI

A simple web UI for interactive testing is provided in `gradio_app.py`.

### Run the UI

```bash
# Install UI dependencies
pip install -r requirements-ui.txt

# Set data directory (optional, defaults to "data")
export RAG_DATA_DIR="/path/to/your/documents"

# Set LLM API key (OpenRouter example)
export OPENROUTER_API_KEY="sk-or-..."

# Launch
python gradio_app.py
```

Then open http://localhost:7860.

### Features

- Query input with adjustable `n_results` (1-10)
- Session ID management for multi-turn conversations
- Toggle to show/hide retrieved context
- Session history info display
- Rebuild index button (forces re-embedding on next query)
- Clear session button

The UI uses the same `RAGWrapper` class, so all configuration (vector store, LLM model) is consistent.

## PR Creation

### Using GitHub App Token (Recommended for this repo)

```bash
cd /home/mjyousif/.openclaw/workspace/src/rag-wrapper

TOKEN=$(gh token generate \
  --app-id 2912162 \
  --installation-id 111427031 \
  --key /home/mjyousif/.ssh/trovu-agent.private-key.pem | jq -r .token)

git remote set-url origin "https://x-access-token:${TOKEN}@github.com/mjyousif/rag-wrapper.git"

# Push branch
git push origin <branch-name>

# Create PR via web or gh
gh pr create \
  --base main \
  --head <branch-name> \
  --title "..." \
  --body "..."

git remote set-url origin https://github.com/mjyousif/rag-wrapper.git
```

### Manual PR Creation

1. Push the branch: `git push origin <branch-name>`
2. Go to GitHub repo → "Compare & pull request"
3. Base: `main`, Compare: `<branch-name>`
4. Fill in title and description
5. Create PR

## Troubleshooting

**ImportError: cannot import name 'VectorStore'**
- Ensure you're using the refactored branch (`vector-abstraction`)

**Tests fail with Chroma metadata errors**
- We fixed `vector_store.insert` to pass `None` for metadatas, not `{}`. Ensure you have the latest `vector_store.py`.

**Token expired when pushing**
- Regenerate GitHub App token using `gh token generate` with app-id, installation-id, and private key.

**Tests take long (1+ minute)**
- SentenceTransformer downloads model on first run (~100MB). Subsequent runs are faster. Cache is stored in `~/.cache/torch/sentence_transformers`.

## Next Steps

- Consider adding alternative vector store implementations (FAISS, Qdrant) as separate modules
- Add more edge case tests (empty files, unicode, large PDFs)
- Add benchmarking for different backends
- Document embedding model selection

---

**Created:** March 10, 2026  
**Branch:** `vector-abstraction`  
**Status:** Ready for PR