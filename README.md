# RAG Wrapper

A simple, modular RAG (Retrieval-Augmented Generation) system with a clean vector storage abstraction, OpenAI-compatible API, Gradio UI, and Telegram bot integration.

## Features

- **Vector storage abstraction** – swap Chroma for FAISS, Qdrant, etc.
- **Configuration file support** – YAML, TOML, or JSON
- **OpenAI-compatible API** – `/v1/chat/completions` endpoint
- **Gradio web UI** – interactive testing
- **Telegram bot** – with user/chat authorization
- **Fast, reproducible installs** – uses `uv`

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip/venv

### Install and Run

```bash
# Clone and enter the repository
cd /path/to/rag-wrapper

# Install dependencies (uses uv)
uv sync --all-extras

# Start the API server (on 127.0.0.1:8000)
uv run uvicorn rag_wrapper.api:app --host 127.0.0.1 --port 8000

# In another terminal, start the Gradio UI
uv run python gradio_app.py

# Or start the Telegram bot (requires TELEGRAM_BOT_TOKEN)
uv run python telegram_bot.py
```

### Managing Services

Use the included `chatter` CLI:

```bash
./chatter start all       # Start API, UI, and Telegram bot
./chatter stop all        # Stop all services
./chatter logs            # Follow logs
./chatter status          # Show running services
```

## Configuration

Create a `config.yaml` (or `.toml` / `.json`) in the project root:

```yaml
files:
  - "data"                # Directory or list of files to embed
db_path: "db"             # Vector database directory

llm:
  model: "openrouter/stepfun/step-3.5-flash:free"

chunk_size: 1000
overlap: 200
log_level: "INFO"

# Optional: context and character files
context_file: "context.txt"
character_file: "character.txt"

# Telegram bot authorization (optional)
allowed_user_ids: []      # List of allowed Telegram user IDs
allowed_chat_ids: []      # List of allowed Telegram chat IDs
```

Then run with:

```bash
CONFIG_PATH=config.yaml uv run uvicorn rag_wrapper.api:app --host 127.0.0.1 --port 8000
```

Or set `CONFIG_PATH` in your `.env` file.

Environment variables also override config values (e.g., `OPENROUTER_API_KEY`).

## Telegram Bot Authorization

To restrict the bot to specific users or groups:

1. Get user/chat IDs (see [telegram.md](telegram.md))
2. Set either:
   - Environment variables: `ALLOWED_USER_IDS` and/or `ALLOWED_CHAT_IDS` (comma-separated integers)
   - Or include `allowed_user_ids` / `allowed_chat_ids` in your config file

**Secure default:** if neither is set, the bot denies all access.

## Testing

```bash
# Install test dependencies (if not using uv sync --all-extras)
uv pip install -r tests/requirements-test.txt

# Run tests
pytest tests/ -v
# or with coverage
pytest tests/ --cov=rag_wrapper --cov-report=html
```

Tests are also run automatically via GitHub Actions on PRs.

## Project Structure

- `rag_wrapper/` – core library
  - `wrapper.py` – main RAGWrapper class
  - `vector_store.py` – VectorStore abstraction + ChromaVectorStore
  - `config.py` – configuration loading
  - `api.py` – FastAPI server
- `telegram_bot.py` – Telegram bot integration
- `gradio_app.py` – Gradio web UI
- `chatter` – CLI for managing local services

## Development

- Use `uv` for dependency management
- Format code with `black` (`uv run black .`)
- Tests live in `tests/`
- Follow conventional commits for PRs

## License

MIT (or choose your license)

---

**Note:** The `data/` directory is not tracked. Place your `.txt`, `.md`, or `.pdf` files there and the wrapper will embed them on first run.