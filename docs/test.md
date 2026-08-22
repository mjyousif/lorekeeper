# Testing Strategy

LoreKeeper uses `pytest` as its primary testing framework.

## Test Layout

Tests are located in the `tests/` directory and mirror the structure of the `src/` directory.

*   **Unit Tests**: Focus on isolated components such as `test_config.py`, `test_vector_store.py`, and `test_auth_storage.py`.
*   **Integration Tests**: The `test_wrapper.py` file validates the integration between the RAG system, LLM invocation, and session management.
*   **API Tests**: `test_api.py` ensures the FastAPI endpoints function correctly and maintain OpenAI compliance.
*   **Bot Tests**: Files like `test_mention.py`, `test_group.py`, and `test_is_mention.py` validate the parsing and routing logic for the Telegram bot interface.

## Running Tests

To run the entire test suite, ensure your virtual environment is active and run:

```shell
pytest
```

To run tests with output or for a specific file:
```shell
pytest tests/test_wrapper.py -v
```

## Testing Philosophy

*   **Mock External APIs**: When testing LLM calls, use mocks or `litellm`'s test mode to avoid hitting live APIs or incurring costs during CI/CD.
*   **Vector Store Isolation**: Tests involving the vector store should use temporary directories or in-memory representations to prevent cross-test pollution.

## Code Quality & Linting

In addition to testing, LoreKeeper uses the following tools to maintain code quality:

*   **Black**: Code formatting.
*   **Ruff**: Fast linting.
*   **Mypy**: Static type checking.
*   **Bandit**: Security vulnerability scanning.

To run these tools:
```shell
uv run black src tests
uv run ruff check src tests
uv run mypy src tests
uv run bandit -c pyproject.toml -r src
```
