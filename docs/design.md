# Application Design

LoreKeeper is designed as a modular, lightweight wrapper around Large Language Models (LLMs) that enriches interactions with deep lore (via Retrieval-Augmented Generation) and character personas. 

## Core Architecture

The system follows a domain-driven structure separated into the following components (located in `src/`):

1.  **`core`**: Contains `wrapper.py` (the main `LoreKeeper` orchestrator) and `chat_manager.py` (handles direct interaction with `litellm`, token counting, and prompt injection).
2.  **`rag`**: Contains `vector_store.py`, `document_loader.py`, and `text_chunker.py` for creating embeddings, chunking text, and providing similarity search.
3.  **`storage`**: Contains `session_storage.py` and `auth_storage.py` to manage persistence for user sessions (chat history) and authentication.
4.  **`interfaces`**: User interaction endpoints.

## Interfaces

LoreKeeper exposes multiple interfaces for user interaction:
*   **FastAPI Backend (`src/interfaces/api.py`)**: An OpenAI-compliant `/v1/chat/completions` REST API endpoint.
*   **Telegram Bot (`src/interfaces/telegram_bot.py`)**: A bot integration allowing users to chat directly via Telegram.
*   **Gradio App (`src/interfaces/gradio_app.py`)**: A web-based UI for testing and chatting.
*   **CLI (`src/interfaces/cli.py`)**: A command line application.

## Future Agentic Design

The architecture is currently being evolved to support an **Agentic Loop**. 
*   `src/core/chat_manager.py` will be extended to support LLM Tool Calling (function calling).
*   When a tool call is requested by the model (e.g., "generate image"), the execution loop will pause, execute the tool locally, feed the result back to the LLM, and yield a final character-driven response.
