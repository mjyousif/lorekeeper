# Application Design

LoreKeeper is designed as a modular, lightweight wrapper around Large Language Models (LLMs) that enriches interactions with deep lore (via Retrieval-Augmented Generation) and character personas. 

## Core Architecture

The system is broken down into the following core components (located in `src/`):

1.  **`wrapper.py`**: The main entry point (`LoreKeeper`) that orchestrates the integration between the LLM, the vector store, and session history.
2.  **`chat_manager.py`**: Handles the direct interaction with `litellm`. It manages token counting, context window limits, and injects both the `character` persona and the retrieved `context` into the system prompt.
3.  **`vector_store.py` / `document_loader.py`**: Responsible for creating embeddings from local files, chunking text, and providing similarity search for the RAG pipeline.
4.  **`session_storage.py` / `auth_storage.py`**: Manages persistence for user sessions (chat history) and authentication.

## Interfaces

LoreKeeper exposes multiple interfaces for user interaction:
*   **FastAPI Backend (`api.py`)**: An OpenAI-compliant `/v1/chat/completions` REST API endpoint.
*   **Telegram Bot (`telegram_bot.py`)**: A bot integration allowing users to chat directly via Telegram.
*   **Gradio App (`gradio_app.py`)**: A web-based UI for testing and chatting.

## Future Agentic Design

The architecture is currently being evolved to support an **Agentic Loop**. 
*   `chat_manager.py` will be extended to support LLM Tool Calling (function calling).
*   When a tool call is requested by the model (e.g., "generate image"), the execution loop will pause, execute the tool locally, feed the result back to the LLM, and yield a final character-driven response.
