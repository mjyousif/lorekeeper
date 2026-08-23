# Configuration Guide

LoreKeeper is highly configurable via environment variables (`.env`) or a YAML/TOML configuration file (`config.yaml`).

The configuration is parsed by Pydantic, ensuring that all settings are strictly typed and validated upon startup.

## Configuration Loading Order

1. Environment variables (`.env` or system environment variables).
2. Configuration file (`config.yaml` or `config.toml`).

## Available Settings

Below is a list of all available settings you can tweak for LoreKeeper:

### General Settings
*   **`log_level`**: Controls the verbosity of application logs. (Default: `INFO`, Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`)
*   **`files`**: Directory or list of files to ingest for RAG embeddings. (Default: `"data"`)
*   **`db_path`**: The local directory path where SQLite and vector store files will be saved. (Default: `"db"`)

### Character and Context
*   **`character_file`**: Path to a markdown/text file containing the persona or system prompt instruction (e.g., `data/character.md`).
*   **`context_file`**: Path to a markdown/text file containing static context that is always injected into the system prompt regardless of RAG.

### Vector Store / RAG Options
*   **`chunk_size`**: The number of characters/tokens to chunk documents into before embedding. (Default: `1000`)
*   **`overlap`**: The number of characters that overlap between chunks to preserve context boundaries. (Default: `200`)
*   **`chunk_threshold`**: Size limit threshold for creating chunks. (Default: `10000`)
*   **`vector_store`**: Dictionary configuration for the vector store engine.

### LLM Settings (via `litellm`)
All LLM configuration falls under the `llm` dictionary (or environment variables prefixed with `LLM_`).
*   **`llm.model`**: The model string (e.g., `gpt-4o`, `openrouter/anthropic/claude-3-opus`, `ollama/llama3`).
*   **`llm.api_key`**: API key for cloud providers. (Leave blank or omit for local Ollama).
*   **`llm.api_base`**: Base URL, required when using local Ollama (e.g., `http://localhost:11434`) or custom OpenAI-compatible endpoints.

### TTS Settings
Text-to-Speech settings are configured under the `tts` dictionary.
*   **`tts.engine`**: The TTS engine to use. (Default: `gtts`, Options: `gtts`, `gemini`)
*   **`tts.gemini_api_key`**: API key for Google Gemini if using the `gemini` engine.

### Telegram Bot Settings
Settings for `telegram_bot.py`.
*   **`telegram.bot_token`**: Your Telegram bot token provided by BotFather.
*   **`telegram.session_db`**: SQLite database path for telegram chat sessions.
*   **`allowed_user_ids`**: A comma-separated string or list of integers containing the user IDs permitted to interact with the bot.
*   **`allowed_chat_ids`**: A comma-separated string or list of integers containing the chat/group IDs permitted to interact with the bot.

### Tools Settings
Agent tools are configured under the `tools_config` dictionary (or `tools_config` key in YAML). This allows you to enable optional tools and specify their pluggable providers.
*   **`tools_config.image_generation.provider`**: The provider to use for image generation. Options: `google`, `comfyui`.
*   **`tools_config.image_generation.api_key`**: API key for cloud providers (like Google).
*   **`tools_config.image_generation.model`**: Model name to use (e.g. `imagen-3.0-generate-002`).
*   **`tools_config.image_generation.url`**: URL for local providers like ComfyUI (e.g. `http://127.0.0.1:8188`).

*   **`tools_config.music_generation.provider`**: The provider to use for music generation. Options: `google`, `comfyui`.
*   **`tools_config.music_generation.api_key`**: API key for cloud providers (like Google).
*   **`tools_config.music_generation.model`**: Model name to use (e.g. `gemini-pro-audio`).
*   **`tools_config.music_generation.url`**: URL for local providers like ComfyUI.
