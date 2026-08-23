# End-User Features

This document outlines the current and upcoming capabilities of the LoreKeeper application.

## Current Features

*   **Deep Lore Knowledge (RAG via Tools)**: The system indexes a knowledge base from local files. The LLM can autonomously invoke the `memory_search` tool to fetch relevant lore when it needs more context to inform its response.
*   **In-Character Persona**: The assistant can be configured with a strict system prompt (`character`) to ensure it always responds within a specific persona or character constraints.
*   **Persistent Sessions**: Chat history is tracked and managed across multiple turns using session IDs.
*   **Model Agnosticism**: Powered by `litellm`, users can configure the app to use OpenAI, Anthropic (via OpenRouter), or local models (via Ollama).
*   **OpenAI-Compliant API**: A `/v1/chat/completions` endpoint allows drop-in replacement for applications expecting an OpenAI backend.
*   **Multi-Platform Access**: Chat through a web UI (Gradio) or directly via a Telegram bot.
*   **Text-to-Speech (TTS)**: The Telegram bot can generate and send voice messages using Google TTS (`gTTS`). This can be toggled by authorized users using the `/tts on` and `/tts off` commands.
*   **Agentic Tool Use**: The bot executes tools on behalf of the user or of its own volition via an internal execution loop.
*   **Image Generation (Pluggable Providers)**: Integration with image generation APIs to allow the character to "generate" or "share" images during the conversation. Supports Google (Imagen 3) and local ComfyUI via a pluggable architecture.
*   **Music Generation (Pluggable Providers)**: Integration with music/audio generation APIs. Supports Google and local ComfyUI via a pluggable architecture.

## Planned / Upcoming Features

*   **Expanded Tool Ecosystem**: Additional tool integrations and web searching capabilities.
