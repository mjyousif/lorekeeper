# End-User Features

This document outlines the current and upcoming capabilities of the LoreKeeper application.

## Current Features

*   **Deep Lore Knowledge (RAG)**: Users can chat with a knowledge base built from local files. The system automatically retrieves the most relevant lore to inform the LLM's response.
*   **In-Character Persona**: The assistant can be configured with a strict system prompt (`character`) to ensure it always responds within a specific persona or character constraints.
*   **Persistent Sessions**: Chat history is tracked and managed across multiple turns using session IDs.
*   **Model Agnosticism**: Powered by `litellm`, users can configure the app to use OpenAI, Anthropic (via OpenRouter), or local models (via Ollama).
*   **OpenAI-Compliant API**: A `/v1/chat/completions` endpoint allows drop-in replacement for applications expecting an OpenAI backend.
*   **Multi-Platform Access**: Chat through a web UI (Gradio) or directly via a Telegram bot.

## Planned / Upcoming Features

*   **Agentic Tool Use**: The bot will soon be able to execute tools on behalf of the user or its own volition (e.g., fetching live data or performing actions).
*   **Image Generation**: Integration with an image generation API (like DALL-E or Stable Diffusion) to allow the character to "generate" or "share" images during the conversation.
