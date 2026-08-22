# Deployment Guide

LoreKeeper is containerized using Docker, allowing for reproducible and sandboxed deployments. The provided `docker-compose.yml` supports both standard cloud LLM deployments and a fully local LLM deployment via Ollama.

## Standard Deployment (Cloud LLMs)

If you are using a cloud provider like OpenAI, Anthropic, or OpenRouter, you can deploy the core LoreKeeper services (Telegram Bot, API, and UI).

1. Ensure your `.env` is configured with your API keys (e.g., `LLM_MODEL=openrouter/...`, `LLM_API_KEY=sk-...`).
2. Run the compose stack:
   ```shell
   docker compose up -d
   ```

This will spin up three containers:
*   `lorekeeper-bot`: The Telegram bot integration.
*   `lorekeeper-api`: The FastAPI backend exposed on port `8000`.
*   `lorekeeper-ui`: The Gradio testing interface exposed on port `7860`.

### Volumes
The standard deployment mounts two volumes:
*   `./data:/app/data`: A bind mount to your local data folder so you can edit `character.md` or lore documents in real-time.
*   `lorekeeper-db`: A managed Docker volume storing the SQLite session database and vector store embeddings.

## Local Deployment (Ollama Profiling)

If you prefer maximum privacy and want to run the LLM locally on your own hardware, the `docker-compose.yml` includes an `ollama` profile.

This will launch:
1. The three LoreKeeper containers (`bot`, `api`, `ui`).
2. A local Ollama server.
3. A sidecar container (`ollama-pull-model`) that automatically downloads the model specified in your `.env`.

### Steps:
1. Configure your `.env` for local Ollama:
   ```env
   LLM_MODEL=ollama/llama3
   LLM_API_BASE=http://ollama:11434
   ```
2. Start the stack with the `local-llm` profile:
   ```shell
   docker compose --profile local-llm up -d
   ```

### Hardware Acceleration (GPUs)
By default, the Ollama container runs on the CPU. If you have an NVIDIA GPU, you can uncomment the `deploy` block inside `docker-compose.yml` under the `ollama` service to pass the GPU to the container:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```
*(Note: You must have the NVIDIA Container Toolkit installed on your host machine.)*
