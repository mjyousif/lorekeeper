# LLM LoreKeeper

This project is a wrapper around Large Language Model (LLM) calls that uses Retrieval-Augmented Generation (RAG) to provide context from a local set of files.

## How it Works

1. **Initialization**: You initialize the wrapper by providing a set of files.
2. **Embedding Creation**: The wrapper creates vector embeddings from the content of these files.
3. **Local Storage**: These embeddings are stored locally for efficient retrieval.
4. **Session-based Interaction**: When you make a call, you provide a `session_id` and a `message`.
5. **Contextual Augmentation**: The wrapper retrieves relevant information from the stored embeddings based on your message.
6. **LLM Call with History**: The original message, retrieved context, and the conversation history (tracked via `session_id`) are all sent to the LLM to generate a comprehensive response.

## Features

- **RAG from local files**: Automatically creates and manages a knowledge base from your documents.
- **Session Management**: Maintains conversation history for coherent, multi-turn dialogues.
- **Simple Interface**: Easy to integrate and use.

## Usage Example

```python
from src.core.wrapper import LoreKeeper

# 1. Initialize the wrapper with your files
file_paths = ["path/to/document1.txt", "path/to/document2.md"]
wrapper = LoreKeeper(files=file_paths)

# 2. Start a conversation (or continue one)
session_id = "user123_session_abc"
user_message = "What is the main topic of the documents?"

# 3. Get the LLM's response
response = wrapper.chat(session_id=session_id, message=user_message)

print(response)

# Continue the conversation
user_message_2 = "Can you elaborate on the first point?"
response_2 = wrapper.chat(session_id=session_id, message=user_message_2)

print(response_2)
```

## Configuration Examples

The project uses `config.yaml` or `.env` to configure the LLM provider, as it is powered by `litellm`. Here are examples of how to configure different providers.

### Local Ollama

To use a local instance of [Ollama](https://ollama.com/), you need to prefix your model name with `ollama/` and specify the `api_base` pointing to your local Ollama server.

**In `config.yaml`:**
```yaml
llm:
  model: "ollama/llama3"
  api_base: "http://localhost:11434"
  # api_key is not required for local Ollama, but can be left empty
  api_key: ""
```

### OpenRouter

To use [OpenRouter](https://openrouter.ai/), prefix the model name with `openrouter/` and provide your OpenRouter API key.

**In `config.yaml`:**
```yaml
llm:
  model: "openrouter/anthropic/claude-3-opus"
  api_key: "sk-or-v1-..."
```

**Using environment variables (`.env`):**
Alternatively, you can set these using environment variables if your `config.yaml` references them like `${LLM_API_KEY}`.

```env
LLM_MODEL="openrouter/anthropic/claude-3-opus"
LLM_API_KEY="sk-or-v1-..."
```

## Development Setup

This project uses [black](https://github.com/psf/black) for code formatting. To format your code:

```shell
black .
```

## Running the API

The project includes a FastAPI server that exposes an OpenAI-compliant `/v1/chat/completions` endpoint.

### 1. Setup

First, create and activate a virtual environment. This keeps the project's dependencies isolated.

```shell
# Install development dependencies including black
pip install -r requirements.txt
```

```shell
# Create the virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# On macOS/Linux, you would use:
# source .venv/bin/activate
```

Next, install the required packages using pip.

```shell
pip install -r requirements.txt
```

### 2. Launch the Server

Once the dependencies are installed, start the API server with `uvicorn`.

```shell
uvicorn src.interfaces.api:app --reload
```

The server will be running at `http://127.0.0.1:8000`.

### 3. Interact with the Endpoint

You can send a POST request to the `/v1/chat/completions` endpoint using a tool like `curl`. The response will be a JSON object that mimics the OpenAI Chat Completions API format.

```shell
curl -X "POST" "http://127.0.0.1:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
         "model": "local-rag-model",
         "messages": [
             {
                 "role": "user",
                 "content": "What is ChromaDB?"
             }
         ]
     }'
```

## Running the Telegram Bot in Docker

You can run the Telegram bot either as a standalone container or alongside a bundled Ollama server using Docker Compose.

### Option A: Using Docker Compose (Bundled Ollama for Local LLM)

This is the recommended method to run LoreKeeper fully local/offline alongside a bundled Ollama LLM server.

#### 1. Configure the Environment
Uncomment the **Local Ollama Configuration (Bundled in Docker Compose)** section in your `.env` file, ensuring `COMPOSE_PROFILES` is set to `local-llm`:
```env
COMPOSE_PROFILES=local-llm
LLM_API_KEY=ollama
LLM_MODEL=ollama/llama3.2
LLM_API_BASE=http://ollama:11434
```
*Note: The `COMPOSE_PROFILES=local-llm` setting instructs Docker Compose to load the conditional `ollama` service. If it is omitted or commented out, `docker compose up` will only start the `lorekeeper` bot container (useful when connecting to external APIs).*

#### 2. Start the Services
Launch the stack in detached mode:
```shell
docker compose up -d
```
This builds your local LoreKeeper bot image and spins up the active services based on your profile.

#### 3. Automatic Model Pulling
The stack includes a companion helper service (`ollama-pull-model`) that automatically pulls the configured `LLM_MODEL` (e.g., `llama3.2`) once the Ollama server starts up and becomes healthy. You don't need to pull it manually.

*Note: If you ever want to manually pull a different model, you can run:*
```shell
docker exec -it ollama ollama pull <model-name>
```

#### 4. Authorizing the Bot (Pairing)
If `ALLOWED_USER_IDS` is not pre-configured in `.env`, the bot will deny access by default. To authorize yourself:
1. Search for your bot in Telegram and send `/pair` to generate a pairing code.
2. Approve the code inside the running container using `uv`:
   ```shell
   docker compose exec lorekeeper uv run python -m scripts.approve_pair <CODE>
   ```

#### 5. Optional: GPU Acceleration
To speed up local inference with an NVIDIA GPU, uncomment the `deploy` section under the `ollama` service in `docker-compose.yml`. (Note: This requires the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) to be installed on your host system).

---

### Option B: Standalone Docker Container

To run only the Telegram bot container connecting to an external API (like OpenRouter or a separately running local Ollama):

#### 1. Build the Docker Image

```shell
docker build -t rag-telegram-bot .
```

#### 2. Run the Container

```shell
docker run -d --name my-telegram-bot \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e TELEGRAM_BOT_TOKEN=your_token_here \
  -e LLM_API_KEY=your_llm_api_key_here \
  rag-telegram-bot
```

Make sure to mount any local directories needed for configuration or data storage.
