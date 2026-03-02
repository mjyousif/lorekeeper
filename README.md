# LLM RAG Wrapper

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
from rag_wrapper import RAGWrapper

# 1. Initialize the wrapper with your files
file_paths = ["path/to/document1.txt", "path/to/document2.md"]
wrapper = RAGWrapper(files=file_paths)

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

## Running the API

The project includes a FastAPI server that exposes an OpenAI-compliant `/v1/chat/completions` endpoint.

### 1. Setup

First, create and activate a virtual environment. This keeps the project's dependencies isolated.

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
uvicorn rag_wrapper.api:app --reload
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
