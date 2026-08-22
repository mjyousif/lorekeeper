# API Reference

LoreKeeper exposes an OpenAI-compliant REST API using FastAPI. This allows you to easily connect standard frontend UI frameworks (like Chatbot-UI, LobeChat, or TypingMind) to LoreKeeper as if it were the native OpenAI API, while transparently receiving the benefits of LoreKeeper's RAG and Character persona injections.

## Base URL
When running the FastAPI server, the API is available at:
```text
http://localhost:8000
```

## `POST /v1/chat/completions`

Generates a response for a given chat conversation. The request and response schemas mirror OpenAI's API.

### Request Body

```json
{
  "model": "gpt-4o", 
  "messages": [
    {
      "role": "user",
      "content": "What happened during the First Age?"
    }
  ]
}
```

*Note: The `model` parameter is required by the schema, but LoreKeeper will use the underlying model defined in your `config.yaml` or `.env` to actually process the request.*

### Response Body

Returns a standard chat completion object, with an optional custom `context` array containing the specific chunks of text retrieved from the Lore vector database.

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1693000000,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "During the First Age, the great awakening occurred..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 50,
    "total_tokens": 200
  },
  "context": [
    "Retrieved lore chunk 1...",
    "Retrieved lore chunk 2..."
  ]
}
```

### Authentication
Currently, the FastAPI endpoints are exposed without forced Bearer token authentication, designed primarily for local networking and Docker-composed frontends. Ensure the API is placed behind a secure gateway or reverse proxy if exposing it to the public internet.
