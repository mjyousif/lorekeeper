import time
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

from rag_wrapper.wrapper import RAGWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load .env if present (e.g., for OPENROUTER_API_KEY)
load_dotenv()

# --- Pydantic Models for OpenAI Compatibility ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    # We'll ignore other parameters like temperature, stream, etc. for now


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage = Field(default_factory=Usage)


# --- FastAPI Application ---

app = FastAPI()

# Initialize the RAG Wrapper
logger.info("Initializing RAG Wrapper for API...")
rag_wrapper = RAGWrapper(
    files="data", db_path="api_db"  # point at the folder, not individual files
)
logger.info("Initialization complete.")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Accepts full conversation history and returns the next assistant message,
    with RAG context automatically injected.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # Convert Pydantic messages to simple dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Find the last user message for context retrieval
    last_user_msg = None
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
    if last_user_msg is None:
        raise HTTPException(status_code=400, detail="No user message found in conversation.")

    logger.debug("Received chat request with last user message: %s", last_user_msg[:100])

    # Retrieve relevant context from the RAG wrapper
    try:
        context_chunks = rag_wrapper.get_relevant_context(last_user_msg)
        context_str = "\n---\n".join(context_chunks) if context_chunks else "No relevant context found."
        logger.debug("Retrieved %d context chunks", len(context_chunks))
    except Exception as e:
        logger.exception("Error retrieving context")
        raise HTTPException(status_code=500, detail=f"Context retrieval error: {str(e)}") from e

    # Build system message with context, additional context from file, and character
    system_content = (
        "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
        "If the context does not contain the answer, say so. Keep responses concise.\n\n"
        f"Context:\n{context_str}\n\n"
    )
    if rag_wrapper.context:
        system_content += f"Key Context:\n{rag_wrapper.context}\n\n"
    if rag_wrapper.character:
        system_content += f"Character:\n{rag_wrapper.character}\n"

    # Prepend the system message to the conversation
    full_messages = [{"role": "system", "content": system_content.strip()}] + messages

    # Call the LLM
    try:
        logger.debug("Calling LLM model: %s", rag_wrapper.llm_model)
        response = litellm.completion(
            model=rag_wrapper.llm_model,
            messages=full_messages,
            api_key=rag_wrapper.llm_api_key,
            api_base=rag_wrapper.llm_api_base,
        )
        assistant_message = response.choices[0].message.content
        logger.info("LLM call successful")
    except Exception as e:
        logger.exception("Error calling LLM")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}") from e

    # Build OpenAI-compatible response
    assistant_msg_obj = ChatMessage(role="assistant", content=assistant_message)
    choice = ChatCompletionResponseChoice(index=0, message=assistant_msg_obj, finish_reason="stop")
    response_obj = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=rag_wrapper.llm_model,
        choices=[choice],
        usage=Usage(),
    )
    logger.info("Successfully processed chat request")
    return response_obj


@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {
        "message": "RAG Wrapper API is running. POST to /v1/chat/completions to interact."
    }
