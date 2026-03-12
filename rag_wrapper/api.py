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
# In a real application, you might configure the files and db_path via
# environment variables or a config file.  The wrapper now accepts a
# directory and will scan it recursively for supported document types.
logger.info("Initializing RAG Wrapper for API...")
rag_wrapper = RAGWrapper(
    files="data", db_path="api_db"  # point at the folder, not individual files
)
logger.info("Initialization complete.")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compliant chat completions endpoint.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # Extract the last user message
    user_message = request.messages[-1].content
    logger.debug("Received chat request with message: %s", user_message[:100])

    # For now, we'll use a dummy session_id. In a real app, this might be
    # managed via headers, API keys, or another mechanism.
    session_id = "api_session_placeholder"

    try:
        # Use the RAG wrapper to get context and a placeholder response.
        wrapper_response = rag_wrapper.chat(session_id=session_id, message=user_message)
    except Exception as e:
        logger.exception("Error in RAG wrapper")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}") from e

    # The wrapper's `chat` method returns a dict with a placeholder message
    # and the retrieved context. We'll format this into an OpenAI-style response.
    context_str = "\n\n--- Retrieved Context ---\n" + "\n".join(
        wrapper_response.get("context", [])
    )

    assistant_message_content = (
        wrapper_response.get("message", "No response from wrapper.") + context_str
    )

    # Create the response payload
    assistant_message = ChatMessage(role="assistant", content=assistant_message_content)
    choice = ChatCompletionResponseChoice(index=0, message=assistant_message)
    response = ChatCompletionResponse(model=request.model, choices=[choice])

    logger.info("Successfully processed chat request")
    return response


@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {
        "message": "RAG Wrapper API is running. POST to /v1/chat/completions to interact."
    }
