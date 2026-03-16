import os
import time
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import litellm
from rag_wrapper.wrapper import RAGWrapper
from rag_wrapper.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load .env if present (e.g., for OPENROUTER_API_KEY)
load_dotenv()

# Load configuration from config.yaml
CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", "config.yaml")
try:
    cfg = Config.from_file(CONFIG_PATH)
    DB_PATH = cfg.db_path
    FILES = cfg.files
    LLM_MODEL = cfg.llm.get("model", "openrouter/stepfun/step-3.5-flash:free")
except Exception as e:
    logging.warning(f"Failed to load config from {CONFIG_PATH}: {e}. Using defaults.")
    DB_PATH = "shared_db"
    FILES = "data"
    LLM_MODEL = "openrouter/stepfun/step-3.5-flash:free"

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
    # Custom field to expose retrieved context separately (non-OpenAI standard)
    context: Optional[List[str]] = None


# --- FastAPI Application ---

app = FastAPI()

# Initialize the RAG Wrapper
# In a real application, you might configure the files and db_path via
# environment variables or a config file.  The wrapper now accepts a
# directory and will scan it recursively for supported document types.
logger.info("Initializing RAG Wrapper for API...")
rag_wrapper = RAGWrapper(
    files=FILES,
    db_path=DB_PATH,
    llm_model=LLM_MODEL,
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

    # The wrapper's `chat` method returns a dict with the LLM message
    # and the retrieved context. We return a proper OpenAI-compliant response
    # with the pure message in choices, and include context as a separate field.
    llm_message = wrapper_response.get("message", "No response from wrapper.")
    retrieved_context = wrapper_response.get("context", [])

    # Create the response payload - only the LLM message in the choice
    assistant_message = ChatMessage(role="assistant", content=llm_message)
    choice = ChatCompletionResponseChoice(index=0, message=assistant_message)
    response = ChatCompletionResponse(
        model=request.model,
        choices=[choice],
        context=retrieved_context if retrieved_context else None,
    )

    logger.info("Successfully processed chat request")
    return response


@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {
        "message": "RAG Wrapper API is running. POST to /v1/chat/completions to interact."
    }
