import logging
import time
import uuid
from functools import lru_cache
from typing import Annotated, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.config import Config, get_config
from src.core.wrapper import LoreKeeper

config = get_config()
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Pydantic Models for OpenAI Compatibility ---


class ChatMessage(BaseModel):
    """A message within a chat conversation."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""

    model: str
    messages: List[ChatMessage]


class ChatCompletionResponseChoice(BaseModel):
    """Choice in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Usage statistics for the chat completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage = Field(default_factory=Usage)
    context: Optional[List[str]] = None


# --- Dependency Factories ---

ConfigDep = Annotated[Config, Depends(get_config)]


@lru_cache()
def get_lorekeeper() -> LoreKeeper:
    """Dependency provider for LoreKeeper."""
    config = get_config()
    logger.info("Initializing LoreKeeper for API...")
    start = time.perf_counter()
    wrapper = LoreKeeper(config)
    elapsed = time.perf_counter() - start
    logger.info("LoreKeeper initialization complete (took %.2fs)", elapsed)
    return wrapper


RAGDep = Annotated[LoreKeeper, Depends(get_lorekeeper)]


# --- FastAPI Application ---

app = FastAPI()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    rag: RAGDep,
):
    """Handle chat completion requests."""
    if not request.messages:
        logger.warning("Empty messages list in chat request")
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    user_message = request.messages[-1].content
    logger.info(
        "API chat request: model=%s messages=%d last_msg=%d chars",
        request.model,
        len(request.messages),
        len(user_message),
    )
    logger.debug("User message: %s", user_message[:200])

    # Convert Pydantic models to standard dictionaries
    message_dicts = [
        {"role": msg.role, "content": msg.content} for msg in request.messages
    ]

    try:
        request_start = time.perf_counter()
        wrapper_response = rag.chat_stateless(messages=message_dicts)
        request_elapsed = time.perf_counter() - request_start
    except Exception as e:
        logger.exception("Error in LoreKeeper during API request")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}") from e

    llm_message = wrapper_response.get("message", "No response from wrapper.")
    retrieved_context = wrapper_response.get("context", [])

    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=llm_message),
    )
    response = ChatCompletionResponse(
        model=request.model,
        choices=[choice],
        context=retrieved_context if retrieved_context else None,
    )

    logger.info(
        "API request completed in %.2fs: response=%d chars, context_chunks=%d",
        request_elapsed,
        len(llm_message),
        len(retrieved_context),
    )
    return response


@app.get("/")
def read_root():
    """Health check endpoint."""
    logger.info("Health check endpoint called")
    return {
        "message": (
            "LoreKeeper API is running. POST to /v1/chat/completions to interact."
        )
    }
