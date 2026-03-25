import time
import uuid
import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional

from src.wrapper import LoreKeeper
from src.config import Config, get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Pydantic Models for OpenAI Compatibility ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]


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
    context: Optional[List[str]] = None


# --- Dependency Factories ---

ConfigDep = Annotated[Config, Depends(get_config)]


@lru_cache()
def get_lorekeeper() -> LoreKeeper:
    config = get_config()
    logger.info("Initializing LoreKeeper...")
    wrapper = LoreKeeper(config)
    logger.info("LoreKeeper initialization complete.")
    return wrapper


RAGDep = Annotated[LoreKeeper, Depends(get_lorekeeper)]


# --- FastAPI Application ---

app = FastAPI()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    rag: RAGDep,
):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    user_message = request.messages[-1].content
    logger.debug("Received chat request with message: %s", user_message[:100])

    # For now, use a placeholder session. In production, derive from auth/headers.
    session_id = "api_session_placeholder"

    try:
        wrapper_response = rag.chat(session_id=session_id, message=user_message)
    except Exception as e:
        logger.exception("Error in LoreKeeper")
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

    logger.info("Successfully processed chat request")
    return response


@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {
        "message": "LoreKeeper API is running. POST to /v1/chat/completions to interact."
    }
