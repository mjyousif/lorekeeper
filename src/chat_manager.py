from __future__ import annotations

import logging
import litellm

logger = logging.getLogger(__name__)


class ChatManager:
    """Encapsulates LLM interaction and conversation history management."""

    def __init__(
        self,
        llm_model: str | None,
        llm_api_key: str | None,
        llm_api_base: str | None,
        max_context_size: int = 64000,
        context: str = "",
        character: str = "",
    ):
        """Initialize the ChatManager with LLM configuration.

        Args:
            llm_model: The LLM model to use (e.g., 'gpt-3.5-turbo').
            llm_api_key: The API key for the LLM service.
            llm_api_base: The base URL for the API (optional).
            max_context_size: The maximum allowed tokens in the context window.
            context: Static system context string.
            character: Static persona or character instruction string.
        """
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.max_context_size = max_context_size
        self.context = context
        self.character = character

    def generate_response(
        self,
        message: str,
        retrieved_context: list[str],
        history: list[dict],
    ) -> str:
        """Construct messages, enforce limits, and call the LLM.

        Args:
            message: The user's new message.
            retrieved_context: Relevant text chunks from vector store.
            history: Conversation history list containing role/content dicts.

        Returns:
            The assistant's generated message or an error/placeholder message.
        """
        context_str = (
            "\n---\n".join(retrieved_context)
            if retrieved_context
            else "No relevant context found."
        )

        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
                "If the context does not contain the answer, say so. Keep responses concise.\n\n"
                f"Context:\n{context_str}\n\n---\n\nKey Context:\n{self.context}\n\n---\n\nCharacter:\n{self.character}"
            ),
        }

        # Enforce max context size
        # We loop until the token count of the combined messages is under the limit, or history runs out.
        try:
            while len(history) > 0:
                messages = (
                    [system_msg] + history + [{"role": "user", "content": message}]
                )
                current_tokens = litellm.token_counter(
                    model=self.llm_model, messages=messages
                )
                if current_tokens <= self.max_context_size:
                    break
                # If too large, remove the oldest message in history
                history.pop(0)
        except Exception as e:
            logger.warning("Failed to count tokens or truncate history: %s", e)

        messages = [system_msg] + history + [{"role": "user", "content": message}]

        # Call LLM
        if not self.llm_api_key:
            logger.warning("LLM API key not configured; returning placeholder message")
            return "LLM not configured: set OPENROUTER_API_KEY environment variable or provide llm.api_key in config."

        try:
            logger.debug("Calling LLM model: %s", self.llm_model)
            response = litellm.completion(
                model=self.llm_model,
                messages=messages,
                api_key=self.llm_api_key,
                api_base=self.llm_api_base,
            )
            logger.info("LLM call successful")
            return response.choices[0].message.content
        except Exception as e:
            logger.exception("Error calling LLM")

            # Truncate the error message to avoid polluting output with massive HTML pages
            error_str = str(e)
            if len(error_str) > 1000:
                error_str = error_str[:1000] + "... [truncated]"
            return f"Error calling LLM: {error_str}"
