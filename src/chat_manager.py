from __future__ import annotations

import logging
import time

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
        logger.info(
            "ChatManager initialized: model=%s api_base=%s max_context=%d context_len=%d character_len=%d",
            self.llm_model, self.llm_api_base, self.max_context_size,
            len(self.context), len(self.character),
        )

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
        logger.debug(
            "Building prompt: %d retrieved chunks, context_str=%d chars",
            len(retrieved_context), len(context_str),
        )

        model_name = self.llm_model or "gpt-3.5-turbo"
        try:
            char_tokens = litellm.token_counter(model=model_name, text=self.character) if self.character else 0
            context_file_tokens = litellm.token_counter(model=model_name, text=self.context) if self.context else 0
            retrieved_context_tokens = litellm.token_counter(model=model_name, text=context_str) if context_str else 0
            history_tokens = litellm.token_counter(model=model_name, messages=history) if history else 0
            user_message_tokens = litellm.token_counter(model=model_name, text=message) if message else 0
            logger.debug(
                "Token counts (pre-trimming): character=%d, context_files=%d, retrieved_context=%d, history=%d, user_message=%d",
                char_tokens,
                context_file_tokens,
                retrieved_context_tokens,
                history_tokens,
                user_message_tokens,
            )
        except Exception as e:
            logger.warning("Failed to count tokens for prompt components: %s", e)

        system_msg = {
            "role": "system",
            "content": (
                f"Character:\n{self.character}\n\n---\n\n"
                f"Key Context:\n{self.context}\n\n---\n\n"
                "You must fully embody the Character described above. "
                "Use the following retrieved context to answer the user's question. "
                "If the context does not contain the answer, do not guess or make up information. "
                "Simply state that you do not know, while remaining in character. "
                "CRITICAL: Your response MUST be under 3 sentences. Be extremely brief.\n\n"
                f"Retrieved Context:\n{context_str}"
            ),
        }
        logger.debug("System prompt length: %d chars", len(system_msg["content"]))

        # Enforce max context size
        # We loop until the token count of the combined messages is under the limit, or history runs out.
        try:
            original_history_len = len(history)
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
            trimmed = original_history_len - len(history)
            if trimmed > 0:
                logger.info(
                    "Trimmed %d messages from history to fit context window (was %d tokens)",
                    trimmed, current_tokens if 'current_tokens' in dir() else -1,
                )
            logger.debug(
                "Final message count: %d (system + %d history + user)",
                len(history) + 2, len(history),
            )
        except Exception as e:
            logger.warning("Failed to count tokens or truncate history: %s", e)

        messages = [system_msg] + history + [{"role": "user", "content": message}]

        # Call LLM
        if not self.llm_api_key:
            logger.warning("LLM API key not configured; returning placeholder message")
            return "LLM not configured: set OPENROUTER_API_KEY environment variable or provide llm.api_key in config."

        try:
            logger.info("Calling LLM: model=%s api_base=%s", self.llm_model, self.llm_api_base)
            llm_start = time.perf_counter()
            response = litellm.completion(
                model=self.llm_model,
                messages=messages,
                api_key=self.llm_api_key,
                api_base=self.llm_api_base,
            )
            llm_elapsed = time.perf_counter() - llm_start
            reply = response.choices[0].message.content
            usage = getattr(response, 'usage', None)
            logger.info(
                "LLM call successful in %.2fs — response=%d chars, usage=%s",
                llm_elapsed, len(reply) if reply else 0,
                {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens, "total": usage.total_tokens} if usage else "N/A",
            )
            return reply
        except Exception as e:
            logger.exception("Error calling LLM (model=%s)", self.llm_model)

            # Truncate the error message to avoid polluting output with massive HTML pages
            error_str = str(e)
            if len(error_str) > 1000:
                error_str = error_str[:1000] + "... [truncated]"
            return f"Error calling LLM: {error_str}"
