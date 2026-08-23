from __future__ import annotations

import logging
import time

import litellm

logger = logging.getLogger(__name__)


class ChatManager:
    """Encapsulates LLM interaction, conversation history management,
    and agentic tool loop.
    """

    def __init__(
        self,
        llm_model: str | None,
        llm_api_key: str | None,
        llm_api_base: str | None,
        max_context_size: int = 64000,
        context: str = "",
        character: str = "",
        tools: list[dict] | None = None,
        tool_implementations: dict | None = None,
    ):
        """Initialize the ChatManager with LLM configuration and tools.

        Args:
            llm_model: The LLM model to use (e.g., 'gpt-3.5-turbo').
            llm_api_key: The API key for the LLM service.
            llm_api_base: The base URL for the API (optional).
            max_context_size: The maximum allowed tokens in the context window.
            context: Static system context string.
            character: Static persona or character instruction string.
            tools: List of tool schemas.
            tool_implementations: Dictionary mapping tool names to callable
                                  implementations.
        """
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.max_context_size = max_context_size
        self.context = context
        self.character = character
        self.tools = tools or []
        self.tool_implementations = tool_implementations or {}
        logger.info(
            "ChatManager initialized: model=%s api_base=%s max_context=%d "
            "context_len=%d character_len=%d tools=%d",
            self.llm_model,
            self.llm_api_base,
            self.max_context_size,
            len(self.context),
            len(self.character),
            len(self.tools),
        )

    def generate_response(
        self,
        message: str,
        retrieved_context: list[str],
        history: list[dict],
    ) -> str:
        """Construct messages, enforce limits, and call the LLM in an agentic loop.

        Args:
            message: The user's new message.
            retrieved_context: (Deprecated) Relevant text chunks. Now relies on tools.
            history: Conversation history list containing role/content dicts.

        Returns:
            The assistant's generated message or an error/placeholder message.
        """
        system_content = (
            f"Character:\n{self.character}\n\n---\n\n"
            f"Key Context:\n{self.context}\n\n---\n\n"
            "You must fully embody the Character described above. "
            "If you need more information to answer the user's question, "
            "use the available tools to search the lore memory. "
            "If the context does not contain the answer, do not guess or "
            "make up information. "
            "Simply state that you do not know, while remaining in character. "
            "CRITICAL: Your final response MUST be under 3 sentences. "
            "Be extremely brief.\n\n"
        )

        # Include retrieved_context for backward compatibility if provided
        if retrieved_context:
            context_str = "\n---\n".join(retrieved_context)
            system_content += f"Retrieved Context:\n{context_str}"

        system_msg = {"role": "system", "content": system_content}
        logger.debug("System prompt length: %d chars", len(system_content))

        # We append the new message to a local copy of history for the tool loop
        messages = [system_msg] + history + [{"role": "user", "content": message}]

        if not self.llm_api_key:
            logger.warning("LLM API key not configured; returning placeholder message")
            return (
                "LLM not configured: set OPENROUTER_API_KEY environment variable "
                "or provide llm.api_key in config."
            )

        max_steps = 5
        for step in range(max_steps):
            # Trim history if needed
            try:
                while len(messages) > 1:
                    current_tokens = litellm.token_counter(
                        model=self.llm_model or "gpt-3.5-turbo", messages=messages
                    )
                    if current_tokens <= self.max_context_size:
                        break
                    # If too large, remove the oldest non-system message
                    messages.pop(1)
            except Exception as e:
                logger.warning("Failed to count tokens or truncate history: %s", e)

            try:
                logger.info(
                    "Calling LLM (step %d/%d): model=%s api_base=%s",
                    step + 1,
                    max_steps,
                    self.llm_model,
                    self.llm_api_base,
                )
                llm_start = time.perf_counter()

                kwargs = {
                    "model": self.llm_model,
                    "messages": messages,
                    "api_key": self.llm_api_key,
                    "api_base": self.llm_api_base,
                }
                if self.tools:
                    kwargs["tools"] = self.tools

                response = litellm.completion(**kwargs)

                llm_elapsed = time.perf_counter() - llm_start
                response_message = response.choices[0].message

                if response_message.tool_calls:
                    logger.info(
                        "LLM requested %d tool calls", len(response_message.tool_calls)
                    )
                    messages.append(response_message.model_dump())

                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments
                        logger.debug(
                            "Executing tool: %s with args: %s", tool_name, tool_args
                        )

                        tool_result = "Tool not implemented"
                        if tool_name in self.tool_implementations:
                            try:
                                import json

                                parsed_args = json.loads(tool_args)
                                result = self.tool_implementations[tool_name](
                                    **parsed_args
                                )
                                tool_result = str(result)
                            except Exception as e:
                                logger.error(
                                    "Error executing tool %s: %s", tool_name, e
                                )
                                tool_result = f"Error: {e}"

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": tool_result,
                            }
                        )
                    continue  # Continue the loop to let the LLM use the tool results
                else:
                    reply = response_message.content
                    logger.info(
                        "LLM call successful in %.2fs — response=%d chars",
                        llm_elapsed,
                        len(reply) if reply else 0,
                    )
                    return reply
            except Exception as e:
                logger.exception("Error calling LLM (model=%s)", self.llm_model)
                error_str = str(e)
                if len(error_str) > 1000:
                    error_str = error_str[:1000] + "... [truncated]"
                return f"Error calling LLM: {error_str}"

        return "Error: Exceeded maximum tool execution steps."
