#!/usr/bin/env python3
"""
Gradio UI for manually testing the RAGWrapper.

Run: python gradio_app.py
Then open http://localhost:7860
"""

import os
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from rag_wrapper.wrapper import RAGWrapper
from rag_wrapper.config import Config

# Load configuration from config.yaml
CONFIG_PATH = os.getenv("RAG_CONFIG_PATH", "config.yaml")
try:
    cfg = Config.from_file(CONFIG_PATH)
    DATA_DIR = cfg.files
    DB_PATH = cfg.db_path
    LLM_MODEL = cfg.llm.get("model", "openrouter/stepfun/step-3.5-flash:free")
    CHUNK_SIZE = cfg.chunk_size
    OVERLAP = cfg.overlap
    LOG_LEVEL = cfg.log_level
    CONTEXT_FILE = cfg.context_file
    CHARACTER_FILE = cfg.character_file
except Exception as e:
    print(f"Warning: Failed to load config from {CONFIG_PATH}: {e}. Using defaults.")
    DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
    DB_PATH = os.getenv("RAG_DB_PATH", "shared_db")
    LLM_MODEL = os.getenv("RAG_LLM_MODEL", "openrouter/stepfun/step-3.5-flash:free")
    CHUNK_SIZE = 1000
    OVERLAP = 200
    LOG_LEVEL = "INFO"
    CONTEXT_FILE = None
    CHARACTER_FILE = None

# Initialize wrapper (lazy, will be created on first use)
_wrapper: RAGWrapper | None = None


def get_wrapper():
    """Get or create the RAGWrapper singleton."""
    global _wrapper
    if _wrapper is None:
        _wrapper = RAGWrapper(
            files=DATA_DIR,
            db_path=DB_PATH,
            llm_model=LLM_MODEL,
        )
    return _wrapper


def rag_query(
    query: str,
    session_id: str,
    n_results: int,
    include_context: bool,
    include_history: bool,
):
    """Process a RAG query and return response + retrieved context."""
    if not query.strip():
        return "Please enter a query.", "", session_id

    wrapper = get_wrapper()

    # For history tracking, we store per session. In this UI, we'll keep history in Gradio's state.
    # The wrapper's `sessions` dict lives in memory; we pass the session_id.

    # Get relevant context
    context_chunks = wrapper.get_relevant_context(query, n_results=n_results)
    context_str = (
        "\n---\n".join(context_chunks)
        if context_chunks
        else "No relevant context found."
    )

    # Call chat (which includes context in system prompt and manages history)
    response = wrapper.chat(session_id=session_id, message=query)

    assistant_message = response["message"]

    # Format output
    if include_context:
        full_output = f"**Response:**\n{assistant_message}\n\n**Retrieved Context:**\n{context_str}"
    else:
        full_output = assistant_message

    # Optionally include conversation history length info
    history_info = ""
    if include_history:
        history = wrapper.sessions.get(session_id, [])
        history_info = f"Session has {len(history)} messages (including this turn)."

    return full_output, context_str, session_id, history_info


def clear_session(session_id: str):
    """Clear conversation history for a session."""
    wrapper = get_wrapper()
    if session_id in wrapper.sessions:
        del wrapper.sessions[session_id]
        return f"Cleared session {session_id}."
    return f"Session {session_id} not found."


def rebuild_index():
    """Force rebuild of the vector index."""
    wrapper = get_wrapper()
    wrapper._manifest = {}  # Invalidate to force rebuild on next query
    return "Index will rebuild on next query."


# Build Gradio interface
with gr.Blocks(title="RAG Wrapper UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Wrapper Testing UI")
    gr.Markdown(f"**Data directory:** `{DATA_DIR}` | **DB:** `{DB_PATH}`")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Query",
                placeholder="Ask something about the documents...",
                lines=3,
            )
            session_id = gr.Textbox(
                label="Session ID",
                value="test-session",
                info="Keep the same to maintain conversation history across turns.",
            )
            with gr.Row():
                n_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of context chunks",
                )
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Accordion("Options", open=False):
                include_context = gr.Checkbox(
                    label="Show retrieved context in output", value=True
                )
                include_history = gr.Checkbox(
                    label="Show session history info", value=True
                )

            with gr.Row():
                clear_btn = gr.Button("Clear Session")
                rebuild_btn = gr.Button("Rebuild Index")

        with gr.Column(scale=1):
            context_output = gr.Textbox(
                label="Retrieved Context", lines=20, interactive=False
            )
            info_output = gr.Textbox(label="Info", lines=2, interactive=False)

    # Main output area (full response)
    response_output = gr.Markdown(label="Response")

    # Event handlers
    submit_event = submit_btn.click(
        fn=rag_query,
        inputs=[query_input, session_id, n_results, include_context, include_history],
        outputs=[response_output, context_output, session_id, info_output],
    )

    clear_btn.click(
        fn=clear_session,
        inputs=[session_id],
        outputs=[info_output],
    )

    rebuild_btn.click(
        fn=rebuild_index,
        inputs=[],
        outputs=[info_output],
    )

    # Allow Enter key to submit (with Ctrl/Cmd modifier to avoid accidental submits)
    query_input.submit(
        fn=rag_query,
        inputs=[query_input, session_id, n_results, include_context, include_history],
        outputs=[response_output, context_output, session_id, info_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        "**Note:** The LLM requires an OpenRouter API key set in environment variable `OPENROUTER_API_KEY` or `LLM_API_KEY`."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
