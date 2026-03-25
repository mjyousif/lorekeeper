#!/usr/bin/env python3
"""
Gradio UI for manually testing the LoreKeeper.

Run: python gradio_app.py
Then open http://localhost:7860
"""

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from src.wrapper import LoreKeeper
from src.config import get_config

cfg = get_config()  # Single load, cached, falls back to defaults if no config.yaml


@lru_cache()
def get_wrapper() -> LoreKeeper:
    """Get or create the LoreKeeper singleton."""
    return LoreKeeper(get_config())


def rag_query(
    query: str,
    session_id: str,
    n_results: int,
    include_context: bool,
    include_history: bool,
):
    """Process a RAG query and return response + retrieved context."""
    if not query.strip():
        return "Please enter a query.", "", session_id, ""

    wrapper = get_wrapper()

    context_chunks = wrapper.get_relevant_context(query, n_results=n_results)
    context_str = (
        "\n---\n".join(context_chunks)
        if context_chunks
        else "No relevant context found."
    )

    response = wrapper.chat(session_id=session_id, message=query)
    assistant_message = response["message"]

    if include_context:
        full_output = f"**Response:**\n{assistant_message}\n\n**Retrieved Context:**\n{context_str}"
    else:
        full_output = assistant_message

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
    wrapper._manifest = {}
    return "Index will rebuild on next query."


# Build Gradio interface
with gr.Blocks(title="LoreKeeper UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LoreKeeper Testing UI")
    gr.Markdown(f"**Data directory:** `{cfg.files}` | **DB:** `{cfg.db_path}`")

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

    response_output = gr.Markdown(label="Response")

    submit_btn.click(
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

    query_input.submit(
        fn=rag_query,
        inputs=[query_input, session_id, n_results, include_context, include_history],
        outputs=[response_output, context_output, session_id, info_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        "**Note:** The LLM requires an OpenRouter API key set in `OPENROUTER_API_KEY` or `LLM_API_KEY`."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
