# AI Agent Development Guide

Welcome to the LoreKeeper project! This file serves as an entry point for AI coding agents and human developers alike.

The high-level vision for LoreKeeper is a lightweight, agentic framework that behaves in character, bases its responses on deep lore knowledge (via RAG), and utilizes tools to interact with its environment (e.g., generating images).

## Documentation Directory

For detailed documentation, please refer to the `docs/` folder:

* **[Design Document](docs/design.md)**: Architecture, components, and the guiding design principles of the application.
* **[Features Document](docs/features.md)**: Current and planned end-user features.
* **[Testing Strategy](docs/test.md)**: How tests are structured, executed, and the overall testing philosophy.
* **[Configuration Guide](docs/configuration.md)**: Available environment variables and `config.yaml` settings.
* **[Deployment Guide](docs/deployment.md)**: Instructions for running the app via Docker, including local LLM profiles.
* **[API Reference](docs/api.md)**: Details on the FastAPI REST endpoints.

## Maintenance Instructions for Agents

**CRITICAL INSTRUCTION**: When making changes to the codebase (e.g., adding new features, modifying the architecture, adding configuration variables), you MUST update the corresponding markdown files in the `docs/` directory to reflect those changes.

* If you add a new `.env` variable or modify `config.py`, update `docs/configuration.md`.
* If you implement a new agent tool or feature, update `docs/features.md` and `docs/design.md`.
* If you change how tests are run, update `docs/test.md`.
* **If you create a new documentation file, you MUST update this file (`AGENTS.md`) to include a pointer to it and explain when to use it.**

Always keep the documentation in sync with the codebase.
