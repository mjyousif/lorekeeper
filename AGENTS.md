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

## Strict Development Rules for Agents

To prevent broken builds, CI failures, and wasted time, all AI agents working in this repository **MUST** adhere to the following rules:

### 1. NEVER Bypass Pre-Commit Hooks
* **DO NOT** use `git commit --no-verify`. This is strictly forbidden.
* All commits must successfully pass the repository's pre-commit hooks (`black`, `ruff format`, `ruff check`, `mypy`, `bandit`).
* If a pre-commit hook fails—even if the failure is in a file you didn't originally touch, or is related to trailing whitespaces/EOF—you must fix the issue properly before committing.
* Local hook bypasses simply push failures to the GitHub Actions CI pipeline, which is unacceptable.

### 2. Cross-Platform Awareness
* The local development environment is often Windows, but the GitHub Actions CI pipeline runs on **Linux** (`ubuntu-latest`).
* Code must be platform-agnostic.
* When writing OS-specific code (e.g., Windows `subprocess` flags like `CREATE_NO_WINDOW`), you must account for `mypy` running on Linux in CI. Use `# type: ignore[attr-defined]` on platform-specific attributes to prevent Linux CI type-checking failures.

### 3. Testing and Execution
* Use `uv` to manage the environment and run tools.
* To avoid Windows script path canonicalization issues, run tests explicitly as a module: `uv run python -m pytest`.
* Always run tests before committing.
* Ensure code is explicitly formatted (`uv run ruff format .`) and passes linters (`uv run ruff check .`, `uv run python -m mypy src`) before finalizing work.

### 4. Avoiding CI Collection Failures
* **Module-Level Execution**: NEVER raise exceptions or execute side effects at the module level (e.g., checking environment variables, opening files). Doing so will cause `pytest` to fail during the test collection phase when it imports the module. Place configuration and validation checks inside initialization functions or entrypoints (e.g., `main()`).
* **CI Environment Dependencies**: If you introduce tests that require optional dependency groups (e.g., `gradio` in the `ui` group), you MUST ensure the `.github/workflows/test.yml` file is updated to install those dependencies (e.g., adding `--group ui` to `uv sync`). Local tests passing is not enough; the CI environment must mirror the required dependencies.
