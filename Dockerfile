FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen to ensure reproducibility based on uv.lock)
RUN uv sync --frozen

# Copy the rest of the application source code
COPY . .

# Run the Telegram bot
CMD ["uv", "run", "python", "-m", "src.telegram_bot"]
