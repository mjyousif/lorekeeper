import os
import json
import sqlite3
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Load .env file automatically
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
DB_PATH = os.getenv("SESSION_DB", "sessions.db")


# Initialize SQLite DB for per-chat history
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    chat_id INTEGER PRIMARY KEY,
                    messages TEXT
                )
            """)
        logger.info("Initialized SQLite session DB at %s", DB_PATH)
    except Exception as e:
        logger.error("Failed to initialize DB: %s", e)


init_db()


def get_history(chat_id: int) -> list[dict]:
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.execute("SELECT messages FROM history WHERE chat_id=?", (chat_id,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
            return []
    except Exception as e:
        logger.error("Error reading history for chat_id %d: %s", chat_id, e)
        return []


def set_history(chat_id: int, messages: list[dict]):
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT OR REPLACE INTO history (chat_id, messages) VALUES (?, ?)",
                (chat_id, json.dumps(messages)),
            )
    except Exception as e:
        logger.error("Error saving history for chat_id %d: %s", chat_id, e)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_msg = update.message.text
    logger.info("Received message from chat_id %d: %s", chat_id, user_msg[:100])

    # Load conversation history
    messages = get_history(chat_id)
    messages.append({"role": "user", "content": user_msg})

    # Call rag-wrapper (OpenAI format)
    # Include model parameter as required by OpenAI spec
    model = os.getenv("OPENROUTER_MODEL", "openrouter/stepfun/step-3.5-flash:free")
    payload = {"model": model, "messages": messages}
    try:
        logger.debug("Calling RAG endpoint: %s", RAG_ENDPOINT)
        resp = requests.post(RAG_ENDPOINT, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        assistant_msg = data["choices"][0]["message"]["content"]
        logger.info("RAG response received for chat_id %d", chat_id)
    except Exception as e:
        logger.exception("Error contacting RAG service")
        assistant_msg = f"Error contacting RAG service: {e}"

    # Append assistant reply and save history
    messages.append({"role": "assistant", "content": assistant_msg})
    # Trim history to last 20 messages (10 turns) to cap token usage
    if len(messages) > 20:
        messages = messages[-20:]
    set_history(chat_id, messages)

    # Telegram message limit is 4096 characters, truncate if needed
    max_length = 4096
    if len(assistant_msg) > max_length:
        assistant_msg = assistant_msg[:max_length-50] + "...\n\n[Message truncated due to Telegram limit]"

    await update.message.reply_text(assistant_msg)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Start command from chat_id %d", update.effective_chat.id)
    await update.message.reply_text("Ask me anything about the lore!")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    logger.info("Starting Telegram bot")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
