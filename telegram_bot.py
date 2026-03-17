import os
import json
import sqlite3
import logging
import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from rag_wrapper.config import Config
from telegramify_markdown import convert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load config from config.yaml
config = Config()
try:
    config = Config.from_file("config.yaml")
    logger.info("Loaded config from config.yaml")
except Exception as e:
    logger.warning("Failed to load config from config.yaml: %s", e)

# Authorization configuration
ALLOWED_USER_IDS = set(config.allowed_user_ids or [])
ALLOWED_CHAT_IDS = set(config.allowed_chat_ids or [])

logger.info(
    "Authorization configured: %d allowed users, %d allowed chats",
    len(ALLOWED_USER_IDS),
    len(ALLOWED_CHAT_IDS),
)

def is_authorized(update: Update) -> bool:
    """Check if the user/chat is authorized to use the bot.

    Default: deny all unless at least one allowlist is set.
    """
    user = update.effective_user
    chat = update.effective_chat

    # If no allowlist is configured, deny all (secure default)
    if not ALLOWED_USER_IDS and not ALLOWED_CHAT_IDS:
        logger.debug("No allowlist configured: access denied for user=%s chat=%s",
                     user.id if user else None, chat.id if chat else None)
        return False

    # Check user ID if allowlist is set
    if ALLOWED_USER_IDS and user and user.id in ALLOWED_USER_IDS:
        return True

    # Check chat ID if allowlist is set
    if ALLOWED_CHAT_IDS and chat and chat.id in ALLOWED_CHAT_IDS:
        return True

    return False

# Configuration - read from config.telegram
telegram_cfg = getattr(config, 'telegram', {})
RAG_ENDPOINT = telegram_cfg.get("endpoint", "http://127.0.0.1:8000/v1/chat/completions")
DB_PATH = "sessions.db"  # Hardcoded default for session history
TELEGRAM_BOT_TOKEN = telegram_cfg.get("bot_token")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set in config")
    raise RuntimeError("TELEGRAM_BOT_TOKEN not configured")


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
    # Authorization check
    if not is_authorized(update):
        logger.warning(
            "Unauthorized access: user=%s chat=%s",
            update.effective_user.id if update.effective_user else None,
            update.effective_chat.id if update.effective_chat else None,
        )
        await update.message.reply_text("\u274c You are not authorized to use this bot.")
        return

    chat_id = update.effective_chat.id
    user_msg = update.message.text
    logger.info("Received message from chat_id %d: %s", chat_id, user_msg[:100])

    # Show typing indicator to user
    await update.message.chat.send_action("typing")

    # Load conversation history
    messages = get_history(chat_id)
    messages.append({"role": "user", "content": user_msg})

    # Call rag-wrapper (OpenAI format)
    # Use model from config.llm
    model = config.llm.get("model")
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

    # Format message for Telegram using telegramify-markdown
    text, entities = convert(assistant_msg)

    # Append assistant reply and save history
    messages.append({"role": "assistant", "content": text})
    # Trim history to last 20 messages (10 turns) to cap token usage
    if len(messages) > 20:
        messages = messages[-20:]
    set_history(chat_id, messages)

    # Telegram message limit is 4096 characters, truncate if needed
    max_length = 4096
    if len(text) > max_length:
        text = text[:max_length-50] + "...\n\n[Message truncated due to Telegram limit]"

    await update.message.reply_text(text, entities=[e.to_dict() for e in entities])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        logger.warning(
            "Unauthorized /start from user=%s chat=%s",
            update.effective_user.id if update.effective_user else None,
            update.effective_chat.id if update.effective_chat else None,
        )
        await update.message.reply_text("\u274c You are not authorized to use this bot.")
        return
    logger.info("Start command from chat_id %d", update.effective_chat.id)
    await update.message.reply_text("Ask me anything about the lore!")


def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    logger.info("Starting Telegram bot")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()