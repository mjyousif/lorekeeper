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
from rag_wrapper.config import get_config
from telegramify_markdown import convert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Single config load
config = get_config()

# Authorization
ALLOWED_USER_IDS = set(config.allowed_user_ids or [])
ALLOWED_CHAT_IDS = set(config.allowed_chat_ids or [])
logger.info(
    "Authorization configured: %d allowed users, %d allowed chats",
    len(ALLOWED_USER_IDS),
    len(ALLOWED_CHAT_IDS),
)

# Telegram config
telegram_cfg = config.telegram or {}
RAG_ENDPOINT = telegram_cfg.get("endpoint", "http://127.0.0.1:8000/v1/chat/completions")
DB_PATH = telegram_cfg.get("session_db", "sessions.db")
TELEGRAM_BOT_TOKEN = telegram_cfg.get("bot_token")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set in config.telegram.bot_token")
    raise RuntimeError("TELEGRAM_BOT_TOKEN not configured")

# LLM model from config
LLM_MODEL = (config.llm or {}).get("model", "openrouter/stepfun/step-3.5-flash:free")


def is_authorized(update: Update) -> bool:
    """Check if the user/chat is authorized. Denies all if no allowlist is configured."""
    user = update.effective_user
    chat = update.effective_chat

    if not ALLOWED_USER_IDS and not ALLOWED_CHAT_IDS:
        logger.debug(
            "No allowlist configured: access denied for user=%s chat=%s",
            user.id if user else None,
            chat.id if chat else None,
        )
        return False

    if ALLOWED_USER_IDS and user and user.id in ALLOWED_USER_IDS:
        return True

    if ALLOWED_CHAT_IDS and chat and chat.id in ALLOWED_CHAT_IDS:
        return True

    return False


# --- SQLite session history ---

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


def get_history(chat_id: int) -> list[dict]:
    try:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.execute("SELECT messages FROM history WHERE chat_id=?", (chat_id,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else []
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


init_db()


# --- Handlers ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        logger.warning(
            "Unauthorized access: user=%s chat=%s",
            update.effective_user.id if update.effective_user else None,
            update.effective_chat.id if update.effective_chat else None,
        )
        await update.message.reply_text("❌ You are not authorized to use this bot.")
        return

    chat_id = update.effective_chat.id
    user_msg = update.message.text
    logger.info("Received message from chat_id %d: %s", chat_id, user_msg[:100])

    await update.message.chat.send_action("typing")

    messages = get_history(chat_id)
    messages.append({"role": "user", "content": user_msg})

    payload = {"model": LLM_MODEL, "messages": messages}
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

    text, entities = convert(assistant_msg)

    messages.append({"role": "assistant", "content": text})
    if len(messages) > 20:
        messages = messages[-20:]
    set_history(chat_id, messages)

    if len(text) > 4096:
        text = text[:4046] + "...\n\n[Message truncated due to Telegram limit]"

    await update.message.reply_text(text, entities=[e.to_dict() for e in entities])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        logger.warning(
            "Unauthorized /start from user=%s chat=%s",
            update.effective_user.id if update.effective_user else None,
            update.effective_chat.id if update.effective_chat else None,
        )
        await update.message.reply_text("❌ You are not authorized to use this bot.")
        return
    logger.info("Start command from chat_id %d", update.effective_chat.id)
    await update.message.reply_text("Ask me anything about the lore!")


def main():
    logger.info("Starting Telegram bot")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
