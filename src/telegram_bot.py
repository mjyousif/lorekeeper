import logging
from functools import lru_cache

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from src.config import get_config
from src.wrapper import LoreKeeper
from src.session_storage import SessionStorage
from telegramify_markdown import convert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

config = get_config()

ALLOWED_USER_IDS = set(config.allowed_user_ids or [])
ALLOWED_CHAT_IDS = set(config.allowed_chat_ids or [])
logger.info(
    "Authorization configured: %d allowed users, %d allowed chats",
    len(ALLOWED_USER_IDS),
    len(ALLOWED_CHAT_IDS),
)

telegram_cfg = config.telegram or {}
DB_PATH = telegram_cfg.get("session_db", "sessions.db")
session_storage = SessionStorage(db_path=DB_PATH)

TELEGRAM_BOT_TOKEN = telegram_cfg.get("bot_token")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set in config.telegram.bot_token")
    raise RuntimeError("TELEGRAM_BOT_TOKEN not configured")


@lru_cache()
def get_wrapper() -> LoreKeeper:
    logger.info("Initializing LoreKeeper for Telegram bot...")
    wrapper = LoreKeeper(config)
    logger.info("LoreKeeper initialization complete.")
    return wrapper


def is_authorized(update: Update) -> bool:
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

    wrapper = get_wrapper()
    messages = session_storage.get_history(chat_id)

    # Sync SQLite history into wrapper session so context is preserved on restart
    session_id = str(chat_id)
    if session_id not in wrapper.sessions:
        wrapper.sessions[session_id] = messages

    try:
        response = wrapper.chat(session_id=session_id, message=user_msg)
        assistant_msg = response["message"]
        logger.info("RAG response received for chat_id %d", chat_id)
    except Exception as e:
        logger.exception("Error in LoreKeeper")
        assistant_msg = "An error occurred while generating the response."

    text, entities = convert(assistant_msg)

    # Persist updated history to SQLite
    updated_history = wrapper.sessions.get(session_id, [])
    if len(updated_history) > 20:
        updated_history = updated_history[-20:]
        wrapper.sessions[session_id] = updated_history
    session_storage.set_history(chat_id, updated_history)

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
