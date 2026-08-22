import asyncio
import logging
import time
from functools import lru_cache

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegramify_markdown import convert

from src.auth_storage import AuthStorage
from src.config import get_config
from src.session_storage import SessionStorage
from src.wrapper import LoreKeeper

config = get_config()

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_USER_IDS = set(config.allowed_user_ids or [])  # type: ignore
ALLOWED_CHAT_IDS = set(config.allowed_chat_ids or [])  # type: ignore
logger.info(
    "Authorization configured: %d allowed users, %d allowed chats",
    len(ALLOWED_USER_IDS),
    len(ALLOWED_CHAT_IDS),
)

telegram_cfg = config.telegram or {}
DB_PATH = telegram_cfg.get("session_db", "sessions.db")
session_storage = SessionStorage(db_path=DB_PATH)
auth_storage = AuthStorage(db_path=DB_PATH)

TELEGRAM_BOT_TOKEN = telegram_cfg.get("bot_token")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set in config.telegram.bot_token")
    raise RuntimeError("TELEGRAM_BOT_TOKEN not configured")


@lru_cache()
def get_wrapper() -> LoreKeeper:
    """Dependency provider for LoreKeeper."""
    logger.info("Initializing LoreKeeper for Telegram bot...")
    start = time.perf_counter()
    wrapper = LoreKeeper(config)
    elapsed = time.perf_counter() - start
    logger.info("LoreKeeper initialization complete (took %.2fs)", elapsed)
    return wrapper


def is_authorized(update: Update) -> bool:
    """Check if the user/chat is allowed to interact with the bot."""
    user = update.effective_user
    chat = update.effective_chat
    user_id = user.id if user else None
    chat_id = chat.id if chat else None
    logger.debug("Checking authorization for user=%s chat=%s", user_id, chat_id)

    # Check if they're statically allowed via config
    if ALLOWED_USER_IDS and user and user.id in ALLOWED_USER_IDS:
        logger.debug("User %s authorized via static config", user_id)
        return True

    if ALLOWED_CHAT_IDS and chat and chat.id in ALLOWED_CHAT_IDS:
        logger.debug("Chat %s authorized via static config", chat_id)
        return True

    # Check if they're dynamically allowed via db
    if user and auth_storage.is_user_authorized(user.id):
        logger.debug("User %s authorized via database", user_id)
        return True

    if chat and auth_storage.is_chat_authorized(chat.id):
        logger.debug("Chat %s authorized via database", chat_id)
        return True

    # Note: By default we deny if not explicitly allowed
    logger.debug("Authorization denied for user=%s chat=%s", user_id, chat_id)
    return False


def is_user_authorized_only(user) -> bool:
    """Check if a user (not chat) is authorized."""
    if not user:
        return False
    if ALLOWED_USER_IDS and user.id in ALLOWED_USER_IDS:
        logger.debug("User %s authorized (user-only check, static)", user.id)
        return True
    if auth_storage.is_user_authorized(user.id):
        logger.debug("User %s authorized (user-only check, database)", user.id)
        return True
    logger.debug("User %s not authorized (user-only check)", user.id)
    return False


# --- Handlers ---


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process incoming user messages."""
    user_id = update.effective_user.id if update.effective_user else None
    chat_id = update.effective_chat.id if update.effective_chat else None
    username = update.effective_user.username if update.effective_user else "unknown"

    if not is_authorized(update):
        logger.warning(
            "Unauthorized message rejected: user=%s (@%s) chat=%s",
            user_id,
            username,
            chat_id,
        )
        await update.message.reply_text("❌ You are not authorized to use this bot.")  # type: ignore
        return

    user_msg = update.message.text  # type: ignore

    # In group chats, only respond if mentioned or replied to
    chat_type = update.effective_chat.type  # type: ignore
    if chat_type in ["group", "supergroup"]:
        bot = context.bot
        replied_to_bot = (
            update.message.reply_to_message  # type: ignore
            and update.message.reply_to_message.from_user.id == bot.id  # type: ignore
        )
        mentioned = bot.username and f"@{bot.username}" in user_msg  # type: ignore

        if not (replied_to_bot or mentioned):
            logger.debug(
                "Ignoring group message (not mentioned/replied): chat=%s user=%s",
                chat_id,
                user_id,
            )
            return

        if bot.username:
            user_msg = user_msg.replace(f"@{bot.username}", "").strip()  # type: ignore

    if not user_msg:
        logger.debug("Empty message after processing, ignoring (chat=%s)", chat_id)
        return

    logger.info(
        "[chat=%s user=%s @%s] Received message (%d chars): %s",
        chat_id,
        user_id,
        username,
        len(user_msg),
        user_msg[:120],
    )

    # --- Continuous typing indicator ---
    # Telegram's "typing..." status expires after ~5 seconds.
    # We keep re-sending it every 4 seconds until the LLM responds.
    typing_active = True

    async def keep_typing():
        """Send typing action every 4 seconds until cancelled."""
        while typing_active:
            try:
                await update.message.chat.send_action("typing")
            except Exception as e:
                logger.warning("[chat=%s] Failed to send typing action: %s", chat_id, e)
                break
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(keep_typing())

    wrapper = get_wrapper()
    logger.debug("[chat=%s] Loading session history from SQLite", chat_id)
    messages = session_storage.get_history(chat_id)  # type: ignore
    logger.debug("[chat=%s] Loaded %d history messages", chat_id, len(messages))

    # Sync SQLite history into wrapper session so context is preserved on restart
    session_id = str(chat_id)
    if session_id not in wrapper.sessions:
        wrapper.sessions[session_id] = messages
        logger.debug(
            "[chat=%s] Initialized new wrapper session from SQLite history", chat_id
        )

    try:
        llm_start = time.perf_counter()
        logger.info("[chat=%s] Calling LoreKeeper.chat()...", chat_id)
        response = wrapper.chat(session_id=session_id, message=user_msg)
        llm_elapsed = time.perf_counter() - llm_start
        assistant_msg = response["message"]
        logger.info(
            "[chat=%s] LoreKeeper response received in %.2fs (%d chars)",
            chat_id,
            llm_elapsed,
            len(assistant_msg),
        )
    except Exception:
        logger.exception("[chat=%s] Error in LoreKeeper.chat()", chat_id)
        assistant_msg = "An error occurred while generating the response."
    finally:
        # Stop the typing indicator
        typing_active = False
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    logger.debug("[chat=%s] Converting response to Telegram markdown", chat_id)
    text, entities = convert(assistant_msg)

    # Persist updated history to SQLite
    updated_history = wrapper.sessions.get(session_id, [])
    if len(updated_history) > 20:
        logger.debug(
            "[chat=%s] Trimming session history from %d to 20 messages",
            chat_id,
            len(updated_history),
        )
        updated_history = updated_history[-20:]
        wrapper.sessions[session_id] = updated_history
    session_storage.set_history(chat_id, updated_history)  # type: ignore
    logger.debug(
        "[chat=%s] Persisted %d history messages to SQLite",
        chat_id,
        len(updated_history),
    )

    if len(text) > 4096:
        logger.warning(
            "[chat=%s] Response truncated from %d to 4096 chars for Telegram",
            chat_id,
            len(text),
        )
        text = text[:4046] + "...\n\n[Message truncated due to Telegram limit]"

    await update.message.reply_text(text, entities=[e.to_dict() for e in entities])  # type: ignore
    logger.info("[chat=%s] Reply sent successfully", chat_id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    user_id = update.effective_user.id if update.effective_user else None
    chat_id = update.effective_chat.id if update.effective_chat else None
    username = update.effective_user.username if update.effective_user else "unknown"
    logger.info(
        "[chat=%s user=%s @%s] /start command received", chat_id, user_id, username
    )

    if not is_authorized(update):
        logger.warning(
            "[chat=%s user=%s @%s] Unauthorized /start rejected",
            chat_id,
            user_id,
            username,
        )
        await update.message.reply_text("❌ You are not authorized to use this bot.")  # type: ignore
        return
    await update.message.reply_text("Ask me anything about the lore!")  # type: ignore
    logger.info("[chat=%s] /start response sent", chat_id)


async def pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /pair command for user and chat authorization."""
    user = update.effective_user
    chat = update.effective_chat

    if not user:
        logger.warning(
            "[chat=%s] /pair command with no user, ignoring", chat.id if chat else None
        )
        return

    chat_type = chat.type  # type: ignore
    logger.info(
        "[chat=%s user=%s @%s] /pair command received (chat_type=%s)",
        chat.id,  # type: ignore
        user.id,
        user.username,
        chat_type,
    )

    if chat_type == "private":
        # Check if user is already authorized
        if is_user_authorized_only(user):
            logger.info("[user=%s] /pair: user already authorized", user.id)
            await update.message.reply_text("✅ You are already authorized!")  # type: ignore
            return

        code = auth_storage.create_pending_pair(user.id)
        logger.info("[user=%s] Pairing code generated: %s", user.id, code)
        msg = (
            f"🔑 Your pairing code is: `{code}`\n\n"
            "Take this code to the terminal where the bot is running and run:\n"
            f"`chatter approve {code}`"
        )
        await update.message.reply_markdown(msg)  # type: ignore
        return

    # In a group/channel chat
    if not is_user_authorized_only(user):
        logger.warning(
            "[chat=%s user=%s] /pair rejected: user not authorized to pair channels",
            chat.id,  # type: ignore
            user.id,
        )
        await update.message.reply_text("❌ Only authorized users can pair channels.")  # type: ignore
        return

    if is_authorized(update):
        # We know the user is authorized, so if the overall chat is authorized, we are good.
        # But let's check specifically if the chat is authorized.
        if (
            ALLOWED_CHAT_IDS and chat.id in ALLOWED_CHAT_IDS  # type: ignore
        ) or auth_storage.is_chat_authorized(chat.id):  # type: ignore
            logger.info("[chat=%s] /pair: chat already authorized", chat.id)  # type: ignore
            await update.message.reply_text("✅ This chat is already authorized!")  # type: ignore
            return

    code = auth_storage.create_pending_pair(user.id, chat.id)  # type: ignore
    logger.info(
        "[chat=%s user=%s] Chat pairing code generated: %s", chat.id, user.id, code  # type: ignore
    )
    msg = (
        f"🔑 Chat pairing code is: `{code}`\n\n"
        "Take this code to the terminal where the bot is running and run:\n"
        f"`chatter approve {code}`"
    )
    await update.message.reply_markdown(msg)  # type: ignore


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /clear command to wipe history for this chat."""
    user = update.effective_user
    chat = update.effective_chat
    user_id = user.id if user else None
    chat_id = chat.id if chat else None
    username = user.username if user else "unknown"

    logger.info(
        "[chat=%s user=%s @%s] /clear command received", chat_id, user_id, username
    )

    if not is_authorized(update):
        logger.warning(
            "[chat=%s user=%s @%s] Unauthorized /clear rejected",
            chat_id,
            user_id,
            username,
        )
        await update.message.reply_text("❌ You are not authorized to use this bot.")  # type: ignore
        return

    chat_type = chat.type if chat else "private"
    if chat_type in ["group", "supergroup"] and not is_user_authorized_only(user):
        logger.warning(
            "[chat=%s user=%s] /clear rejected: user not authorized to clear group chat history",
            chat_id,
            user_id,
        )
        await update.message.reply_text(  # type: ignore
            "❌ Only individually authorized users can clear group chat history."
        )
        return

    # Clear SQLite history
    if chat_id:
        session_storage.set_history(chat_id, [])
        # Clear in-memory history
        wrapper = get_wrapper()
        session_id = str(chat_id)
        wrapper.sessions[session_id] = []
        logger.info("[chat=%s] History cleared by user=%s", chat_id, user_id)
        await update.message.reply_text("🧹 Chat history has been cleared.")  # type: ignore


def main():
    """Start the Telegram bot."""
    logger.info("Starting Telegram bot (token ending ...%s)", TELEGRAM_BOT_TOKEN[-6:])
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("pair", pair))
    app.add_handler(CommandHandler("clear", clear_history))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Handlers registered: /start, /pair, /clear, message")
    logger.info("Bot polling started — waiting for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()
