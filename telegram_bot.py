import os
import json
import sqlite3
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Configuration
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
DB_PATH = os.getenv("SESSION_DB", "sessions.db")

# Initialize SQLite DB for per-chat history
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                chat_id INTEGER PRIMARY KEY,
                messages TEXT
            )
        """)
init_db()

def get_history(chat_id: int) -> list[dict]:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("SELECT messages FROM history WHERE chat_id=?", (chat_id,))
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return []

def set_history(chat_id: int, messages: list[dict]):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT OR REPLACE INTO history (chat_id, messages) VALUES (?, ?)",
            (chat_id, json.dumps(messages))
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_msg = update.message.text

    # Load conversation history
    messages = get_history(chat_id)
    messages.append({"role": "user", "content": user_msg})

    # Call rag-wrapper (OpenAI format)
    payload = {"messages": messages}
    try:
        resp = requests.post(RAG_ENDPOINT, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        assistant_msg = data["choices"][0]["message"]["content"]
    except Exception as e:
        assistant_msg = f"Error contacting RAG service: {e}"

    # Append assistant reply and save history
    messages.append({"role": "assistant", "content": assistant_msg})
    # Trim history to last 20 messages (10 turns) to cap token usage
    if len(messages) > 20:
        messages = messages[-20:]
    set_history(chat_id, messages)

    await update.message.reply_text(assistant_msg)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ask me anything about the lore!")

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
