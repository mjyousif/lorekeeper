import asyncio
from telegram import Message, User, Chat, Update, MessageEntity
from telegram.ext import filters
from datetime import datetime

async def main():
    u = User(id=1, first_name="Test", is_bot=False)
    c = Chat(id=-1001234, type="group")

    # What if they tag the bot via inline mention, or mention the username exactly?
    # Actually, in group chats with privacy mode on, the bot *only* receives messages that start with its username, or reply to its messages.
    # WAIT! There is a specific issue with `filters.TEXT & ~filters.COMMAND`.
    # Does `~filters.COMMAND` filter out mentions? NO.

    m_mention = Message(
        message_id=1,
        date=datetime.now(),
        chat=c,
        text="@mybot hello",
        entities=[MessageEntity(type=MessageEntity.MENTION, offset=0, length=6)]
    )

    up = Update(update_id=1, message=m_mention)

    f_text = filters.TEXT & ~filters.COMMAND
    print("Does TEXT pass?", f_text.check_update(up))

asyncio.run(main())
