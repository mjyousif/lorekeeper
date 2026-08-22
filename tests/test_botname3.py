import asyncio
from datetime import datetime

from telegram import Chat, Message, MessageEntity, Update, User
from telegram.ext import filters


async def main():
    User(id=1, first_name="Test", is_bot=False)
    c = Chat(id=-1001234, type="group")

    # Text mention
    m_mention = Message(
        message_id=1,
        date=datetime.now(),
        chat=c,
        text="@mybot hello",
        entities=[MessageEntity(type=MessageEntity.MENTION, offset=0, length=6)],
    )

    up = Update(update_id=1, message=m_mention)

    f_mention = filters.Entity(MessageEntity.MENTION)
    print("Does it pass Entity.MENTION?", f_mention.check_update(up))

    # Check what filters.TEXT is
    print("Is TEXT matching it?", filters.TEXT.check_update(up))


asyncio.run(main())
