import os
import sys
import importlib.util
from typing import Any

spec = importlib.util.spec_from_file_location("config", "src/config.py")
config_module = importlib.util.module_from_spec(spec)
sys.modules["src.config"] = config_module
config_module.Any = Any
spec.loader.exec_module(config_module)

from src.config import Config
Config.model_rebuild()

os.environ['ALLOWED_CHAT_IDS'] = '-100123'
os.environ['ALLOWED_USER_IDS'] = '456'

c = Config()
print("Chat IDs:", c.allowed_chat_ids)
print("Type of Chat IDs:", type(c.allowed_chat_ids))

os.environ['ALLOWED_CHAT_IDS'] = '-100123,-456789'
c2 = Config()
print("Chat IDs comma:", c2.allowed_chat_ids)
print("Type of Chat IDs comma:", type(c2.allowed_chat_ids))
