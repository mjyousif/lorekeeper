import sys
import importlib.util
from typing import Any
import os

spec = importlib.util.spec_from_file_location("config", "src/config.py")
config_module = importlib.util.module_from_spec(spec)
sys.modules["src.config"] = config_module
config_module.Any = Any
spec.loader.exec_module(config_module)

from src.config import Config
Config.model_rebuild()

try:
    c = Config(allowed_chat_ids=["-1001234", "-567890"])
    print("Chat IDs:", c.allowed_chat_ids)
    print("Type of Chat IDs elements:", [type(x) for x in c.allowed_chat_ids])
except Exception as e:
    print(e)
