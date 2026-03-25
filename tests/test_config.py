import pytest
import json
from src.config import Config

def test_from_file_unsupported_format(tmp_path):
    """Test that Config.from_file raises ValueError for unsupported formats."""
    # Create a mock file with an unsupported extension (.ini)
    config_file = tmp_path / "config.ini"
    config_file.write_text("[section]\nkey=value")

    with pytest.raises(ValueError, match="Unsupported format: .ini"):
        Config.from_file(config_file)

def test_from_file_not_found():
    """Test that Config.from_file raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError, match="Config file not found: non_existent.yaml"):
        Config.from_file("non_existent.yaml")

def test_from_file_json(tmp_path):
    """Test that Config.from_file correctly loads a JSON file."""
    config_data = {
        "files": "test_data",
        "db_path": "test_db",
        "log_level": "DEBUG"
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    config = Config.from_file(config_file)

    assert config.files == "test_data"
    assert config.db_path == "test_db"
    assert config.log_level == "DEBUG"
