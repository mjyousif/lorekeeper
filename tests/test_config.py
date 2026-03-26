import pytest
import json
from src.config import Config, _expand_env_vars, get_config

def test_expand_env_vars(monkeypatch):
    """Test environment variable expansion in strings, lists, and dicts."""
    monkeypatch.setenv("TEST_VAR", "hello")
    monkeypatch.setenv("TEST_INT", "42")

    # Simple replacement
    assert _expand_env_vars("${TEST_VAR}") == "hello"

    # Replacement with default
    assert _expand_env_vars("${MISSING_VAR:world}") == "world"
    assert _expand_env_vars("${TEST_VAR:world}") == "hello"

    # Missing without default should be empty string
    assert _expand_env_vars("${MISSING_VAR}") == ""

    # Mixed in string
    assert _expand_env_vars("value is ${TEST_INT}") == "value is 42"

    # Lists
    assert _expand_env_vars(["${TEST_VAR}", "static", "${MISSING:def}"]) == ["hello", "static", "def"]

    # Dicts
    assert _expand_env_vars({"key1": "${TEST_VAR}", "key2": {"nested": "${TEST_INT}"}}) == {"key1": "hello", "key2": {"nested": "42"}}

    # Other types unchanged
    assert _expand_env_vars(42) == 42
    assert _expand_env_vars(True) is True


def test_config_id_parsing():
    """Test parsing of allowed_user_ids and allowed_chat_ids strings to lists of ints."""
    # Test valid comma-separated string
    config = Config(allowed_user_ids="123, 456,789", allowed_chat_ids=" -100 , 200 ")
    assert config.allowed_user_ids == [123, 456, 789]
    assert config.allowed_chat_ids == [-100, 200]

    # Test empty string / whitespace
    config_empty = Config(allowed_user_ids="", allowed_chat_ids="   ")
    assert config_empty.allowed_user_ids is None
    assert config_empty.allowed_chat_ids is None

    # Test already a list
    config_list = Config(allowed_user_ids=[1, 2], allowed_chat_ids=[-3])
    assert config_list.allowed_user_ids == [1, 2]
    assert config_list.allowed_chat_ids == [-3]


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

def test_from_file_yaml(tmp_path, monkeypatch):
    """Test that Config.from_file correctly loads a YAML file."""
    yaml_content = """
    files: test_yaml_data
    db_path: test_yaml_db
    log_level: WARNING
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = Config.from_file(config_file)

    assert config.files == "test_yaml_data"
    assert config.db_path == "test_yaml_db"
    assert config.log_level == "WARNING"

    # Test ImportError handling when yaml is not installed
    import src.config
    monkeypatch.setattr(src.config, "yaml", None)
    with pytest.raises(ImportError, match="PyYAML is required"):
        Config.from_file(config_file)


def test_from_file_toml(tmp_path, monkeypatch):
    """Test that Config.from_file correctly loads a TOML file."""
    toml_content = """
    files = "test_toml_data"
    db_path = "test_toml_db"
    log_level = "CRITICAL"
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml_content)

    config = Config.from_file(config_file)

    assert config.files == "test_toml_data"
    assert config.db_path == "test_toml_db"
    assert config.log_level == "CRITICAL"

    # Test ImportError handling when toml is not installed
    import src.config
    monkeypatch.setattr(src.config, "toml", None)
    with pytest.raises(ImportError, match="toml is required"):
        Config.from_file(config_file)


def test_get_config(tmp_path):
    """Test get_config caching and fallback behavior."""
    # Ensure cache is clear before test
    get_config.cache_clear()

    # 1. Test fallback to defaults when file is missing
    missing_path = str(tmp_path / "missing.yaml")
    config_default = get_config(missing_path)
    assert config_default.files == "data"  # Default value

    # Verify cache info
    cache_info = get_config.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 0

    # 2. Test cache hit
    config_cached = get_config(missing_path)
    assert config_cached is config_default

    cache_info = get_config.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1

    # Clear cache to test actual file loading
    get_config.cache_clear()

    # 3. Test successful load from file
    yaml_content = "files: cached_yaml_data"
    config_file = tmp_path / "valid.yaml"
    config_file.write_text(yaml_content)

    config_loaded = get_config(str(config_file))
    assert config_loaded.files == "cached_yaml_data"

    # Verify cache works for the loaded file
    config_loaded_cached = get_config(str(config_file))
    assert config_loaded_cached is config_loaded

    cache_info = get_config.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1


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
