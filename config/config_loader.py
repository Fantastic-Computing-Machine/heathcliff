# ABOUTME: Configuration loader that merges environment variables and YAML settings
# ABOUTME: Provides centralized access to all configuration values

import yaml
from dotenv import load_dotenv
import os
from typing import Dict, Any


class Config:
    """Centralized configuration manager for Heathcliff assistant."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from .env and YAML files.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load environment variables
        load_dotenv()

        # Load YAML configuration
        with open(config_path, "r") as f:
            self.yaml_config = yaml.safe_load(f)

        # API Keys from environment
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.openweathermap_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.google_search_api_key = os.getenv("GOOGLE_CSE_API_KEY")
        self.google_search_cse_id = os.getenv("GOOGLE_CSE_ID")

        # Langfuse observability
        self.langfuse_base_url = os.getenv("LANGFUSE_BASE_URL")
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST")
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")

        # Optional Twitter/X keys
        self.twitter_api_key = os.getenv("TWITTER_API_KEY")
        self.twitter_api_secret = os.getenv("TWITTER_API_SECRET")
        self.twitter_access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.twitter_access_secret = os.getenv("TWITTER_ACCESS_SECRET")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value from YAML by dot notation.

        Args:
            key: Dot-separated key path (e.g., 'news.sources')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.yaml_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def validate(self) -> bool:
        """
        Validate that all required API keys are present.

        Returns:
            True if all required keys present, False otherwise
        """
        required_keys = {
            "GEMINI_API_KEY": self.gemini_key,
        }

        missing_keys = [k for k, v in required_keys.items() if not v]

        if missing_keys:
            print(f"Missing required API keys: {', '.join(missing_keys)}")
            return False

        return True

    def __repr__(self) -> str:
        """String representation hiding sensitive data."""
        return f"<Config wake_word='{self.get('wake_word')}' model='{self.get('llm.model')}'>"


# Singleton instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get or create singleton Config instance.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
