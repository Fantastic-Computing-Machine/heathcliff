# ABOUTME: Class-based configuration with attribute access and cached singleton

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from config.master_info import MASTER_INFO
from logger import logger

load_dotenv(".env")


class MasterConf:
    MASTER_INFO = MASTER_INFO
    TZ = "America/New_York"


class ChromaConf:
    USE_REMOTE_CHROMA = os.getenv("USE_REMOTE_CHROMA", "false").lower() == "true"

    CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "heathcliff")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "https://api.trychroma.com")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"


class RuntimeConf:
    # Model def
    # MODEL = "gemini-2.5-flash-lite"
    SUPERVISOR_MODEL = "google_genai:gemini-3-flash-preview"
    TOOL_MODEL = "google_genai:gemini-2.5-pro"
    # Supervisor Hyperparameters
    TEMPERATURE = 0.5
    MAX_TOKENS = 8192
    TOP_P = 0.7
    MAX_ITERATIONS = 20
    TIMEOUT_SECONDS = 300
    MAX_RETRIES = 3


class PlatformConf:
    # Google / Gemini API Keys
    AI_KEY = os.getenv("AI_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", "credentials.json"
    )
    # Google Search
    GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

    # Spotify API Credentials
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

    # Messaging Platform Tokens
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Twitter API Credentials
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")


class Mem0Conf:
    USER_ID = "heathcliff"
    MEM0_AGENT_ID = "heathcliff_agent"

    MEMORY_CHAT_CONTEXT = 10
    MEMORY_MAX_MEMORIES = 5
    # Number of recent chronological message pairs (1 pair = 1 user + 1 AI msg)
    RECENT_PAIRS_COUNT = 6
    # Number of semantic history message pairs retrieved per query
    SEMANTIC_PAIRS_COUNT = 3

    MEMORY_COLLECTION = "heathcliff_memories"

    _llm_config: Dict[str, Any] = {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.5-flash-lite",
            "temperature": 0.2,
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
    }
    _embedder_config: Dict[str, Any] = {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-001",
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
    }

    _vector_store_config: Dict[str, Any] = {
        "provider": "chroma",
        "config": {
            "collection_name": MEMORY_COLLECTION,
            "host": ChromaConf.CHROMA_HOST,
            "port": ChromaConf.CHROMA_PORT,
            "api_key": ChromaConf.CHROMA_API_KEY,
            "tenant": ChromaConf.CHROMA_TENANT,
        },
    }

    MEM0_CONFIG: Dict[str, Dict] = {
        "llm": _llm_config,
        "embedder": _embedder_config,
        "vector_store": _vector_store_config,
    }


class LangFuseConf:
    LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
    LANGFUSE_RELEASE = os.getenv("LANGFUSE_RELEASE")

    TRACE_NAME = "heathcliff.agent"
    LANGFUSE_USER_ID = "adiagarwal"
    ENVIRONMENT = os.getenv("LANGFUSE_ENVIRONMENT", "local-dev")


class NewsConfig:
    # News API
    NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
    DEFAULT_SOURCES = [
        "bbc-news",
        "techcrunch",
        "hacker-news",
        "the-verge",
        "wired",
        "engadget",
        "ars-technica",
        "y-combinator",
        "linkekdin",
        "github",
    ]
    DEFAULT_TOPICS = [
        "technology",
        "artificial-intelligence",
        "science",
        "startups",
        "programming",
        "gadgets",
    ]
    MAX_ARTICLES = 5


class WeatherConfig:
    # Weather API
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
    DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Jersey City")
    UNITS = os.getenv("WEATHER_UNITS", "metric")


class AudioConfig:
    WAKE_WORD = "heathcliff"
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512
    TTS_RATE = 175
    TTS_VOLUME = 0.9
    TTS_VOICE = None


class Conf(
    RuntimeConf,
    MasterConf,
    ChromaConf,
    Mem0Conf,
    PlatformConf,
    LangFuseConf,
    NewsConfig,
    WeatherConfig,
    AudioConfig,
):
    _instance: Optional["Conf"] = None

    def __new__(cls) -> "Conf":
        if cls._instance is None:
            cls._instance = super(Conf, cls).__new__(cls)
        return cls._instance

    @classmethod
    def validate(cls) -> None:
        missing_keys = []
        if not cls.AI_KEY:
            missing_keys.append("AI_KEY")
        if not cls.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY")
        if not cls.CHROMA_API_KEY:
            missing_keys.append("CHROMA_API_KEY")
        if not cls.CHROMA_TENANT:
            missing_keys.append("CHROMA_TENANT")
        if not cls.CHROMA_DATABASE:
            missing_keys.append("CHROMA_DATABASE")

        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

        logger.info(f"Configuration validated successfully!")

    def __repr__(self) -> str:
        return f"<Config wake_word='{self.WAKE_WORD}' model='{self.SUPERVISOR_MODEL}'>"
