# ABOUTME: Class-based configuration with attribute access and cached singleton

import os
import sys
import tomllib
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from logger import logger

load_dotenv(".env")

MASTER_INFO_LOC = ".data/master_info.toml"


def _ai_api_key() -> Optional[str]:
    """Return the provider-neutral AI key, accepting legacy names during migration."""
    return (
        os.getenv("AI_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


class MasterConf:
    MASTER_INFO = {}
    TZ = "America/New_York"

    @classmethod
    def _read_master_info(cls) -> None:
        try:
            with open(MASTER_INFO_LOC, "rb") as f:
                loaded_info = tomllib.load(f)
        except FileNotFoundError:
            logger.error(f"Master info file not found: {MASTER_INFO_LOC}")
            sys.exit(1)
        except tomllib.TOMLDecodeError as exc:
            logger.error(
                f"Master info file appears to be corrupted at ({MASTER_INFO_LOC}): {exc}"
            )
            sys.exit(1)

        if loaded_info is None or loaded_info == {}:
            logger.error(f"Master info is empty in {MASTER_INFO_LOC}")
            sys.exit(1)

        if not isinstance(loaded_info, dict):
            logger.error(
                f"Master info TOML must decode to a dictionary in {MASTER_INFO_LOC}"
            )
            sys.exit(1)

        cls.MASTER_INFO = loaded_info


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
    SUPERVISOR_MODEL = "google_genai:gemini-3.1-flash-lite-preview"
    SUBAGENT_MODEL = "google_genai:gemini-3.1-flash-lite-preview"
    # Specialists default to the same low-quota model as Heathcliff. Override
    # this explicitly when a task needs a more capable provider/model.
    TOOL_MODEL = os.getenv("TOOL_MODEL", "google_genai:gemini-3.1-flash-lite-preview")
    # Supervisor Hyperparameters
    TEMPERATURE = 0.5
    MAX_TOKENS = 8192
    TOP_P = 0.7
    MAX_ITERATIONS = 20
    TIMEOUT_SECONDS = 300
    MAX_RETRIES = 3


class PlatformConf:
    # Provider-neutral AI key. Legacy Gemini names are accepted by _ai_api_key.
    AI_KEY = _ai_api_key()
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", "credentials.json"
    )
    # Google Search
    GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    # Tavily's agent-oriented web search and extraction API.
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    @classmethod
    def get_ai_api_key(cls) -> str:
        """Return the configured provider-neutral AI key."""
        if not cls.AI_KEY:
            raise ValueError("AI_KEY is not set in the environment.")
        return cls.AI_KEY

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

    _llm_config: Dict[str, Any] = {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.5-flash-lite",
            "temperature": 0.2,
            "api_key": PlatformConf.get_ai_api_key(),
        },
    }
    _embedder_config: Dict[str, Any] = {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-001",
            "api_key": PlatformConf.get_ai_api_key(),
        },
    }

    _vector_store_config: Dict[str, Any] = {
        "provider": "chroma",
        "config": {
            "collection_name": "heathcliff_memories",
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
    LANGFUSE_USER_ID = "heathcliff_user"
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


class InfoAgentConfig:
    INFO_RECURSION_LIMIT = int(os.getenv("INFO_RECURSION_LIMIT", "45"))


class RecentContextConfig:
    # Recent context snippet store
    RECENT_CONTEXT_TTL_SECONDS = int(os.getenv("RECENT_CONTEXT_TTL_SECONDS", "7200"))
    RECENT_CONTEXT_MAX_ITEMS = int(os.getenv("RECENT_CONTEXT_MAX_ITEMS", "100"))
    RECENT_CONTEXT_MAX_SNIPPET_CHARS = int(
        os.getenv("RECENT_CONTEXT_MAX_SNIPPET_CHARS", "1200")
    )
    RECENT_CONTEXT_MAX_RETURN = int(os.getenv("RECENT_CONTEXT_MAX_RETURN", "5"))
    RECENT_CONTEXT_STORE_PATH = os.getenv(
        "RECENT_CONTEXT_STORE_PATH", "temp/recent_memory.json"
    )


class AudioConfig:
    WAKE_WORD = "heathcliff"
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512
    TTS_RATE = 175
    TTS_VOLUME = 0.9
    TTS_VOICE = None


class CoordinatorConf:
    """Budget and execution limits for the coordinator graph."""

    MAX_TASKS_PER_REQUEST = int(os.getenv("COORDINATOR_MAX_TASKS", "10"))
    PER_TASK_TIMEOUT_MS = int(os.getenv("COORDINATOR_TASK_TIMEOUT_MS", "60000"))
    MAX_TOTAL_RUNTIME_MS = int(os.getenv("COORDINATOR_MAX_RUNTIME_MS", "300000"))


class Conf(
    RuntimeConf,
    MasterConf,
    ChromaConf,
    Mem0Conf,
    PlatformConf,
    LangFuseConf,
    NewsConfig,
    WeatherConfig,
    InfoAgentConfig,
    RecentContextConfig,
    AudioConfig,
    CoordinatorConf,
):
    _instance: Optional["Conf"] = None

    def __new__(cls) -> "Conf":
        if cls._instance is None:
            cls._instance = super(Conf, cls).__new__(cls)
            cls._read_master_info()
        return cls._instance

    @classmethod
    def validate(cls) -> None:
        missing_keys = []
        if not cls.AI_KEY:
            missing_keys.append("AI_KEY")
        if not cls.CHROMA_API_KEY:
            missing_keys.append("CHROMA_API_KEY")
        if not cls.CHROMA_TENANT:
            missing_keys.append("CHROMA_TENANT")
        if not cls.CHROMA_DATABASE:
            missing_keys.append("CHROMA_DATABASE")

        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

        logger.info("Configuration validated successfully!")

    def __repr__(self) -> str:
        return f"<Config wake_word='{self.WAKE_WORD}' model='{self.SUPERVISOR_MODEL}'>"
