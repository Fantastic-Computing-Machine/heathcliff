# ABOUTME: Centralized logging configuration for Heathcliff
# ABOUTME: Supports console and file logging with environment variable control

import os
import sys
from pathlib import Path
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get logger
logger = getLogger("heathcliff")

# Determine log level from environment
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = DEBUG if log_level_str == "DEBUG" else INFO

logger.setLevel(log_level)

# Console handler - always enabled
console_handler = StreamHandler(sys.stdout)
console_handler.setLevel(log_level)
console_formatter = Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler - controlled by environment variable
enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"

if enable_file_logging:
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "debug.log"

    file_handler = FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(DEBUG)  # File always gets DEBUG level
    file_formatter = Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"📝 File logging enabled → {log_file.absolute()}")

# Prevent propagation to root logger
logger.propagate = False
