# ABOUTME: Config package initialization
# ABOUTME: Exports Config class and get_config function

from .config_loader import Config, get_config

__all__ = ["Config", "get_config"]
