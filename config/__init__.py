# ABOUTME: Config package initialization
# ABOUTME: Exports Conf class and singleton Config instance

from config.config import Conf

Config = Conf()

__all__ = ["Config", "Conf"]
