# ABOUTME: Core package initialization
# ABOUTME: Exports core components for Heathcliff assistant

from db.memory_manager import MemoryManager

from .agent_core import HeathcliffAgent

__all__ = ["MemoryManager", "AudioHandler", "HeathcliffAgent"]


def __getattr__(name: str):
    """Load optional audio support only when it is explicitly requested."""
    if name == "AudioHandler":
        from .audio_handler import AudioHandler

        return AudioHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
