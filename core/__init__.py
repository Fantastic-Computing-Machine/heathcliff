# ABOUTME: Core package initialization
# ABOUTME: Exports core components for Heathcliff assistant

from .agent_core import HeathcliffAgent
from .memory_manager import MemoryManager

# AudioHandler has optional dependencies (pyaudio, porcupine) that may not be installed
try:
    from .audio_handler import AudioHandler

    __all__ = ["MemoryManager", "AudioHandler", "HeathcliffAgent"]
except ImportError:
    AudioHandler = None
    __all__ = ["MemoryManager", "HeathcliffAgent"]
