class HeathcliffException(Exception):
    """Base class for all custom exceptions in Heathcliff."""

    pass


class AgentMemoryError(HeathcliffException):
    """Raised when an operation in the memory manager fails."""

    pass


class ToolExecutionError(HeathcliffException):
    """Raised when an external tool fails to execute."""

    pass


class VoiceServiceError(HeathcliffException):
    """Raised when the voice layer (STT/TTS) encounters an error."""

    pass


class AgentInitializationError(HeathcliffException):
    """Raised when the agent fails to initialize properly."""

    pass
