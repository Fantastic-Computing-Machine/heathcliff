# ABOUTME: Main entry point for Heathcliff voice assistant
# ABOUTME: Orchestrates audio, agent, and memory components

import argparse
import signal
import sys
import uuid
from typing import Any, Optional

from config import Config
from core.agent_core import HeathcliffAgent
from core.runtime.http_client import RuntimeV2HttpClient
from db.memory_manager import MemoryManager
from logger import logger
from utils.errors import AgentMemoryError


class HeathcliffAssistant:
    """Main orchestrator for the Heathcliff voice assistant."""

    def __init__(self, enable_audio: bool = True):
        """Initialize core components and optionally the voice stack."""
        logger.info("Starting Heathcliff Assistant...")

        # Initialize memory manager
        logger.info("Initializing memory manager...")
        try:
            self.memory = None if Config.RUNTIME_V2_ENABLED else MemoryManager()
        except AgentMemoryError as exc:
            logger.error(str(exc))
            print("Memory Not found, Heathcliff shutting down.")
            sys.exit(1)

        # Initialize agent (self-assembles all subagent + skill tools)
        logger.info("Initialising supervisor agent...")
        self.agent: Any
        if Config.RUNTIME_V2_ENABLED:
            self.agent = RuntimeV2HttpClient(Config.RUNTIME_V2_URL)
        else:
            self.agent = HeathcliffAgent(memory_manager=self.memory)

        self.audio: Optional[Any] = None
        if enable_audio:
            logger.info("Initializing audio handler...")
            from core.audio_handler import AudioHandler

            self.audio = AudioHandler(wake_word=Config.WAKE_WORD)

        # Initialize session
        self.conversation_id = str(uuid.uuid4())
        self.pending_approval: Optional[dict[str, Any]] = None
        logger.info(f"Conversation ID: {self.conversation_id}")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Heathcliff Assistant initialized successfully!")
        logger.info(f"Agent: {self.agent}")
        logger.info(f"Memory: {self.memory}")
        logger.info(f"Audio: {self.audio}")

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\nShutting down Heathcliff Assistant...")
        if self.audio is not None:
            self.audio.stop()
        logger.info("Goodbye!")
        sys.exit(0)

    def process_voice_input(self, text: str) -> str:
        """
        Process voice input through the agent.

        Args:
            text: Transcribed user speech

        Returns:
            Agent's response
        """
        try:
            logger.info(f"Processing: {text}")
            response = self.agent.invoke(text, conversation_id=self.conversation_id)
            logger.info(f"Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "I'm sorry, I encountered an error processing that request."

    def run_voice_mode(self):
        """Run in voice-activated mode."""
        if self.audio is None:
            raise RuntimeError("Voice mode was not initialized")
        logger.info(f"Starting voice mode. Say '{self.audio.wake_word}' to activate...")
        print(f"\n🎤 Heathcliff is listening for '{self.audio.wake_word}'...\n")

        # Run audio loop with agent callback
        self.audio.listen_loop(self.process_voice_input, continuous=True)

    def run_text_mode(self):
        """Run in text-only mode (for testing without audio)."""
        logger.info("Starting text mode...")
        print("\n💬 Heathcliff is ready! (Type 'quit' to exit)\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                response = self._process_text_input(user_input)
                print(f"Heathcliff: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in text mode: {e}")
                print(f"Error: {e}\n")

    def _process_text_input(self, user_input: str) -> str:
        """Process text input, including approval commands for pending actions."""
        if self.pending_approval is not None:
            decision = user_input.lower().strip()
            if decision in {"approve", "approved", "yes", "y", "sure"}:
                pending = self.pending_approval
                self.pending_approval = None
                return self.agent.resume_approval(
                    conversation_id=pending["session_id"],
                    user_input=pending["user_input"],
                    approved=True,
                )
            if decision in {"reject", "rejected", "no", "n", "cancel"}:
                pending = self.pending_approval
                self.pending_approval = None
                return self.agent.resume_approval(
                    conversation_id=pending["session_id"],
                    user_input=pending["user_input"],
                    approved=False,
                )
            return "This action is awaiting approval. Type 'approve' or 'reject'."

        response = "I encountered an error processing your request."
        for event in self.agent.stream_invoke(
            user_input, conversation_id=self.conversation_id
        ):
            event_type = event.get("type")
            if event_type == "approval_required":
                approval = dict(event.get("data") or {})
                approval["session_id"] = approval.get(
                    "session_id", self.conversation_id
                )
                approval["user_input"] = user_input
                self.pending_approval = approval
                return (
                    f"Approval required for {approval.get('tool_name', 'this action')}. "
                    "Type 'approve' (or 'sure') to proceed, or 'reject' to cancel."
                )
            if event_type == "response":
                response = event.get("data") or response
            elif event_type == "error":
                response = event.get("data") or response
        return response


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Heathcliff voice assistant")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text", "-t", action="store_true", help="run without audio")
    mode.add_argument("--voice", "-v", action="store_true", help="run with audio")
    args = parser.parse_args()

    # Text mode is the safe default; audio requires a configured microphone.
    try:
        assistant = HeathcliffAssistant(enable_audio=args.voice)

        if args.voice:
            assistant.run_voice_mode()
        else:
            assistant.run_text_mode()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
