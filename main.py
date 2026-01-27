# ABOUTME: Main entry point for Heathcliff voice assistant
# ABOUTME: Orchestrates audio, agent, and memory components

import signal
import sys
import uuid

from config import Config
from core.agent_core import HeathcliffAgent
from core.audio_handler import AudioHandler
from logger import logger
from tools import get_all_tools


class HeathcliffAssistant:
    """Main orchestrator for the Heathcliff voice assistant."""

    def __init__(self):
        """Initialize all components."""
        logger.info("Starting Heathcliff Assistant...")

        # Initialize agent (auto-creates memory manager and loads all tools)
        logger.info("Initializing agent...")
        try:
            self.agent = HeathcliffAgent.create()
        except Exception as exc:
            logger.error(f"Agent initialization failed: {exc}")
            print("Agent initialization failed, Heathcliff shutting down.")
            sys.exit(1)

        # Initialize audio handler
        logger.info("Initializing audio handler...")
        wake_word = Config.WAKE_WORD
        self.audio = AudioHandler(wake_word=wake_word)

        # Initialize session
        self.session_id = str(uuid.uuid4())
        logger.info(f"Session ID: {self.session_id}")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Heathcliff Assistant initialized successfully!")
        logger.info(f"Agent: {self.agent}")
        logger.info(f"Audio: {self.audio}")

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\nShutting down Heathcliff Assistant...")
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
            response = self.agent.invoke(text, session_id=self.session_id)
            logger.info(f"Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "I'm sorry, I encountered an error processing that request."

    def run_voice_mode(self):
        """Run in voice-activated mode."""
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

                response = self.process_voice_input(user_input)
                print(f"Heathcliff: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in text mode: {e}")
                print(f"Error: {e}\n")


def main():
    """Main entry point."""
    # Parse command line arguments
    mode = "voice"  # Default mode

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Support both --text and mode=text formats
        if arg in ["--text", "-t"] or arg == "mode=text":
            mode = "text"
        elif arg in ["--voice", "-v"] or arg == "mode=voice":
            mode = "voice"
        elif arg.startswith("mode="):
            # Parse mode=value format
            mode_value = arg.split("=", 1)[1].lower()
            if mode_value in ["text", "voice"]:
                mode = mode_value
            else:
                print(f"Error: Invalid mode '{mode_value}'. Must be 'text' or 'voice'.")
                sys.exit(1)
        elif arg in ["--help", "-h"]:
            print(
                """
Heathcliff Voice Assistant

Usage:
    python main.py                  Run in voice mode (default)
    python main.py "mode=text"      Run in text mode (MUST use quotes!)
    python main.py "mode=voice"     Run in voice mode (MUST use quotes!)
    python main.py --text           Run in text mode (recommended)
    python main.py --voice          Run in voice mode
    python main.py --help           Show this help message

Note: The mode=text format requires quotes because bash interprets
      unquoted mode=text as an environment variable assignment.
      We recommend using --text instead for simplicity.

Voice Mode:
    - Say the wake word 'heathcliff' to activate
    - Speak your command
    - Heathcliff will respond via text-to-speech

Text Mode:
    - Type your commands
    - Responses are printed to console
    - Useful for testing without audio hardware
            """
            )
            sys.exit(0)
        else:
            print(f"Error: Unknown argument '{arg}'. Use --help for usage information.")
            sys.exit(1)

    # Create and run assistant
    try:
        assistant = HeathcliffAssistant()

        if mode == "text":
            assistant.run_text_mode()
        else:
            assistant.run_voice_mode()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
