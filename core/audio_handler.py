# ABOUTME: Audio handling for wake word detection, speech-to-text, and text-to-speech
# ABOUTME: Uses OpenWakeWord for wake word, Google Speech Recognition for STT, and pyttsx3 for TTS

import threading
from typing import Callable, Optional

import numpy as np
import openwakeword
import pyaudio
import pyttsx3
import speech_recognition as sr
from openwakeword.model import Model as WakeWordModel

from config import Config
from logger import logger

# Models will be downloaded on initialization if needed


class AudioHandler:
    """
    Handles all audio I/O for Heathcliff:
    - Wake word detection using OpenWakeWord
    - Speech-to-text using Google Speech Recognition
    - Text-to-speech using pyttsx3
    """

    # OpenWakeWord audio settings
    SAMPLE_RATE = 16000
    FRAME_SIZE = 1280  # 80ms chunks recommended by openwakeword

    def __init__(self, wake_word: str = "hey_jarvis"):
        """
        Initialize audio components.

        Args:
            wake_word: Wake word model to use (default: "hey_jarvis")
                      Available models: alexa, hey_mycroft, hey_jarvis, etc.
        """
        self.config = Config
        self.wake_word = wake_word

        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Text-to-Speech
        self.tts_engine = pyttsx3.init()
        self._configure_tts()

        # Wake word detection (OpenWakeWord)
        self.oww_model = None
        self.audio_stream = None
        self.pa = None
        self._init_openwakeword()

        # Detection threshold
        self.threshold = 0.5

        # State
        self.listening = False
        self.listen_thread = None

    def _configure_tts(self):
        """Configure TTS engine with settings from config."""
        rate = getattr(self.config, "TTS_RATE", 175)
        volume = getattr(self.config, "TTS_VOLUME", 0.9)
        voice = getattr(self.config, "TTS_VOICE", None)

        self.tts_engine.setProperty("rate", rate)
        self.tts_engine.setProperty("volume", volume)

        if voice:
            voices = self.tts_engine.getProperty("voices")
            for v in voices:
                if voice.lower() in v.name.lower():
                    self.tts_engine.setProperty("voice", v.id)
                    break

    def _init_openwakeword(self):
        """Initialize OpenWakeWord wake word detector."""
        try:
            # Ensure models are available (lazy download)
            openwakeword.utils.download_models()

            # Validate wake word model and fallback if needed
            model_paths = openwakeword.get_pretrained_model_paths() or {}
            available_models = (
                list(model_paths)
                if isinstance(model_paths, dict)
                else list(model_paths)
            )
            wake_word_model = self.wake_word
            if (
                wake_word_model not in available_models
                and f"{wake_word_model}.onnx" not in available_models
            ):
                logger.warning(
                    "Wake word '%s' not found in available OpenWakeWord models; falling back to 'hey_jarvis'",
                    wake_word_model,
                )
                wake_word_model = "hey_jarvis"

            # Load the wake word model
            self.oww_model = WakeWordModel(
                wakeword_models=[wake_word_model],
                inference_framework="onnx",
            )

            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.FRAME_SIZE,
            )
        except Exception as e:
            print(f"Error initializing OpenWakeWord: {e}")
            print("Wake word detection will not be available")

    def listen_for_wake_word(self) -> bool:
        """
        Listen for wake word in audio stream.

        Returns:
            True if wake word detected, False otherwise
        """
        if not self.oww_model or not self.audio_stream:
            return False

        try:
            # Read audio frame
            audio_data = self.audio_stream.read(
                self.FRAME_SIZE, exception_on_overflow=False
            )
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Get predictions
            predictions = self.oww_model.predict(audio_array)

            # Check if any wake word exceeds threshold
            for model_name, score in predictions.items():
                if score >= self.threshold:
                    return True

            return False
        except Exception as e:
            print(f"Error detecting wake word: {e}")
            return False

    def speech_to_text(
        self, timeout: int = 5, phrase_time_limit: int = 10
    ) -> Optional[str]:
        """
        Capture audio from microphone and convert to text.

        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for the phrase

        Returns:
            Recognized text or None if recognition failed
        """
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for audio
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

                # Recognize speech using Google Speech Recognition
                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                return text

        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

    def text_to_speech(self, text: str, blocking: bool = True):
        """
        Convert text to speech and play it.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        try:
            if blocking:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                # Non-blocking TTS
                threading.Thread(
                    target=self._speak_async, args=(text,), daemon=True
                ).start()
        except Exception as e:
            print(f"TTS error: {e}")

    def _speak_async(self, text: str):
        """Helper method for async TTS."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen_loop(self, callback: Callable[[str], str], continuous: bool = True):
        """
        Main loop: listen for wake word → capture speech → process → speak response.

        Args:
            callback: Function that takes user input and returns response
            continuous: If True, keep listening after each interaction
        """
        print(f"Listening for wake word: '{self.wake_word}'...")
        self.listening = True

        try:
            while self.listening:
                # Wait for wake word
                if self.listen_for_wake_word():
                    print(f"Wake word '{self.wake_word}' detected!")
                    self.text_to_speech("Yes?", blocking=True)

                    # Capture user speech
                    user_input = self.speech_to_text()

                    if user_input:
                        print(f"User: {user_input}")

                        # Process input through callback
                        response = callback(user_input)

                        if response:
                            print(f"Assistant: {response}")
                            self.text_to_speech(response, blocking=True)

                    if not continuous:
                        break

        except KeyboardInterrupt:
            print("\nStopping listener...")
        finally:
            self.stop()

    def start_background_listener(self, callback: Callable[[str], str]):
        """
        Start wake word listener in background thread.

        Args:
            callback: Function that takes user input and returns response
        """
        if self.listen_thread and self.listen_thread.is_alive():
            print("Listener already running")
            return

        self.listen_thread = threading.Thread(
            target=self.listen_loop, args=(callback,), daemon=True
        )
        self.listen_thread.start()
        print("Background listener started")

    def stop(self):
        """Stop listening and clean up resources."""
        self.listening = False

        if self.audio_stream:
            self.audio_stream.close()

        if self.pa:
            self.pa.terminate()

        # OpenWakeWord models don't require explicit cleanup

        print("Audio handler stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

    def __repr__(self) -> str:
        """String representation."""
        return f"<AudioHandler wake_word='{self.wake_word}' listening={self.listening}>"
