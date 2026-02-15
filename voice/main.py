# ABOUTME: Voice listener module for wake word detection
# ABOUTME: Uses OpenWakeWord for local wake word detection

import threading

import numpy as np
import openwakeword
import pyaudio
import speech_recognition as sr
from openwakeword.model import Model as WakeWordModel


class VoiceListener:
    """Voice listener with OpenWakeWord wake word detection."""

    # OpenWakeWord audio settings
    SAMPLE_RATE = 16000
    FRAME_SIZE = 1280  # 80ms chunks recommended by openwakeword

    def __init__(self, wake_word="hey_jarvis", callback=None):
        """
        Initialize voice listener.

        Args:
            wake_word: Wake word model to use (e.g., "hey_jarvis", "alexa")
            callback: Function to call when command is recognized
        """
        self.wake_word = wake_word
        self.callback = callback
        self.oww_model = None
        self.pa = None
        self.audio_stream = None
        self.is_listening = False
        self.threshold = 0.5

    def start(self):
        """Start the voice listener."""
        # Ensure models are available lazily
        openwakeword.utils.download_models()

        # Validate wake word; fallback to a known model if missing
        model_paths = openwakeword.get_pretrained_model_paths() or {}
        available_models = (
            list(model_paths) if isinstance(model_paths, dict) else list(model_paths)
        )
        wake_word_model = self.wake_word
        if (
            wake_word_model not in available_models
            and f"{wake_word_model}.onnx" not in available_models
        ):
            wake_word_model = "hey_jarvis"

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
        self.is_listening = True

        threading.Thread(target=self._listen_loop).start()

    def _listen_loop(self):
        """Main listening loop for wake word detection."""
        while self.is_listening:
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
                    # Wake word detected
                    self._process_command()
                    break

    def _process_command(self):
        """Process a voice command after wake word detection."""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening for command...")
            audio = r.listen(source)

            try:
                command = r.recognize_google(audio)
                if self.callback:
                    self.callback(command)
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError:
                print("Could not request results")

    def stop(self):
        """Stop the voice listener and clean up resources."""
        self.is_listening = False
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        # OpenWakeWord models don't require explicit cleanup
