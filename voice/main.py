import speech_recognition as sr
import pvporcupine
import pyaudio
import struct
import threading


class VoiceListener:
    def __init__(self, wake_word="heathcliff", callback=None):
        self.wake_word = wake_word
        self.callback = callback
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        self.is_listening = False

    def start(self):
        self.porcupine = pvporcupine.create(keywords=[self.wake_word])
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )
        self.is_listening = True

        threading.Thread(target=self._listen_loop).start()

    def _listen_loop(self):
        while self.is_listening:
            pcm = struct.unpack_from(
                "h" * self.porcupine.frame_length,
                self.audio_stream.read(self.porcupine.frame_length),
            )
            keyword_index = self.porcupine.process(pcm)

            if keyword_index >= 0:
                # Wake word detected
                self._process_command()

    def _process_command(self):
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
        self.is_listening = False
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
