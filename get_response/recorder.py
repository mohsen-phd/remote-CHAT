"""Module for recording voice."""

import math
import os
import struct
import time
import wave

import pyaudio
from loguru import logger

SHORT_NORMALIZE = (
    1.0 / 32768.0
)  # Normalize the sound, the value is because of the Int16 range -32768 to +32767.

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLING_RATE = 44100
sample_width = 2


class Recorder:
    """Record sound when there is a noise."""

    @staticmethod
    def rms(frame: bytes) -> float:
        """Calculate RMS of the input signal.

        Convert the input bytes to a list of values and
        calculate the root mean square error of the input signal.

        Args:
            frame (bytes): input stream as bytes

        Returns:
            float: rms of the input signal.
        """
        count = len(frame) / sample_width
        count_format = "%dh" % (count)
        shorts = struct.unpack(count_format, frame)
        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)
        return rms * 1000

    def __init__(
        self,
        store: bool,
        chunk: int,
        rms_threshold: int,
        timeout_length: int,
        save_dir: str,
    ):
        """Initialize the PyAudio stream for voice recording.

        Args:
            store (bool): Store the recorded file or not.
            chunk (int): How much data to read from microphone stream.
            rms_threshold (int): Threshold for detecting the presence of the sound.
            timeout_length (int): How long to wait if there is no sound.
            save_dir (str): Where to save the recorded sound
        """
        self.timeout_length = timeout_length
        self.store = store
        self.p = pyaudio.PyAudio()
        self.chunk = chunk
        self.rms_threshold = rms_threshold
        self.save_dir = save_dir

    def record(self) -> str:
        """Listen to microphone and record as long as sound is present.

        Returns:
            str: Address of saved file.
        """
        logger.debug("Noise detected, recording beginning")
        rec = []
        current = time.time()
        end = time.time() + self.timeout_length
        recording_max_length = 10
        max_len_timer = time.time()
        while current <= end and time.time() < recording_max_length + max_len_timer:
            data = self.stream.read(self.chunk)
            if self.rms(data) >= self.rms_threshold:
                end = time.time() + self.timeout_length
            current = time.time()
            rec.append(data)
        sound = b"".join(rec)
        address = ""
        if self.store:
            address = self.write(sound)
        return address

    def write(self, recording: bytes) -> str:
        """Write the recorded sound as a WAV file.

        Args:
            recording (bytes): Recorded sound

        Returns:
            str: Address of saved file.
        """
        n_files = len(os.listdir(self.save_dir))
        filename = os.path.join(self.save_dir, "{}.wav".format(n_files))
        wf = wave.open(filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLING_RATE)
        wf.writeframes(recording)
        wf.close()
        logger.debug("Written to file: {}".format(filename))
        return filename

    def listen(self) -> str:
        """Listen for the presence of a sound, and record the sound until it stop.

        Returns:
            str: Address of recorded file.
        """
        logger.debug("Start Talking")
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLING_RATE,
            input=True,
            output=True,
            frames_per_buffer=self.chunk,
        )

        file_address = self.record()

        return file_address
