"""Convert text to speech."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from speechbrain.pretrained import HIFIGAN, Tacotron2

from audio_processing.util import convert_to_specific_db_spl


class Vocalizer(ABC):
    """Interface for the TTS system."""

    @abstractmethod
    def get_sound(self, text: str) -> np.ndarray:
        """Get a text and generate the corresponding sound with level of 65 dB SPL.

        Args:
            text (str): input string

        Returns:
            np.ndarray: waveform in shape of (1,length_of_wave)
        """


class Recorded(Vocalizer):
    """Class for generating waveform from recorded audio."""

    def __init__(self, src: Path) -> None:
        """
        Initialize a new instance of the Vocalizer class.

        Args:
            src (Path): The path to the recorded audio file.
        """
        self.src = src

    def get_sound(self, text: str) -> np.ndarray:
        """Use the recorded audio to generate the waveform.

        Args:
            text (str): The generated stimuli as an string.

        Returns:
            np.ndarray: Audio signal.
        """
        digits = self._extract_numbers(text)
        full_audio = np.array([])
        for digit in digits:
            _, digit_audio = wavfile.read(self.src / f"{digit}.wav")
            full_audio = np.concatenate((full_audio, digit_audio))
        full_audio = convert_to_specific_db_spl(full_audio, 65)
        return full_audio

    def _extract_numbers(self, text: str) -> list[str]:
        """Extract the numbers from the stimuli.

        Args:
            text (str): Input stimuli as a string.

        Returns:
            list[str]: List of numbers in string format
        """
        split_text = text.split(" ")
        return split_text[-3:]


class TTS(Vocalizer):
    """Class for generating waveform from string.

    From Hugging Face repo: https://huggingface.co/speechbrain/tts-hifigan-ljspeech
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the class and load  tacotron2 and hifi_gan.

        Args:
            device (str): device to run the operations on
                (cpu,cuda,mps). Defaults to "cpu".
        """
        self.tacotron2 = Tacotron2.from_hparams(
            source="speechbrain/tts-tacotron2-ljspeech",
            savedir="models/tmpdir_tts",
            run_opts={"device": device},
        )
        self.hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="models/tmpdir_vocoder",
            run_opts={"device": device},
        )

    def get_sound(self, text: str) -> np.ndarray:
        """Get a text and generate the corresponding sound with level of 65 dB SPL.

        Args:
            text (str): input string

        Returns:
            np.ndarray: waveform in shape of (1,length_of_wave)
        """
        mel_output, mel_length, alignment = self.tacotron2.encode_text(text + " ")

        # Running Vocoder (spectrogram-to-waveform)
        waveforms = self.hifi_gan.decode_batch(mel_output)
        sound = waveforms.to("cpu").squeeze(1).numpy()
        sound = convert_to_specific_db_spl(sound, 65)
        return sound.squeeze(0)
