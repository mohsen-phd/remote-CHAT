"""
This module defines various types of hearing tests, including DIN, ASL, FAAF, and CHAT.

Each test type inherits from the abstract base class TestTypes and implements specific methods.
"""

import string
from abc import ABC, abstractmethod
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

from audio_processing.noise import Babble, Noise, WhiteNoise
from audio_processing.util import convert_to_specific_db_spl, read_wav_file
from hearing_test.util import british_to_american, expand_contractions, lemmatizer
from stimuli_generator.questions import ASLQuestions, DigitQuestions


class TestTypes(ABC):
    """Abstract class for SIN test types. Each type of test must use this api."""

    def __init__(self, config: dict) -> None:
        """Initialize the TestTypes.

        Args:
            config (dict): configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI.

        Args:
            response (str): response given by participants.

        Returns:
            list[str]: list of words in the response. after post processing.
        """
        ...

    @abstractmethod
    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR.

        Args:
            response (str): response given by participants.

        Returns:
            list[str]: list of words in the response. after post processing.
        """
        ...

    @abstractmethod
    def get_noise(self, configs: dict) -> Noise:
        """Get the noise for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            str: The noise src for the test.
        """
        ...

    @abstractmethod
    def get_sound(self, stimuli: list[str]) -> np.ndarray:
        """Get the sound for the test.

        Args:
            stimuli (list[str]): List of stimuli.

        Returns:
            np.ndarray: Audio signal.
        """
        ...

    def _get_stimuli_src(self, test_name) -> Path:
        """Get the stimuli src for the test.

        Args:
            test_name (str): The name of the test.

        Returns:
            Path: The stimuli src for the test.

        Raises:
            NotImplementedError: If the stimuli vocalizer type is not supported.
        """
        if self.config["vocalization_mode"] == "tts":
            return Path(
                self.config["test"]["hearing-test"][test_name]["stimuli-recordings-tts"]
            )
        elif self.config["vocalization_mode"] == "recorded":
            return Path(
                self.config["test"]["hearing-test"][test_name][
                    "stimuli-recordings-natural"
                ]
            )
        else:
            raise NotImplementedError


class DIN(TestTypes):
    """Implementing the DIN test."""

    def __init__(self, config: dict) -> None:
        """Initialize the DIN test.

        Args:
            config (dict): configuration dictionary.
        """
        super().__init__(config)
        self.digit_convertor = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }

        self.common_mistakes = {
            "ate": "eight",
            "for": "four",
            "too": "two",
            "through": "two",
            "to": "two",
            "tree": "three",
            "sixth": "six",
            "seventy": "seven",
            "fifth": "five",
            "fourth": "four",
            "fly": "five",
            "tool": "two",
            "tune": "two",
            "tape": "eight",
            "won": "one",
            "tape": "eight",
            "won": "one",
            "fire": "five",
            "i": "one",
            "it": "eight",
            "seventh": "seven",
            "eighth": "eight",
            "ninth": "nine",
            "third": "three",
            "bore": "four",
            "bride": "five",
            "bribe": "five",
            "or": "four",
            "ford": "four",
            "aid": "eight",
            "o": "zero",
            "oh": "zero",
            "sex": "six",
            "age": "eight",
            "due": "two",
            "du": "two",
            "wait": "eight",
            "want": "one",
            "fay": "five",
            "fhi": "five",
            "run": "one",
            "sikh": "six",
            "seek": "six",
            "sick": "six",
            "sik": "six",
            "age": "eight",
            "fore": "four",
            "full": "four",
            "free": "three",
            "wom": "one",
            "h": "eight",
            "chu": "two",
            "fife": "five",
            "a": "eight",
            "set": "six",
            "ward": "one",
            "hit": "eight",
            "hate": "eight",
            "height": "eight",
            "ave": "eight",
            "true": "two",
            "fort": "four",
            "do": "two",
        }
        self._lemmatizer = self._get_lemmatizer()

        self.stimuli_generator = DigitQuestions()
        self.stimuli_src = self._get_stimuli_src("DIN")

    def _get_lemmatizer(self) -> WordNetLemmatizer:
        """Load the NLTK lemmatizer.

        Returns:
            WordNetLemmatizer: WordNet Lemmatizer
        """
        nltk.download("wordnet")
        return WordNetLemmatizer()

    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI. and remove any common mistakes.

        Args:
            response (str): Response captured by Keyboard.

        Returns:
            list[str]: List of clean words.
        """
        response_list = list(response)
        return response_list[:3]

    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR. and remove any common mistakes.

        Args:
            response (str): Response transcribed by ASR.

        Returns:
            list[str]: list of clean words.
        """
        responses_list = response.lower().split(" ")
        lemmatized_response = [self._lemmatizer.lemmatize(x) for x in responses_list]
        removed_common_mistakes = [
            self.common_mistakes[x] if x in self.common_mistakes else x
            for x in lemmatized_response
        ]
        converted_to_digits = [self.digit_convertor[x] for x in removed_common_mistakes]
        return converted_to_digits

    def get_noise(self, configs: dict) -> Noise:
        """Get the noise for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            Noise: Return the proper noise based on config file.

        Raises:
            NotImplementedError: If the noise type is not supported.
        """
        if configs["test"]["hearing-test"]["DIN"]["noise"]["type"] == "white":
            return WhiteNoise()
        elif configs["test"]["hearing-test"]["DIN"]["noise"]["type"] == "babble":
            return Babble(
                noise_src=configs["test"]["hearing-test"]["DIN"]["noise"]["src"]
            )
        raise NotImplementedError

    def get_sound(self, stimuli: list[int]) -> np.ndarray:
        """Use the recorded audio to generate the waveform.

        Args:
            stimuli (list[str]): The generated stimuli as a list of string.

        Returns:
            np.ndarray: Audio signal.
        """
        full_audio = np.array([])
        for digit in stimuli:
            _, digit_audio = read_wav_file(self.stimuli_src / f"{digit}.wav")
            full_audio = np.concatenate((full_audio, digit_audio))
        full_audio = convert_to_specific_db_spl(full_audio, 65)
        return full_audio


class ASL(TestTypes):
    """Implementing the ASL test."""

    def __init__(self, config: dict) -> None:
        """Initialize the DIN test.

        Args:
            config (dict): configuration dictionary.
        """
        super().__init__(config)

        self.stimuli_generator = ASLQuestions()
        self.stimuli_src = self._get_stimuli_src("ASL")

    def get_sound(self, stimuli: list[int]) -> np.ndarray:
        """Use the recorded audio to generate the waveform.

        Args:
            stimuli (list[str]): The generated stimuli as a list of string.

        Returns:
            np.ndarray: Audio signal.
        """
        stimulus_id = stimuli[0]

        _, digit_audio = read_wav_file(self.stimuli_src / f"{stimulus_id}.wav")
        converted_audio = convert_to_specific_db_spl(digit_audio, 65)
        return converted_audio

    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI. and remove any common mistakes.

        Args:
            response (str): Response captured by Keyboard.

        Returns:
            list[str]: List of clean words.
        """
        return list(response)[:1]

    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR. and remove any common mistakes.

        Args:
            response (str): Response transcribed by ASR.

        Returns:
            list[str]: list of clean words.
        """
        # Convert to lowercase
        response = response.lower()
        response = expand_contractions(response)
        # Remove punctuation
        response = response.translate(str.maketrans("", "", string.punctuation))
        response = "".join(
            [char for char in response if char.isalpha() or char.isspace()]
        )
        response = british_to_american(response)

        response = response.replace("'s", " is ")
        response = response.replace("'re", " are ")

        response = response.strip()

        # Tokenize
        clean_response = lemmatizer(response)

        return [clean_response]

    # todo:update for ASL
    def get_noise(self, configs: dict) -> Noise:
        """Get the noise for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            Noise: Return the proper noise based on config file.

        Raises:
            NotImplementedError: If the noise type is not supported.
        """
        if configs["test"]["hearing-test"]["DIN"]["noise"]["type"] == "white":
            return WhiteNoise()
        elif configs["test"]["hearing-test"]["DIN"]["noise"]["type"] == "babble":
            return Babble(
                noise_src=configs["test"]["hearing-test"]["DIN"]["noise"]["src"]
            )
        raise NotImplementedError


class FAAF(TestTypes):
    """Implementing the FAAF test."""

    # todo: implement FAAF
    pass


class CHAT(TestTypes):
    """Implementing the CHAT test."""

    # todo: implement CHAT
    pass
