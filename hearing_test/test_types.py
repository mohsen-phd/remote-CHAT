from abc import ABC, abstractmethod

import nltk
from nltk.stem import WordNetLemmatizer

from audio_processing.noise import Babble, Noise, WhiteNoise
from stimuli_generator.questions import DigitQuestions


class TestTypes(ABC):
    """Abstract class for SIN test types. Each type of test must use this api."""

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
    def get_recording(self, configs: dict) -> str:
        """Get the recording for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            str: The recording src for the test.
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


class DIN(TestTypes):
    """Implementing the DIN test."""

    def __init__(self) -> None:
        """Initialize the DIN test."""
        self.digit_convertor = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
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

    def _get_lemmatizer(self) -> WordNetLemmatizer:
        """Load the NLTK lemmatizer.

        Returns:
            WordNetLemmatizer: WordNet Lemmatizer
        """
        nltk.download("wordnet")
        return WordNetLemmatizer()

    def get_recording(self, configs: dict) -> str:
        """Get the recording for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            str: The recording src for the test.
        """
        return configs["test"]["hearing-test"]["DIN"]["stimuli-recordings"]

    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI. and remove any common mistakes.

        Args:
            response (str): Response captured by Keyboard.

        Returns:
            list[str]: List of clean words.
        """
        return [self.digit_convertor[i] for i in response if i in self.digit_convertor]

    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR. and remove any common mistakes.

        Args:
            response (str): Response transcribed by ASR.

        Returns:
            list[str]: list of clean words.
        """
        responses_list = response.split(" ")
        lemmatized_response = [self._lemmatizer.lemmatize(x) for x in responses_list]

        return [
            self.common_mistakes[x] if x in self.common_mistakes else x
            for x in lemmatized_response
        ]

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

    def get_prepend_wav_file(self, configs: dict) -> tuple[str, int]:
        """Get the prepend wav file and its length. This is appended to the start of participant's response.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            tuple[str, int]: The wav file source and the length of the prepend string in number of words.
        """
        # todo: check this function, delete if not needed.
        wav_src = configs["test"]["hearing-test"]["DIN"]["Prepend_wav_file"]
        sentence_word_len = configs["test"]["hearing-test"]["DIN"]["prepend_str_len"]
        return wav_src, sentence_word_len


class FAAF(TestTypes):
    """Implementing the FAAF test."""

    # todo: implement FAAF
    pass


class ASL(TestTypes):
    """Implementing the ASL test."""

    # todo: implement ASL
    pass


class CHAT(TestTypes):
    """Implementing the CHAT test."""

    # todo: implement CHAT
    pass
