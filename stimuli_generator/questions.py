"""Module for storing question and their validation method."""

import random
from abc import ABC, abstractmethod


class Questions(ABC):
    """Abstract class for questions. Each type of question must use this api."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        self.question = ""
        self.main_words = []

    @abstractmethod
    def check_answer(self, answer: str) -> bool:
        """Based on question type, check if the answer is correct or not.

        Args:
            answer (str): answer to the question given by the patient.

        Returns:
            bool: Is a match or not.
        """
        pass

    @abstractmethod
    def get_stimuli(self) -> list[str]:
        """Generate a sample stimuli.

        Returns:
            list[str]: stimuli for the question.
        """
        pass


class DigitQuestions(Questions):
    """Class for modeling digit-in-noise test questions."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        self.vocab_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        super().__init__()

    def get_stimuli(self) -> list[str]:
        """Generate a sample stimuli.

          Generate a sample stimuli consist of three words
          by randomly selecting from the list of vocab.

        Returns:
            list[str]: stimuli for the question.
        """
        self.main_words = random.sample(self.vocab_list, 3)
        self.question = "The number is " + " ".join(str(self.main_words))
        return self.main_words

    def check_answer(self, answer: list[str]) -> bool:
        """Check the given number is the same as the one  presented to the patient.

        Args:
            answer (list[str]): The patient's response.

        Returns:
            bool: Is a match or not.
        """
        answer = [item.lower() for item in answer]

        correct_words = []
        for word in answer:
            if word.lower() in self.main_words:
                correct_words.append(word)

        if correct_words == self.main_words:
            return True
        else:
            return False


# todo: Implement the ASL class
class ASL(Questions):
    """Class for modeling ASL test questions."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""

        super().__init__()

    def get_stimuli(self) -> list[str]:
        """Generate a sample stimuli.

        Generate a sample stimuli consist of three words
        by randomly selecting from the list of vocab.

        Returns:
            list[str]: stimuli for the question.
        """
        pass

    def check_answer(self, answer: list[str]) -> bool:
        """Check the given words are the same as the one presented to the patient.

        Args:
            answer (list[str]): The patient's response.

        Returns:
            bool: Is a match or not.
        """
        pass
