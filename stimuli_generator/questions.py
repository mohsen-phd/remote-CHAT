"""Module for storing question and their validation method."""

import json
import random
from abc import ABC, abstractmethod


class Questions(ABC):
    """Abstract class for questions. Each type of question must use this api."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        self.question = ""
        self.main_words = []
        self.question_id = []

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
    def get_stimuli(self) -> tuple[list[str], list[str]]:
        """Generate a sample stimuli.

        Returns:
            tuple[list[str], list[str]]: stimuli ID and the main words in the stimuli.
        """
        pass


class DigitQuestions(Questions):
    """Class for modeling digit-in-noise test questions."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        self.vocab_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        super().__init__()

    def get_stimuli(self) -> tuple[list[str], list[str]]:
        """Generate a sample stimuli.

          Generate a sample stimuli consist of three words
          by randomly selecting from the list of vocab.

        Returns:
            tuple[list[str], list[str]]: stimuli ID and the main words in the stimuli.
        """
        self.main_words = random.sample(self.vocab_list, 3)
        self.question = "The number is " + " ".join(str(self.main_words))
        self.question_id = self.main_words
        return self.question_id, self.main_words

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
class ASLQuestions(Questions):
    """Class for modeling ASL test questions."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        super().__init__()
        self.stimuli_list = self._read_asl_stimuli()

    def _read_asl_stimuli(self) -> dict:
        """Read the ASL stimuli from the a json file.

        Returns:
            dict: The stimuli.
        """
        with open("media/ASL/sentences.json") as f:
            stimuli = json.load(f)
        return stimuli

    def get_stimuli(self) -> tuple[list[str], list[str]]:
        """Generate a sample stimuli.

        Generate a sample stimuli consist of three words
        by randomly selecting from the list of vocab.

        Returns:
            tuple[list[str], list[str]]: stimuli ID and the main words in the stimuli.
        """
        set_num = random.randint(1, 18)
        question_num = random.randint(1, 15)
        full_question_id = f"{set_num}-{question_num}"
        stimulus = self.stimuli_list[full_question_id]
        self.question = stimulus["text"]
        self.main_words = [
            self.question.split()[idx].lower() for idx in stimulus["keywords"]
        ]
        self.question_id = full_question_id
        return [full_question_id], self.main_words

    def check_answer(self, answer: list[str]) -> bool:
        """Check the given words are the same as the one presented to the patient.

        Args:
            answer (list[str]): The patient's response.

        Returns:
            bool: Is a match or not.
        """
        pass
