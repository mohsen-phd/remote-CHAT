"""Module for storing question and their validation method."""

import json
import os
import random
from abc import ABC, abstractmethod
import re
from openai import OpenAI
from hearing_test.util import expand_contractions, lemmatizer, remove_contractions


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
    def get_stimuli(self) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

        Returns:
            tuple[list[str], list[str],str]: stimuli ID and the main words in the stimuli.
        """
        pass

    def _read_json(self, src: str) -> dict:
        """Read the json file and return the data.

        Args:
            src (str): json file location

        Returns:
            dict: data in the json file
        """
        with open(src) as f:
            data = json.load(f)
        return data


class DigitQuestions(Questions):
    """Class for modeling digit-in-noise test questions."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        self.vocab_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        super().__init__()

    def get_stimuli(self) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

          Generate a sample stimuli consist of three words
          by randomly selecting from the list of vocab.

        Returns:
            tuple[list[str], list[str]]: stimuli ID and the main words in the stimuli. The prompt to show the user when
            asking for a response.
        """
        self.main_words = random.sample(self.vocab_list, 3)
        self.question = "The number is " + " ".join(str(self.main_words))
        self.question_id = self.main_words

        response_getting_prompt = "Please enter the digits you heard."

        return self.question_id, self.main_words, response_getting_prompt

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


class ASLQuestions(Questions):
    """Class for modeling ASL test questions."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        super().__init__()
        self.stimuli_list = self._read_json("media/ASL/sentences.json")

    def get_stimuli(self) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

        Generate a sample stimuli consist of three words
        by randomly selecting from the list of vocab.

        Returns:
            tuple[list[str], list[str],str]: stimuli ID and the main words in the stimuli. The prompt to show the user when
            asking for a response.
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
        response_getting_prompt = "Please repeat the sentence you heard."
        return [full_question_id], self.main_words, response_getting_prompt

    def check_answer(self, answer: list[str]) -> bool:
        """Check the given words are the same as the one presented to the patient.

        Args:
            answer (list[str]): The patient's response.

        Returns:
            bool: Is a match or not.
        """
        if answer == ["0"]:
            return False
        elif answer == ["1"]:
            return True
        else:
            return self._check_asr_answer(answer[0])

    def _check_asr_answer(self, answer: str) -> bool:
        """Check the asr output and see if the given answer was correct.

        Args:
            answer (str): The patient's response, transcribed by ASR.

        Returns:
            bool: Is a match or not.
        """
        answer_without_space = re.sub(r"\s+", "", answer)

        correct_count = 0
        processed_main_words = lemmatizer(
            remove_contractions(" ".join(self.main_words))
        ).split(" ")
        for word in processed_main_words:
            if word in answer_without_space:
                correct_count += 1

        if correct_count >= (len(self.main_words) - 1):
            return True
        else:
            return False


class CHATQuestions(Questions):
    """Abstract class for questions. Each type of question must use this api."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        super().__init__()
        self.stimuli_list = self._read_chat_json("media/CHAT/text")
        self.chatGPT = OpenAI(api_key="...")

    def _read_chat_json(self, root_src: str) -> dict:
        """Read all the json file in the root and marge them into one dictionary.

        Args:
            root_src (str): src of root directory.

        Returns:
            dict: Merged json file.
        """
        data = {}
        for root, _, files in os.walk(root_src):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    cat_id = file.split("-")[0]
                    data[cat_id] = json.load(f)
        return data

    def check_answer(self, answer: str) -> bool:
        """Based on question type, check if the answer is correct or not.

        Args:
            answer (str): answer to the question given by the patient.

        Returns:
            bool: Is a match or not.
        """
        statement = self.main_words[0]
        question = self.main_words[1]

    def get_stimuli(self) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

        Returns:
            tuple[list[str], list[str],str]: stimuli ID and the main words in the stimuli. The prompt to show the user when.
        """
        category_num = random.randint(1, 15)
        stimulus_num = random.randint(1, 35)
        full_question_id = f"{category_num}-{stimulus_num}"
        stimulus = self.stimuli_list[str(category_num)][str(stimulus_num)]
        self.main_words = [stimulus["statement"], stimulus["question"]]
        self.question_id = full_question_id
        response_getting_prompt = "Please answer the question based on the statement."
        return [full_question_id], self.main_words, response_getting_prompt


# todo: finish the class for FAAF
class FAAFQuestions(Questions):
    """Abstract class for questions. Each type of question must use this api."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        super().__init__()
        self.stimuli_list = self._read_json("media/FAAF/stimuli.json")

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
