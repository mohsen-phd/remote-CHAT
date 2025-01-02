"""Class to govern the logic of the hearing test."""

from abc import ABC, abstractmethod
from enum import Enum

from loguru import logger


class STATUS(Enum):
    """Enum to represent the status of the test."""

    INIT = 0
    INCREASE = 1
    DECREASE = 2


class HearingTest(ABC):
    """Interface for the hearing test."""

    def __init__(self) -> None:
        """Initialize the HearingTest class."""
        self._important_snr: list[int] = []
        self._srt: float
        super().__init__()

    @property
    def srt(self) -> float:
        """Get the SRT value.

        Raises:
            ValueError: if there is no snr in the important_snr list.

        Returns:
            float: the SRT value, which is the average of the important_snr(SRTs when
            the step size is the smallest defined) list.
        """
        if len(self._important_snr) == 0:
            raise ValueError("Not enough data to calculate SRT")
        return sum(self._important_snr) / len(self._important_snr)

    @abstractmethod
    def get_next_snr(
        self, correct_count: int, incorrect_count: int, snr_db: int
    ) -> int:
        """Get the next SNR value.

        The SNR value is calculated based on the number of correct and incorrect.

        Args:
            correct_count (int): the number of correct answers
            incorrect_count (int): the number of incorrect answers
            snr_db (int): the current snr value

        Returns:
            int: new snr to use
        """
        ...

    @abstractmethod
    def stop_condition(self) -> bool:
        """Check if the test should stop.

        Returns:
            bool: True if the test should stop, False otherwise.
        """
        ...


class SpeechInNoise(HearingTest):
    """Class to govern the logic of the speech-in-noise test."""

    def __init__(
        self,
        correct_threshold: int,
        incorrect_threshold: int,
        step_size: list[int],
        reversal_limit: int,
        minimum_threshold: int,
    ):
        """Initialize the SpeechInNoise class.

        Args:
            correct_threshold (int): how many correct answers before decreasing the SNR
            incorrect_threshold (int): how many incorrect answers
                                            before increasing the SNR.
            step_size (list[int]): In what steps to increase/decrease the SNR
            reversal_limit (int): How many reversals before stopping the test.
            minimum_threshold(int): The minimum SNR to use. is used to  stop the test if multiple correct answers are given at the minimum SNR.
        """
        self._correct_threshold = correct_threshold
        self._incorrect_threshold = incorrect_threshold
        self._step_size = step_size
        self._reversal_count = 0
        self._previous_action = STATUS.INIT
        self._reversal_limit = reversal_limit
        self.correct_count_at_max_snr = 0
        self._current_status: STATUS = STATUS.INIT
        self._minimum_threshold = minimum_threshold

        super().__init__()

    def _is_reversing(self, new_status: STATUS) -> bool:
        """Check if the test is reversing.

        Args:
            new_status (STATUS): the new STATUS of the test in this iteration

        Returns:
            bool: True if the test is reversing, False otherwise
        """
        if self._previous_action == STATUS.INIT:
            return False

        if self._previous_action == new_status:
            return False
        else:
            return True

    def _get_step_size(self) -> int:
        """Get the step size to for changing the SNR.

        Returns:
            int: The amount to change the SNR.
        """
        if self._reversal_count >= 4:
            return self._step_size[2]
        elif self._reversal_count >= 2:
            return self._step_size[1]
        else:
            return self._step_size[0]

    def get_next_snr(
        self, correct_count: int, incorrect_count: int, snr_db: int
    ) -> int:
        """Get the next SNR value.

        The SNR value is calculated based on the number of correct and incorrect.And
        the SNR is stored in the important_snr list if the last step size is being used.

        Args:
            correct_count (int): the number of correct answers
            incorrect_count (int): the number of incorrect answers
            snr_db (int): the current snr value

        Returns:
            int: new snr to use
        """
        new_snr = snr_db
        if correct_count >= self._correct_threshold:
            self._current_status = STATUS.DECREASE
        elif incorrect_count >= self._incorrect_threshold:
            self._current_status = STATUS.INCREASE
        else:
            return new_snr

        snr_change = self._get_snr_change()
        new_snr = snr_db + snr_change

        self._update_important_snr(new_snr, abs(snr_change))
        return new_snr

    def update_variables(self, correct_response: bool, snr: float) -> None:
        """Update internal variables of the test.

        Args:
            correct_response (bool): The response is correct or not.
            snr (float): The current SNR value.
        """
        if correct_response and snr < self._minimum_threshold:
            self.correct_count_at_max_snr += 1

        if self._is_reversing(self._current_status):
            self._reversal_count += 1
            logger.debug(f"Reversal: {self._reversal_count}")
        self._previous_action = self._current_status

    def _update_important_snr(self, new_snr: int, snr_change: int) -> None:
        """Add the new SNR to the list of important SNRs.

        If the snr_change is equal to the last step size defined step size,
          store the news_str in the important_snr list.

        Args:
            new_snr (int): The SNR value of current iteration of the test.
            snr_change (int): The step size for changing the SNR in this iteration.
        """
        if snr_change == self._step_size[-1]:
            self._important_snr.append(new_snr)

    def _get_snr_change(self) -> int:
        """Get the step size of SNR change, based on current status.

        Returns:
            int: how much to change the SNR
        """
        snr_change = self._get_step_size()
        if self._current_status == STATUS.DECREASE:
            snr_change *= -1
        return snr_change

    def stop_condition(self) -> bool:
        """Check if the test should stop.

        Returns:
            bool: True if the test should stop, False otherwise.
        """
        if self._reversal_count >= self._reversal_limit:
            return True
        elif self.correct_count_at_max_snr >= 6:
            return True
        else:
            return False

    @staticmethod
    def calculate_noise_signal_level(
        signal_level: float, noise_level: float, snr_db: float
    ) -> tuple[float, float]:
        """Get the current noise level, stimuli level and desired SNR and calculate the noise and stimuli level for next iteration. Cap the noise add 75 dB.

        Args:
            signal_level (float): The level of the signal.
            noise_level (float): The level of the noise.
            snr_db (float): The desired SNR in dB.

        Returns:
            tuple[float, float]: New levels for noise and stimuli.
        """

        def increase_snr(
            signal_level: float, noise_level: float, step_size: float
        ) -> tuple[float, float]:
            if signal_level <= 60:
                signal_level += step_size
            elif noise_level - step_size > 50:
                noise_level -= step_size
            elif signal_level + step_size < 75:
                signal_level += step_size
            else:
                noise_level -= step_size
            return signal_level, noise_level

        def reduce_snr(
            signal_level: float, noise_level: float, step_size: float
        ) -> tuple[float, float]:
            if signal_level >= 70:
                signal_level -= step_size
            elif 55 < noise_level + step_size < 75:
                noise_level += step_size
            elif 75 > signal_level - step_size >= 50:
                signal_level -= step_size
            else:
                noise_level += step_size
            return signal_level, noise_level

        current_snr = signal_level - noise_level
        step_size = abs(current_snr - snr_db)

        if current_snr == snr_db:
            return signal_level, noise_level

        if snr_db > current_snr:
            signal_level, noise_level = increase_snr(
                signal_level, noise_level, step_size
            )
        elif current_snr > snr_db:
            signal_level, noise_level = reduce_snr(signal_level, noise_level, step_size)

        return signal_level, noise_level
