# flake8: noqa
import pytest
from contextlib import nullcontext as does_not_raise
from hearing_test.test_logic import SpeechInNoise, STATUS


@pytest.fixture(scope="module")
def digit_in_noise_test() -> SpeechInNoise:
    return SpeechInNoise(
        correct_threshold=2,
        incorrect_threshold=1,
        step_size=[5, 3, 1],
        reversal_limit=10,
    )


class Test_DigitInNoise:
    def test_srt(self, digit_in_noise_test: SpeechInNoise):
        digit_in_noise_test._important_snr = [1, 2]
        assert digit_in_noise_test.srt == 1.5

    @pytest.mark.parametrize(
        "input, expected",
        [
            (11, True),
            (9, False),
            (10, False),
            (0, False),
            (-1, False),
        ],
    )
    def test_stop_condition_valid_input(
        self, digit_in_noise_test: SpeechInNoise, input, expected
    ):
        digit_in_noise_test._reversal_count = input
        assert digit_in_noise_test.stop_condition() == expected

    @pytest.mark.parametrize(
        "input, expected",
        [
            ("11", pytest.raises(TypeError)),
            (None, pytest.raises(TypeError)),
            (11, does_not_raise()),
        ],
    )
    def test_stop_condition_invalid_input(
        self, digit_in_noise_test: SpeechInNoise, input, expected
    ):
        with expected:
            digit_in_noise_test._reversal_count = input
            assert digit_in_noise_test.stop_condition() is not None
