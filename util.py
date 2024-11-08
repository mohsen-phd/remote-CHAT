"""Utility module for the main script."""

import json
import numpy as np
import yaml
from loguru import logger
from yaml import YAMLError

from audio_processing.noise import Noise
from hearing_test.test_manager import ASRTestManager, CliTestManager, TestManager
from hearing_test.test_types import TestTypes
from vocalizer.utils import play_sound


def read_conf(src: str = "config.yaml") -> dict:
    """Read the configuration file.

    Args:
        src (str): Path to the configuration file. Defaults to "config.yaml".

    Returns:
        dict: Return the configuration file as a dictionary.

    Raises:
        YAMLError: If an error occurs while reading or parsing the configuration file.
    """
    with open(src, "r") as f:
        try:
            return yaml.safe_load(f)
        except YAMLError as exc:
            logger.error(f"Error while reading the configuration file: {exc}")
            raise exc


def play_stimuli(
    hearing_test: TestTypes, snr_db: int, stimuli: list[str], noise: Noise
):
    """Play the stimuli to the patient.

    Args:
        hearing_test (TestTypes): hearing test object. is used to get the sound wave.
        snr_db (int): signal to noise ratio in db.
        stimuli (str): The stimuli to play.
        noise (Noise): object to generate noise.
    """
    sound_wave = hearing_test.get_sound(stimuli)
    sound_wave = np.pad(sound_wave, (5000, 5000), "constant", constant_values=(0, 0))
    noise_signal = noise.generate_noise(sound_wave, snr_db)
    noisy_wave = sound_wave + noise_signal
    play_sound(wave=noisy_wave, fs=22050)


def get_test_manager(configs: dict) -> TestManager:
    """Return the proper test manager based on config file.

    Args:
        configs (dict): loaded config file.

    Raises:
        NotImplementedError: If the test type is not implemented.

    Returns:
        TestManager: Appropriate test manager.
    """
    if configs["response_capturing"] == "cli":
        return CliTestManager(configs)
    elif configs["response_capturing"] == "asr":
        return ASRTestManager(configs)
    else:
        raise NotImplementedError


def save_results(results: dict):
    """Save the results of the test.

    Args:
        results (dict): Results of the test.
    """
    with open(
        f"records/snr_results/{results['config']['test_number']}/participant_id.json",
        "w",
    ) as outfile:
        json.dump(results, outfile)
