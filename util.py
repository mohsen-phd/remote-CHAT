"""Utility module for the main script."""

import json
import os
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
    sample_rate, sound_wave_dict = hearing_test.get_sound(stimuli)

    padding_size = sample_rate // 3
    if "noisy" in sound_wave_dict:
        sound_wave_noisy = np.pad(
            sound_wave_dict["noisy"],
            (padding_size, padding_size),
            "constant",
            constant_values=(0, 0),
        )
        noise_signal = noise.generate_noise(sound_wave_noisy, snr_db)
        noisy_wave = sound_wave_noisy + noise_signal
        play_sound(wave=noisy_wave, fs=sample_rate)

    if "clean" in sound_wave_dict:
        sound_wave_clean = np.pad(
            sound_wave_dict["clean"],
            (padding_size, padding_size),
            "constant",
            constant_values=(0, 0),
        )
        play_sound(wave=sound_wave_clean, fs=sample_rate)


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
    filename = f"records/snr_results/{results['config']['test_name_presentation']}/{results['config']['participant_id']}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(
        filename,
        "w",
    ) as outfile:
        json.dump(results, outfile)
