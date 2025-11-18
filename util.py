"""Utility module for the main script."""

import json
import os

import numpy as np
import yaml
from loguru import logger
from yaml import YAMLError

from audio_processing.noise import Noise, convert_to_specific_db_spl
from hearing_test.test_logic import SpeechInNoise
from hearing_test.test_manager import ASRTestManager, CliTestManager, TestManager
from hearing_test.test_types import TestTypes
from vocalizer.utils import play_sound
from colorama import Fore


def preparation(participant_id, test_number, test_name, test_mode) -> dict[str, str]:
    """Prepare the test and logging system.

    Returns:
        dict: dictionary containing the custom configurations.
    """
    test_name_presentation = test_name + "-" + str(test_number)

    response_capturing_mode = "asr"
    vocalization_mode = "tts"
    signal_processing = "n"

    save_dir = f"records/{participant_id}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"save_dir/{test_name_presentation}", exist_ok=True)

    logger.add(f"{save_dir}/{test_name_presentation}/out.log")
    logger.debug(f"\nParticipant ID: {participant_id}")

    custom_config = {
        "participant_id": participant_id,
        "save_dir": save_dir,
        "test_number": test_number,
        "response_capturing_mode": response_capturing_mode,
        "vocalization_mode": vocalization_mode,
        "test_name": test_name,
        "test_name_presentation": test_name_presentation,
        "test_mode": test_mode,
        "signal_processing": signal_processing,
    }
    return custom_config


def read_configs(custom_config: dict) -> dict:
    """Read config.yaml and update the save location and response capturing mode based on user input.

    Args:
        custom_config (dict): custom configurations provided by the user.

    Returns:
        dict: Configurations.
    """
    configs = read_conf("config.yaml")
    configs["test"][
        "record_save_dir"
    ] = f"{custom_config['save_dir']}/{custom_config['test_name_presentation']}"
    configs["response_capturing"] = custom_config["response_capturing_mode"]
    configs["test_name"] = custom_config["test_name"]
    configs["vocalization_mode"] = custom_config["vocalization_mode"]
    configs["test_name_presentation"] = custom_config["test_name_presentation"]
    configs["test_mode"] = custom_config["test_mode"]
    configs["signal_processing"] = custom_config["signal_processing"]
    configs["participant_id"] = custom_config["participant_id"]
    logger.debug(f"Config file: {configs}")
    return configs


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


def vocalize_stimuli(
    hearing_test: TestTypes,
    stimuli: list[str],
    noise: Noise,
    signal_level: float,
    noise_level: float,
) -> tuple[np.ndarray, int]:
    """Generate sound wave of the stimuli.

    Args:
        hearing_test (TestTypes): hearing test object. is used to get the sound wave.
        stimuli (list[str]): The stimuli to play.
        noise (Noise): object to generate noise.
        signal_level (float): The level of the signal.
        noise_level (float): The level of the noise.

    """
    sample_rate, sound_wave_dict = hearing_test.get_sound(stimuli)

    padding_size = sample_rate // 3

    sound_wave = None
    if "noisy" in sound_wave_dict:
        sound_wave_noisy = np.pad(
            sound_wave_dict["noisy"],
            (padding_size, padding_size),
            "constant",
            constant_values=(0, 0),
        )
        noise_signal = noise.generate_noise(sound_wave_noisy, noise_level)
        sound_wave_noisy = convert_to_specific_db_spl(sound_wave_noisy, signal_level)
        noisy_wave = sound_wave_noisy + noise_signal
        sound_wave = noisy_wave

    if "clean" in sound_wave_dict:
        sound_wave_clean = np.pad(
            sound_wave_dict["clean"],
            (padding_size, padding_size),
            "constant",
            constant_values=(0, 0),
        )
        sound_wave_clean = convert_to_specific_db_spl(sound_wave_clean, signal_level)
        sound_wave += noisy_wave

    return sound_wave, sample_rate


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


def save_results(results: dict) -> None:
    """Save the results of the test.

    Args:
        results (dict): Results of the test.
    """
    participant_id = results["config"]["participant_id"]
    test_name_presentation = results["config"]["test_name_presentation"]
    save_dir = f"records/{participant_id}/{test_name_presentation}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/results.json"
    with open(filename, "w") as outfile:
        json.dump(results, outfile)
