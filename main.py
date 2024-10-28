"""Main entry point of the program."""

import os
import sys

from colorama import Fore
from loguru import logger

from util import get_test_manager, play_stimuli, read_conf

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def preparation() -> dict[str, str]:
    """Prepare the test and logging system.

    Returns:
        dict: dictionary containing the custom configurations.
    """
    # participant_id = input(Fore.GREEN + "Enter The ID: ")
    # test_number = input(Fore.GREEN + "Enter test number: ")
    # response_capturing_mode = input(Fore.GREEN + "Enter response capturing mode: ")
    # vocalization_mode = input(Fore.GREEN + "Enter vocalization mode: ")
    # test_name = input(Fore.GREEN + "Enter test name: ")
    # todo:replace the following lines with the above lines
    participant_id = 100
    test_number = 1
    response_capturing_mode = "cli"
    test_name = "asl"
    vocalization_mode = "recorded"

    save_dir = f"records/{participant_id}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"save_dir/{test_number}", exist_ok=True)

    logger.add(f"{save_dir}/{test_number}/out.log")
    logger.debug(f"\nParticipant ID: {participant_id}")
    input(Fore.RED + "Press enter to start the test ")
    custom_config = {
        "save_dir": save_dir,
        "test_number": test_number,
        "response_capturing_mode": response_capturing_mode,
        "vocalization_mode": vocalization_mode,
        "test_name": test_name,
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
    ] = f"{custom_config['save_dir']}/{custom_config['test_number']}"
    configs["response_capturing"] = custom_config["response_capturing_mode"]
    configs["test_name"] = custom_config["test_name"]
    configs["vocalization_mode"] = custom_config["vocalization_mode"]
    logger.debug(f"Config file: {configs}")
    return configs


def main():
    """Code entry point."""
    custom_config = preparation()
    configs = read_configs(custom_config)
    manager = get_test_manager(configs)

    snr_db = manager.start_snr
    correct_count = incorrect_count = 0
    iteration = 1
    while not manager.hearing_test.stop_condition():
        print(Fore.RED + "Press Enter to play the next digits")
        input()
        stimuli_id, stimuli_text = manager.test_type.stimuli_generator.get_stimuli()
        print(Fore.YELLOW + "Listen to the numbers")
        logger.debug(f"{iteration} :The stimuli is: {stimuli_text}")
        play_stimuli(manager.test_type, snr_db, stimuli_id, manager.noise)

        transcribe = manager.get_response()

        matched = manager.test_type.stimuli_generator.check_answer(transcribe)
        logger.debug(f"Matched: {matched}")

        if matched:
            correct_count += 1
        else:
            incorrect_count += 1

        new_snr_db = manager.hearing_test.get_next_snr(
            correct_count, incorrect_count, snr_db
        )
        manager.hearing_test.update_variables(matched, snr_db)
        logger.debug(f"New SNR: {new_snr_db}")
        if new_snr_db != snr_db:
            snr_db = new_snr_db
            correct_count = incorrect_count = 0
        iteration += 1

    logger.debug(f"Final SRT: {manager.hearing_test.srt} \n")
    print(Fore.RED + "End of the test")


if __name__ == "__main__":
    main()
