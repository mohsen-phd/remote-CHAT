from flask import Flask, request, jsonify
import os, sys, termios, logging
from colorama import Fore
from test_utils import (
    preparation,
    read_configs,
    get_test_manager,
    SpeechInNoise,
    play_stimuli,
    save_results,
)

app = Flask(__name__)

# Global state (in production: move to session/db)
state = {
    "manager": None,
    "configs": None,
    "track_results": {},
    "snr_db": None,
    "signal_level": 65,
    "noise_level": 60,
    "correct_count": 0,
    "incorrect_count": 0,
    "iteration": 1,
}


@app.route("/start", methods=["POST"])
def start_test():
    custom_config = preparation()
    configs = read_configs(custom_config)
    manager = get_test_manager(configs)

    state.update(
        {
            "manager": manager,
            "configs": configs,
            "snr_db": manager.start_snr,
            "track_results": {},
            "signal_level": 65,
            "noise_level": 60,
            "correct_count": 0,
            "incorrect_count": 0,
            "iteration": 1,
        }
    )

    return jsonify({"message": "Test started", "snr": state["snr_db"]})


@app.route("/next", methods=["POST"])
def next_round():
    manager = state["manager"]
    if manager.hearing_test.stop_condition():
        state["track_results"]["SRT"] = manager.hearing_test.srt
        state["track_results"]["config"] = state["configs"]
        save_results(state["track_results"])
        return jsonify({"end": True, "srt": manager.hearing_test.srt})

    this_round = {"snr": state["snr_db"]}
    stimuli_id, stimuli_text, response_prompt = (
        manager.test_type.stimuli_generator.get_stimuli(
            test_mode=state["configs"]["test_mode"]
        )
    )
    this_round["stimuli"] = stimuli_text

    # Calculate noise/signal
    sig_lvl, noise_lvl = SpeechInNoise.calculate_noise_signal_level(
        state["signal_level"], state["noise_level"], state["snr_db"]
    )
    state["signal_level"], state["noise_level"] = sig_lvl, noise_lvl

    play_stimuli(
        manager.test_type,
        stimuli_id,
        manager.noise,
        signal_level=sig_lvl,
        noise_level=noise_lvl,
    )

    state["track_results"][state["iteration"]] = this_round
    return jsonify(
        {
            "stimuli_text": stimuli_text,
            "snr": state["snr_db"],
            "iteration": state["iteration"],
            "prompt": response_prompt,
        }
    )


@app.route("/response", methods=["POST"])
def handle_response():
    data = request.json
    transcribe = data.get("response")

    manager = state["manager"]
    this_round = state["track_results"][state["iteration"]]
    this_round["response"] = transcribe

    matched = manager.test_type.stimuli_generator.check_answer(transcribe)
    this_round["matched"] = matched

    if matched:
        state["correct_count"] += 1
    else:
        state["incorrect_count"] += 1

    new_snr = manager.hearing_test.get_next_snr(
        state["correct_count"], state["incorrect_count"], state["snr_db"]
    )
    manager.hearing_test.update_variables(matched, state["snr_db"])

    if new_snr != state["snr_db"]:
        state["snr_db"] = new_snr
        state["correct_count"] = state["incorrect_count"] = 0
        state["iteration"] += 1

    return jsonify({"matched": matched, "new_snr": state["snr_db"]})


if __name__ == "__main__":
    app.run(debug=True)
