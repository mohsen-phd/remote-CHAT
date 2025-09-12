import base64
import io
import os
import soundfile as sf
from flask import Flask, jsonify, render_template, request

from hearing_test.test_logic import SpeechInNoise
from util import get_test_manager, play_stimuli, preparation, read_configs, save_results

app = Flask(__name__)

# Global state (for demo; in production use DB or session)
state = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_test():
    custom_config = preparation()
    configs = read_configs(custom_config)
    manager = get_test_manager(configs)

    state.update(
        {
            "manager": manager,
            "configs": configs,
            "track_results": {},
            "snr_db": manager.start_snr,
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

    sig_lvl, noise_lvl = SpeechInNoise.calculate_noise_signal_level(
        state["signal_level"], state["noise_level"], state["snr_db"]
    )
    state["signal_level"], state["noise_level"] = sig_lvl, noise_lvl

    audio_array, sample_rate = play_stimuli(
        manager.test_type,
        stimuli_id,
        manager.noise,
        signal_level=sig_lvl,
        noise_level=noise_lvl,
    )

    # Encode to WAV in-memory
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV")
    buf.seek(0)

    # Convert to base64
    audio_base64 = base64.b64encode(buf.read()).decode("utf-8")

    state["track_results"][state["iteration"]] = this_round

    return jsonify(
        {
            "stimuli_text": stimuli_text,
            "snr": state["snr_db"],
            "iteration": state["iteration"],
            "prompt": response_prompt,
            "audio_base64": audio_base64,
        }
    )


@app.route("/response", methods=["POST"])
def handle_response():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]

    # Save with timestamp or iteration
    upload_folder = state["configs"]["test"]["record_save_dir"]
    file_name = f"{state['iteration']}.wav"

    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file_name)
    file.save(file_path)

    # Here you could run ASR/transcription on the saved wav file:
    # transcribe = your_asr_function(filepath)

    transcribe = state["manager"].get_response(file_path)
    matched = state["manager"].test_type.stimuli_generator.check_answer(transcribe)

    # Update test state
    this_round = state["track_results"][state["iteration"]]
    this_round["response"] = transcribe
    this_round["matched"] = matched

    if matched:
        state["correct_count"] += 1
    else:
        state["incorrect_count"] += 1

    new_snr = state["manager"].hearing_test.get_next_snr(
        state["correct_count"], state["incorrect_count"], state["snr_db"]
    )
    state["manager"].hearing_test.update_variables(matched, state["snr_db"])

    if new_snr != state["snr_db"]:
        state["snr_db"] = new_snr
        state["correct_count"] = state["incorrect_count"] = 0
        state["iteration"] += 1

    return jsonify({"matched": matched, "new_snr": state["snr_db"]})


if __name__ == "__main__":
    app.run(debug=True)
