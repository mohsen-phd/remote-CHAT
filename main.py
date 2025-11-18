import base64
import io
import os
import time
import soundfile as sf
from flask import Flask, jsonify, render_template, request, session
from pydub import AudioSegment
from get_response.asr import Whisper
from hearing_test.test_logic import SpeechInNoise
from util import (
    get_test_manager,
    vocalize_stimuli,
    preparation,
    read_configs,
    save_results,
)

app = Flask(__name__)
app.secret_key = "supersecret"  # required for Flask sessions

# Global state dictionary (keys = session user_ids)
state = {}
whisper = Whisper()


def get_run_key(participant_id, run_number):
    return participant_id + "_" + str(run_number)


@app.route("/start_practice", methods=["POST"])
def start_practice():
    cleanup_state(timeout=1800)
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User session not initialized"}), 400

    custom_config = preparation(
        participant_id=session.get("participant_id"),
        test_number=session.get("test_number"),
        test_name="chat",
        test_mode="practice",
    )
    configs = read_configs(custom_config)
    manager = get_test_manager(configs)

    # Create a per-user state dict
    state[user_id] = {
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

    return jsonify({"message": "Test started", "snr": state[user_id]["snr_db"]})


@app.route("/start", methods=["POST"])
def start_test():
    cleanup_state(timeout=1800)
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "User session not initialized"}), 400

    custom_config = preparation(
        participant_id=session.get("participant_id"),
        test_number=session.get("test_number"),
        test_name="chat",
        test_mode="test",
    )
    configs = read_configs(custom_config)
    manager = get_test_manager(configs)

    # Create a per-user state dict
    state[user_id] = {
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

    return jsonify({"message": "Test started", "snr": state[user_id]["snr_db"]})


def cleanup_state(timeout=1800):  # 30 minutes
    now = time.time()
    to_remove = [
        uid for uid, s in state.items() if now - s.get("last_update", 0) > timeout
    ]
    for uid in to_remove:
        state.pop(uid, None)


@app.route("/next", methods=["POST"])
def next_round():
    user_id = session.get("user_id")
    if not user_id or user_id not in state:
        return jsonify({"error": "User state not found"}), 404

    user_state = state[user_id]
    manager = user_state["manager"]
    user_state["last_update"] = time.time()

    if manager.hearing_test.stop_condition():
        user_state["track_results"]["SRT"] = manager.hearing_test.srt
        user_state["track_results"]["config"] = user_state["configs"]
        save_results(user_state["track_results"])
        return jsonify({"end": True, "srt": manager.hearing_test.srt})

    this_round = {"snr": user_state["snr_db"]}
    stimuli_id, stimuli_text, response_prompt = (
        manager.test_type.stimuli_generator.get_stimuli(
            test_mode=user_state["configs"]["test_mode"]
        )
    )
    this_round["stimuli"] = stimuli_text

    sig_lvl, noise_lvl = SpeechInNoise.calculate_noise_signal_level(
        70, user_state["snr_db"]
    )
    user_state["signal_level"], user_state["noise_level"] = sig_lvl, noise_lvl

    audio_array, sample_rate = vocalize_stimuli(
        manager.test_type,
        stimuli_id,
        manager.noise,
        signal_level=sig_lvl,
        noise_level=noise_lvl,
    )

    # First save the NumPy array into a temporary WAV in memory
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)

    # Convert WAV -> MP3 using pydub
    audio_segment = AudioSegment.from_file(wav_buf, format="wav")
    mp3_buf = io.BytesIO()
    audio_segment.export(mp3_buf, format="mp3", bitrate="192k")
    mp3_buf.seek(0)

    audio_base64 = base64.b64encode(mp3_buf.read()).decode("utf-8")

    user_state["track_results"][user_state["iteration"]] = this_round

    return jsonify(
        {
            "stimuli_text": stimuli_text,
            "snr": user_state["snr_db"],
            "iteration": user_state["iteration"],
            "prompt": response_prompt,
            "audio_base64": audio_base64,
        }
    )


@app.route("/response", methods=["POST"])
def handle_response():
    user_id = session.get("user_id")
    if not user_id or user_id not in state:
        return jsonify({"error": "User state not found"}), 404

    user_state = state[user_id]

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    upload_folder = user_state["configs"]["test"]["record_save_dir"]
    file_name = f"{user_state['iteration']}.wav"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file_name)
    file.save(file_path)

    transcribe = user_state["manager"].get_response(file_path)
    matched = user_state["manager"].test_type.stimuli_generator.check_answer(transcribe)

    this_round = user_state["track_results"][user_state["iteration"]]
    this_round["response"] = transcribe
    this_round["matched"] = matched

    if matched:
        user_state["correct_count"] += 1
    else:
        user_state["incorrect_count"] += 1

    new_snr = user_state["manager"].hearing_test.get_next_snr(
        user_state["correct_count"], user_state["incorrect_count"], user_state["snr_db"]
    )
    user_state["manager"].hearing_test.update_variables(matched, user_state["snr_db"])

    if new_snr != user_state["snr_db"]:
        user_state["snr_db"] = new_snr
        user_state["correct_count"] = user_state["incorrect_count"] = 0
        user_state["iteration"] += 1

    return jsonify(
        {
            "matched": matched,
            "new_snr": user_state["snr_db"],
            "ASR_transcription": transcribe,
        }
    )


@app.route("/<id>/<run>")
def home(id, run):
    user_key = get_run_key(id, run)
    # store user_key in session for later requests
    if "user_id" in session:
        state.pop(session["user_id"], None)  # clear old state if exists

    session["user_id"] = user_key
    session["participant_id"] = id
    session["test_number"] = run

    return render_template("home.html")


@app.route("/calibration")
def calibration():
    return render_template("calibration.html")


@app.route("/microphone")
def microphone():
    return render_template("microphone.html")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/practice")
def practice():
    return render_template("practice.html")


if __name__ == "__main__":
    app.run(debug=True)
