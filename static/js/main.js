document.addEventListener("DOMContentLoaded", () => {
    const startBtn = document.getElementById("startBtn");
    const NextBtn = document.getElementById("NextBtn");
    const output = document.getElementById("output");
    const responseSection = document.getElementById("responseSection");
    const recordBtn = document.getElementById("recordBtn");

    startBtn.addEventListener("click", async () => {
        let res = await fetch("/start", { method: "POST" });
        let data = await res.json();
        output.innerHTML = `<p>Test started. SNR: ${data.snr}</p>`;
        nextRound();
    });
    NextBtn.addEventListener("click", async () => {
        nextRound();
    });
    async function nextRound() {
        let res = await fetch("/next", { method: "POST" });
        let data = await res.json();
        if (data.end) {
            output.innerHTML += '<p>Test finished! Final SRT: ${data.srt}</p>';
            responseSection.style.display = "none";
            return;
        }
        output.innerHTML += `<p><strong>Stimuli:</strong> ${data.stimuli_text}</p>`;
        // 👇 Play audio sent as base64
        if (data.audio_base64) {
            const audioSrc = "data:audio/wav;base64," + data.audio_base64;
            const audio = new Audio(audioSrc);
            audio.play();
        }
        responseSection.style.display = "block";
    }
    // 🎤 Start recording for 3 seconds
    recordBtn.addEventListener("click", async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const formData = new FormData();
            formData.append("audio", audioBlob, "response.wav");
            // Send to backend
            let res = await fetch("/response", {
                method: "POST",
                body: formData
            });
            let data = await res.json();
            output.innerHTML += `<p>Matched: ${data.matched}, New SNR: ${data.new_snr}</p>`;
            nextRound();
        };
        mediaRecorder.start();
        // ⏱ stop automatically after 3 seconds
        setTimeout(() => {
            mediaRecorder.stop();
        }, 3000);
    });
});