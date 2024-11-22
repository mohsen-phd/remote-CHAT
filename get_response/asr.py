"""Run ASR and convert audio to text."""

import math
from abc import abstractmethod

import tensorflow as tf
import torch
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from speechbrain.pretrained import EncoderDecoderASR
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    SpeechT5ForSpeechToText,
    SpeechT5Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from transformers.utils import logging

from audio_processing.util import convert_sample_rate, read_wav_file
from get_response.base import CaptureResponse

logging.set_verbosity_error()


class ASR(CaptureResponse):
    """Interface for the ASR system."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.file_path = ""

    @abstractmethod
    def get(self) -> str:
        """Transcribe the input file.

        Returns:
            str: File transcription.
        """
        ...


class SpeechBrainASR(ASR):
    """Use Attention and RNNLM trained on LibriSpeech to convert audio to text.

    url: https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech
    """

    def __init__(self, source: str, save_dir: str) -> None:
        """Initialize a SpeechBrain based asr model.

        Args:
            source (str): HuggingFace source of the model.
            save_dir (str): The location the model is save on the local machine.
        """
        super().__init__()
        self.asr_model = EncoderDecoderASR.from_hparams(
            source=source,
            savedir=save_dir,
        )

    def _transcribe(self, src: str) -> str:
        """Get a wav file address and return the transcription of it.

        Args:
            src (str): File address.

        Returns:
            str: transcribe of the file.
        """
        result = self.asr_model.transcribe_file(src)
        return result

    def get(self, src: str) -> str:
        """Transcribe the input file.

        Args:
            src (str): File address.

        Returns:
            str: File transcription.
        """
        return self._transcribe(src)


class SimpleASR(ASR):
    """Class for creating an digit recognizer ASR trained on Digit MNIST."""

    def __init__(self) -> None:
        """Initialize the class by loading HugginFace models."""
        super().__init__()
        self.asr_model = tf.keras.models.load_model(
            "/Users/user/Documents/Projects/PhD/Wordnet-Hearing-Test/models/asr/mnistASR/mnist.h5"
        )
        self.label = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }

    def _get_spectrogram(self, waveform: tf.Tensor) -> tf.Tensor:
        """Get spectrogram of the input waveform.

        Args:
            waveform (tf.Tensor): Input Audio.

        Returns:
            tf.Tensor: Spectrogram of the input audio.
        """
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def _read_file(self, src: str) -> tuple[tf.Tensor, tf.Tensor]:
        """Read the wav file and return the audio and sample rate.

        Args:
            src (str): File address.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Audio and sample rate.
        """
        wav_file: tf.Tensor = tf.io.read_file(str(src))
        wav_file, sample_rate = tf.audio.decode_wav(wav_file, desired_channels=1)
        wav_file = tf.squeeze(wav_file, axis=-1)
        return wav_file, sample_rate

    def _get_features(self, wav_file: tf.Tensor) -> tf.Tensor:
        """Extract features from the input audio.

        Args:
            wav_file (tf.Tensor): Input audio.

        Returns:
            tf.Tensor: Extracted features.
        """
        if wav_file.shape[0] >= 44000:
            wav_file = tf.slice(wav_file, [0], [44000])
        else:
            paddings = tf.constant(
                [
                    [
                        0,
                        44000 - wav_file.shape[0],
                    ]
                ]
            )

            wav_file = tf.pad(wav_file, paddings, "CONSTANT")

        wav_file = self._get_spectrogram(wav_file)
        wav_file = wav_file[tf.newaxis, ...]
        return wav_file

    def _split_digits(
        self, wav_file: AudioSegment, tf_audio: tf.Tensor
    ) -> list[tf.Tensor]:
        """Split the audio file based on silence. Each chunk is a digit.

        Args:
            wav_file (AudioSegment): Wave file read by pydub.
            tf_audio (tf.Tensor): Wave file read by tensorflow.

        Returns:
            list[tf.Tensor]: List of start and end number showing the start and end of each chunk.
        """
        chunks = detect_nonsilent(wav_file, min_silence_len=100, silence_thresh=-50)
        audio_segment = []
        added_window = int(math.fabs(wav_file.frame_rate * 0.2))
        for start, end in chunks:
            start_pos = max(
                math.ceil(start / 1000 * wav_file.frame_rate) - added_window,
                0,
            )
            end_pos = min(
                math.ceil(end / 1000 * wav_file.frame_rate) + added_window,
                tf_audio.shape[0],
            )

            audio_segment.append(tf_audio[start_pos:end_pos])
        return audio_segment

    def _transcribe(self, src: str) -> str:
        """Get a wav file address and return the transcription of it.

        Args:
            src (str): File address.

        Returns:
            str: transcribe of the file.
        """
        wav_file, sample_rate = self._read_file(src)
        chunks = self._split_digits(AudioSegment.from_wav(src), wav_file)
        result_text = ""
        for digit in chunks:
            sample = self._get_features(digit)
            prediction = self.asr_model(sample)
            result = tf.nn.softmax(prediction[0])
            result_text = result_text + " " + self.label[tf.argmax(result).numpy()]
        return result_text.strip()

    def get(self, src: str) -> str:
        """Transcribe the input file.

        Args:
            src (str): File address.

        Returns:
            str: File transcription.
        """
        return self._transcribe(src)


class FBWav2Vec2(ASR):
    """Initialize a Facebook Wav2Vec2 model.

    the model is based on this hugging face repository:
    https://huggingface.co/facebook/wav2vec2-base-960h

    """

    def __init__(self) -> None:
        """Initialize a Facebook Wav2Vec asr model."""
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )

    def get(self, src: str) -> str:
        """Transcribe the input file.

        Args:
            src (str): File address.

        Returns:
            str: File transcription.
        """
        sampling_rate, audio_waveform = read_wav_file(src)

        if sampling_rate != 16000:
            audio_waveform = convert_sample_rate(audio_waveform, sampling_rate, 16000)
            sampling_rate = 16000

        input_values = self.processor(
            audio_waveform,
            return_tensors="pt",
            padding="longest",
            sampling_rate=sampling_rate,
        ).input_values

        # retrieve logits
        logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]


class Whisper(ASR):
    """Initialize a Whisper model.

    The model is based on this hugging face repository:
    https://huggingface.co/openai/whisper-large-v3
    """

    def __init__(self) -> None:
        """Initialize a Whisper asr model."""
        super().__init__()

        self.device = "mps"  # "cpu , mps"

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model_id = "openai/whisper-large-v3-turbo"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def get(self, src: str) -> str:
        """Transcribe the input file.

        Args:
            src (str): File address.

        Returns:
            str: File transcription.
        """
        result = self.pipe(src, generate_kwargs={"language": "english"})
        return result["text"]
