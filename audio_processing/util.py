"""Utility functions for audio processing."""
import numpy as np
from loguru import logger


def rms_amplitude(signal: np.ndarray) -> float:
    """Calculate the RMS amplitude of a signal.

    Args:
        signal (np.ndarray): numpy array containing the signal.

    Returns:
        float: float containing the RMS amplitude of the signal.
    """
    return np.sqrt(np.mean(signal**2))


def calculate_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate the SNR in dB of a signal relative to a noise.

    Args:
        signal (np.ndarray): numpy array containing the signal.
        noise (np.ndarray): numpy array containing the noise.

    Returns:
        float: float containing the SNR in dB of the signal relative to the noise.
    """
    # Calculate the amplitude of the signal
    signal_amplitude = rms_amplitude(signal=signal)

    # Calculate the amplitude of the noise
    noise_amplitude = rms_amplitude(signal=noise)

    # Calculate the SNR in dB
    snr_db = 20 * np.log10(signal_amplitude / noise_amplitude)
    return snr_db


def calculate_db_spl(signal: np.ndarray) -> float:
    """Calculate dB SPL of a signal.

    Args:
        signal (np.ndarray): input signal.

    Returns:
        float: level of the signal in dB SPL.
    """
    # Calculate the RMS of the signal
    rms = rms_amplitude(signal=signal)

    # Calculate the dB SPL
    db_spl = 20 * np.log10(rms / 20e-6)

    return db_spl


def convert_to_specific_db_spl(signal: np.ndarray, target_level: float) -> np.ndarray:
    """Get a signal and change it's level to a specific dB SPL.

    Args:
        signal (np.ndarray): inout signal.
        target_level (float): desired level in dB SPL.

    Returns:
        np.ndarray: signal with the desired level.
    """
    # Calculate the current level of the signal
    current_level = calculate_db_spl(signal)

    # Calculate the difference between the current and desired level
    diff = target_level - current_level

    # Calculate the factor to multiply the signal by
    factor = 10 ** (diff / 20)

    # Multiply the signal by the factor
    signal = signal * factor
    logger.debug(f"Current level: {calculate_db_spl(signal):.2f} dB SPL")
    return signal


def convert_to_specific_rms(signal: np.ndarray, desired_rms: float) -> np.ndarray:
    """Convert a signal to a specific RMS.

    Args:
        signal (np.ndarray): input signal.
        desired_rms (float): desired RMS.

    Returns:
        np.ndarray: scaled signal.
    """
    current_rms = rms_amplitude(signal)

    # Calculate the scaling factor
    scaling_factor = desired_rms / current_rms

    # Normalize the signal to the desired RMS amplitude
    normalized_signal = signal * scaling_factor
    return normalized_signal
