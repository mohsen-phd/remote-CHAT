"""Get respond form command line."""

import sys
import termios

from colorama import Fore

from get_response.base import CaptureResponse


class CLI(CaptureResponse):
    """Interface for the ASR system."""

    def get(self) -> str:
        """Get response from command line.

        Returns:
            str: File transcription.
        """
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        return input(Fore.GREEN + "Enter your response: ")
