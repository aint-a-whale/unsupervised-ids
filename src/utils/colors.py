from dataclasses import dataclass
from typing import Tuple


@dataclass
class ColorCodes:
    green: Tuple[int, int, int] = (100, 210, 100)  # #64d264  pastel_green
    dark_yellow: Tuple[int, int, int] = (200, 180, 50)  # #c8b432 old_gold
    purple: Tuple[int, int, int] = (89, 69, 181)  # #5945b5  royal_purple
    blue: Tuple[int, int, int] = (100, 180, 255)  # #64b4ff  maya_blue
    yellow_orange: Tuple[int, int, int] = (220, 120, 20)  # #dc7814
    black: Tuple[int, int, int] = (0, 0, 0)  # #000000
    white: Tuple[int, int, int] = (255, 255, 255)  # #ffffff
    dark_violet: Tuple[int, int, int] = (85, 70, 150)  # #554696  daisy_bush
    maple: Tuple[int, int, int] = (217, 175, 107)  # #D9AF6B  sweet maple

    wheel = wheel_positive = green
    wheel_aux = wheel_negative = dark_yellow
    vehicle = dark_violet
    wheel_border = yellow_orange

    @classmethod
    def gray(cls, level: int = 80) -> Tuple[int, int, int]:
        return (level, level, level)


class ANSIColors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[38m"
    BOLD_RED = "\033[31;1m"

    @staticmethod
    def format(text: str | int | float, color_code: str) -> str:
        return f"{color_code}{text}{ANSIColors.RESET}"
