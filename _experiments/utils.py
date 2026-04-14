import sys, os

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))

from _common.constants import Classification
import numpy as np


class TextColor:
    Yellow = "\033[93m"
    Green = "\033[92m"
    Cyan = "\033[96m"
    Red = "\033[91m"
    End = "\033[0m"


def print_green(text: str) -> str:
    return print_colored_text(text, TextColor.Green)


def print_colored_text(text: str, color: str) -> str:
    return f"{color}{text}{TextColor.End}"


def print_cyan(text: str) -> str:
    return print_colored_text(text, TextColor.Cyan)


def print_yellow(text: str) -> str:
    return print_colored_text(text, TextColor.Yellow)


def print_red(text: str) -> str:
    return print_colored_text(text, TextColor.Red)


def print_mass(mass: np.ndarray) -> None:
    nx, ny = mass.shape
    for i in range(nx - 1, -1, -1):
        for j in range(ny):
            colorizer = print_yellow if mass[i, j] == 0 else print_red
            print(colorizer("%.1f" % mass[i, j]), end=" ")
        print()


def print_classification(classification: np.ndarray) -> None:
    cls_to_str = {
        Classification.Colliding: print_cyan("c"),
        Classification.Empty: print_yellow("e"),
        Classification.Interior: print_red("i"),
    }
    nx, ny = classification.shape
    for i in range(nx - 1, -1, -1):
        for j in range(ny):
            print(cls_to_str[classification[i, j]], end="  ")
        print()
