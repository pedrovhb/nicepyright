from __future__ import annotations

import re
from typing import List, NamedTuple

import rich.console

con = rich.console.Console()


def snake_case_to_camel_case(name: str) -> str:
    """Converts a snake_case string to a camelCase string."""
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def camel_case_to_snake_case(name: str) -> str:
    """Converts a camelCase string to a snake_case string."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def split_camel_case_words(name: str) -> List[str]:
    """Splits a camelCase string into words."""
    return re.findall(r".[^A-Z]*", name)


def camel_case_to_capitalized_text(name: str) -> str:
    """Converts a camelCase string to a capitalized text string.

    Args:
        name: The camelCase string to convert.

    Returns:
        The capitalized text string.

    Examples:
        >>> camel_case_to_capitalized_text("camelCase")
        'Camel case'

        >>> camel_case_to_capitalized_text("camelCaseString")
        'Camel case string'
    """
    return " ".join(split_camel_case_words(name)).capitalize()


# Computers have really advanced by now - see
# https://youtu.be/xx5t5ps-bwc?t=48
NINE_NINE_NINE_NINE_NINE = 999999


class Range(NamedTuple):
    line: int
    character: int


__all__ = (
    "camel_case_to_capitalized_text",
    "camel_case_to_snake_case",
    "con",
    "NINE_NINE_NINE_NINE_NINE",
    "Range",
    "snake_case_to_camel_case",
    "split_camel_case_words",
)
