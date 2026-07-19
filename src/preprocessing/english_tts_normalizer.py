from __future__ import annotations

import re
from dataclasses import dataclass


ONES = {
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
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}

ACRONYMS = {
    "AI": "A I",
}


@dataclass(frozen=True)
class NormalizedTTSInput:
    expected_text: str
    tts_input_text: str


def number_under_100(value: int) -> str:
    if value < 20:
        return ONES[value]
    tens = value // 10 * 10
    remainder = value % 10
    if remainder == 0:
        return TENS[tens]
    return f"{TENS[tens]} {ONES[remainder]}"


def integer_to_words(value: int) -> str:
    if value < 0:
        return "minus " + integer_to_words(abs(value))
    if value < 100:
        return number_under_100(value)
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        if remainder == 0:
            return f"{ONES[hundreds]} hundred"
        return f"{ONES[hundreds]} hundred {number_under_100(remainder)}"
    if 2000 <= value <= 2099:
        return " ".join(ONES[int(digit)] for digit in str(value))
    if value < 10000:
        return " ".join(ONES[int(digit)] for digit in str(value))
    return str(value)


def expand_times(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if minute == 0:
            return f"{integer_to_words(hour)} o clock"
        if minute < 10:
            return f"{integer_to_words(hour)} oh {integer_to_words(minute)}"
        return f"{integer_to_words(hour)} {integer_to_words(minute)}"

    return re.sub(r"\b([1-9]|1[0-2]):([0-5][0-9])\b", repl, text)


def expand_currency(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        amount = int(match.group(1))
        unit = "dollar" if amount == 1 else "dollars"
        return f"{integer_to_words(amount)} {unit}"

    return re.sub(r"\$(\d{1,4})\b", repl, text)


def expand_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = int(match.group(0))
        return integer_to_words(value)

    return re.sub(r"\b\d{1,4}\b", repl, text)


def expand_acronyms(text: str) -> str:
    for acronym, spoken in ACRONYMS.items():
        text = re.sub(rf"\b{re.escape(acronym)}\b", spoken, text)
    return text


def normalize_punctuation(text: str) -> str:
    text = text.replace("&", " and ")
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[;:]", ",", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_tts(expected_text: str) -> NormalizedTTSInput:
    tts_text = expected_text
    tts_text = expand_currency(tts_text)
    tts_text = expand_times(tts_text)
    tts_text = expand_acronyms(tts_text)
    tts_text = expand_numbers(tts_text)
    tts_text = normalize_punctuation(tts_text)
    return NormalizedTTSInput(
        expected_text=expected_text,
        tts_input_text=tts_text,
    )

