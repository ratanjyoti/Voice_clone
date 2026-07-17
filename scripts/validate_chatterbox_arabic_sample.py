from __future__ import annotations

import re
import sys
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import wer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "chatterbox"
    / "arabic"
    / "ar_clone_01_greeting.wav"
)

EXPECTED_TEXT = (
    "مَرْحَبًا بِكَ. أَنَا مُسَاعِدُكَ الصَّوْتِيُّ، "
    "وَيَسْعَدُنِي أَنْ أُسَاعِدَكَ الْيَوْمَ."
)

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

ARABIC_PUNCTUATION = re.compile(
    r"[،؛؟.!«»\"'ـ]"
)


def normalize_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text)

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    text = ARABIC_PUNCTUATION.sub(" ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main() -> None:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(AUDIO_PATH)

    print("Loading Whisper small on CPU...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        str(AUDIO_PATH),
        language="ar",
        beam_size=5,
        vad_filter=True,
    )

    predicted = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    expected_normalized = normalize_arabic(
        EXPECTED_TEXT
    )
    predicted_normalized = normalize_arabic(
        predicted
    )

    score = wer(
        expected_normalized,
        predicted_normalized,
    )

    print(f"Arabic probability: {info.language_probability:.4f}")
    print(f"Expected: {EXPECTED_TEXT}")
    print(f"Predicted: {predicted}")
    print(f"Normalized expected: {expected_normalized}")
    print(f"Normalized predicted: {predicted_normalized}")
    print(f"WER: {score:.4f}")
    print(f"WER percentage: {score * 100:.2f}%")


if __name__ == "__main__":
    main()

