from __future__ import annotations

import re
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import wer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "arabic"
    / "professional_msa"
)

REFERENCES = {
    "arabic_reference_short.wav": (
        "هَلْ تُرِيدُ أَنْ أُشَغِّلَ لَكَ خُطْبَةَ "
        "الْجُمُعَةِ مِنَ الْمَسْجِدِ الْحَرَامِ فِي بَثٍّ "
        "مُبَاشِرٍ عَبْرَ الْإِنْتِرْنِتِ؟"
    ),
    "arabic_reference_standard.wav": (
        "التَّفْسِيرُ الْمَوْضُوعِيُّ لِلْقُرْآنِ يَهْتَمُّ "
        "بِجَمْعِ الْآيَاتِ الَّتِي تَتَحَدَّثُ عَنْ "
        "قَضِيَّةٍ وَاحِدَةٍ وَدِرَاسَتِهَا مَعًا."
    ),
    "arabic_reference_long.wav": (
        "نَعَمْ، يَجُوزُ الْمَسْحُ عَلَى الْخُفَّيْنِ بَدَلًا "
        "مِنْ غَسْلِ الْقَدَمَيْنِ فِي الْوُضُوءِ، بِشَرْطِ "
        "أَنْ يَرْتَدِيَهُمَا الْمُسْلِمُ وَهُوَ عَلَى "
        "طَهَارَةٍ كَامِلَةٍ مِنْ وُضُوءٍ سَابِقٍ."
    ),
}


ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

PUNCTUATION = re.compile(
    r"[^\w\s\u0600-\u06FF]"
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
        "ـ": "",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    text = PUNCTUATION.sub(" ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main() -> None:
    print("Loading Whisper small on CPU...")

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    scores = []

    for filename, expected_text in REFERENCES.items():
        audio_path = REFERENCE_DIR / filename

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        segments, info = model.transcribe(
            str(audio_path),
            language="ar",
            beam_size=5,
            vad_filter=True,
        )

        predicted_text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        normalized_expected = normalize_arabic(
            expected_text
        )

        normalized_predicted = normalize_arabic(
            predicted_text
        )

        score = wer(
            normalized_expected,
            normalized_predicted,
        )

        scores.append(score)

        print("=" * 80)
        print(f"File: {filename}")
        print(
            f"Arabic probability: "
            f"{info.language_probability:.4f}"
        )
        print(f"Expected: {expected_text}")
        print(f"Predicted: {predicted_text}")
        print(
            f"Normalized expected: "
            f"{normalized_expected}"
        )
        print(
            f"Normalized predicted: "
            f"{normalized_predicted}"
        )
        print(f"WER: {score:.4f}")
        print(f"WER percentage: {score * 100:.2f}%")

    average_wer = sum(scores) / len(scores)

    print("=" * 80)
    print(f"Average normalized WER: {average_wer:.4f}")
    print(
        f"Average normalized WER percentage: "
        f"{average_wer * 100:.2f}%"
    )


if __name__ == "__main__":
    main()