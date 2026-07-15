import csv
import re
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import cer, wer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_RESULTS_FILE = PROJECT_ROOT / "results" / "raw_runs.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "wer_results.csv"


LANGUAGE_CODES = {
    "english": "en",
    "arabic": "ar",
    "hindi": "hi",
}


OUTPUT_FIELDS = [
    "language",
    "model_id",
    "sentence_id",
    "run_number",
    "input_text",
    "output_file",
    "asr_transcript",
    "normalized_reference",
    "normalized_transcript",
    "wer",
    "cer",
    "success",
    "error_message",
]


def normalize_english(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic diacritics.
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text,
    )

    # Normalize common alef variants.
    text = re.sub(r"[Ø¥Ø£Ø¢Ù±]", "Ø§", text)

    # Normalize ya and ta marbuta conservatively.
    text = text.replace("Ù‰", "ÙŠ")
    text = text.replace("Ù€", "")

    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    return " ".join(text.split())


def normalize_hindi(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    return " ".join(text.split())


def normalize_text(text: str, language: str) -> str:
    if language == "english":
        return normalize_english(text)

    if language == "arabic":
        return normalize_arabic(text)

    if language == "hindi":
        return normalize_hindi(text)

    return " ".join(text.lower().split())


def load_rows() -> list[dict]:
    with RAW_RESULTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main() -> None:
    if not RAW_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing benchmark file: {RAW_RESULTS_FILE}"
        )

    rows = load_rows()

    if not rows:
        raise ValueError("raw_runs.csv contains no rows.")

    print("Loading faster-whisper tiny model on CPU...")
    print("The first run may download the model.")

    asr_model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
    )

    results = []

    for index, row in enumerate(rows, start=1):
        language = row["language"]
        audio_path = Path(row["output_file"])

        print(
            f"[{index}/{len(rows)}] "
            f"{language} | {row['sentence_id']} "
            f"| run {row['run_number']}"
        )

        success = True
        error_message = ""
        transcript = ""
        normalized_reference = ""
        normalized_transcript = ""
        wer_score = ""
        cer_score = ""

        try:
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            segments, info = asr_model.transcribe(
                str(audio_path),
                language=LANGUAGE_CODES[language],
                beam_size=5,
                vad_filter=False,
            )

            transcript = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )

            normalized_reference = normalize_text(
                row["input_text"],
                language,
            )

            normalized_transcript = normalize_text(
                transcript,
                language,
            )

            wer_score = round(
                wer(
                    normalized_reference,
                    normalized_transcript,
                ),
                4,
            )

            cer_score = round(
                cer(
                    normalized_reference,
                    normalized_transcript,
                ),
                4,
            )

            print("  Transcript:", transcript)
            print("  WER:", wer_score)
            print("  CER:", cer_score)

        except Exception as error:
            success = False
            error_message = (
                f"{type(error).__name__}: {error}"
            )
            print("  FAILED:", error_message)

        results.append(
            {
                "language": language,
                "model_id": row["model_id"],
                "sentence_id": row["sentence_id"],
                "run_number": row["run_number"],
                "input_text": row["input_text"],
                "output_file": row["output_file"],
                "asr_transcript": transcript,
                "normalized_reference": normalized_reference,
                "normalized_transcript": normalized_transcript,
                "wer": wer_score,
                "cer": cer_score,
                "success": success,
                "error_message": error_message,
            }
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("WER evaluation completed.")
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

