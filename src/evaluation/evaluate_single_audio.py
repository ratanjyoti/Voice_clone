import argparse
import csv
import re
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import cer, wer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "wer_results.csv"

CSV_FIELDS = [
    "language",
    "model_id",
    "sentence_id",
    "audio_file",
    "reference_text",
    "asr_transcript",
    "normalized_reference",
    "normalized_transcript",
    "wer",
    "cer",
    "asr_model",
    "compute_type",
    "success",
    "error_message",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one generated WAV file with faster-whisper."
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--language",
        choices=["en", "ar", "hi"],
        required=True,
    )

    parser.add_argument(
        "--model-id",
        required=True,
    )

    parser.add_argument(
        "--sentence-id",
        required=True,
    )

    parser.add_argument(
        "--asr-model",
        default="tiny",
    )

    return parser.parse_args()


def normalize_common(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(
        r"[\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text,
    )

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ـ": "",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text: str, language: str) -> str:
    if language == "ar":
        return normalize_arabic(text)

    return normalize_common(text)


def append_result(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    needs_header = (
        not RESULTS_FILE.exists()
        or RESULTS_FILE.stat().st_size == 0
    )

    with RESULTS_FILE.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if needs_header:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    args = parse_arguments()
    audio_path = args.audio.resolve()

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    compute_type = "int8"

    print("=" * 72)
    print("Single-file WER/CER evaluation")
    print("=" * 72)
    print(f"Audio: {audio_path}")
    print(f"Language: {args.language}")
    print(f"Model ID: {args.model_id}")
    print(f"Sentence ID: {args.sentence_id}")
    print(f"ASR model: {args.asr_model}")
    print()

    print("Loading faster-whisper...")

    model = WhisperModel(
        args.asr_model,
        device="cpu",
        compute_type=compute_type,
    )

    segments, _ = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=5,
        vad_filter=False,
    )

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    normalized_reference = normalize_text(
        args.reference,
        args.language,
    )

    normalized_transcript = normalize_text(
        transcript,
        args.language,
    )

    wer_score = wer(
        normalized_reference,
        normalized_transcript,
    )

    cer_score = cer(
        normalized_reference,
        normalized_transcript,
    )

    row = {
        "language": args.language,
        "model_id": args.model_id,
        "sentence_id": args.sentence_id,
        "audio_file": str(audio_path),
        "reference_text": args.reference,
        "asr_transcript": transcript,
        "normalized_reference": normalized_reference,
        "normalized_transcript": normalized_transcript,
        "wer": round(wer_score, 4),
        "cer": round(cer_score, 4),
        "asr_model": args.asr_model,
        "compute_type": compute_type,
        "success": True,
        "error_message": "",
    }

    append_result(row)

    print()
    print("Evaluation completed.")
    print(f"Reference: {args.reference}")
    print(f"Transcript: {transcript}")
    print(
        f"Normalized reference: "
        f"{normalized_reference}"
    )
    print(
        f"Normalized transcript: "
        f"{normalized_transcript}"
    )
    print(f"WER: {wer_score:.4f}")
    print(f"CER: {cer_score:.4f}")
    print(f"Saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()