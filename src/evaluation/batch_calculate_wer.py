import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel
from jiwer import cer, wer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANGUAGE_CONFIG = {
    "english": {
        "code": "en",
        "test_file": (
            PROJECT_ROOT
            / "data"
            / "test_sentences"
            / "english.json"
        ),
        "audio_dir": (
            PROJECT_ROOT
            / "outputs"
            / "mms"
            / "english"
        ),
    },
    "arabic": {
        "code": "ar",
        "test_file": (
            PROJECT_ROOT
            / "data"
            / "test_sentences"
            / "arabic.json"
        ),
        "audio_dir": (
            PROJECT_ROOT
            / "outputs"
            / "mms"
            / "arabic"
        ),
    },
    "hindi": {
        "code": "hi",
        "test_file": (
            PROJECT_ROOT
            / "data"
            / "test_sentences"
            / "hindi.json"
        ),
        "audio_dir": (
            PROJECT_ROOT
            / "outputs"
            / "mms"
            / "hindi"
        ),
    },
}

MODEL_IDS = {
    "english": "facebook/mms-tts-eng",
    "arabic": "facebook/mms-tts-ara",
    "hindi": "facebook/mms-tts-hin",
}

RESULTS_FILE = PROJECT_ROOT / "results" / "wer_results.csv"

CSV_FIELDS = [
    "language",
    "language_code",
    "model_id",
    "sentence_id",
    "category",
    "audio_file",
    "reference_text",
    "asr_transcript",
    "normalized_reference",
    "normalized_transcript",
    "wer",
    "cer",
    "asr_model",
    "success",
    "error_message",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch WER/CER evaluation for generated MMS audio."
    )

    parser.add_argument(
        "--language",
        choices=LANGUAGE_CONFIG.keys(),
        required=True,
    )

    parser.add_argument(
        "--asr-model",
        default="tiny",
        help="faster-whisper model: tiny, base, small, medium, etc.",
    )

    parser.add_argument(
        "--run-number",
        type=int,
        default=1,
        help="Which generated benchmark run to evaluate.",
    )

    parser.add_argument(
        "--overwrite-language",
        action="store_true",
        help="Remove existing WER rows for this language.",
    )

    return parser.parse_args()


def normalize_common(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
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

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text: str, language_code: str) -> str:
    if language_code == "ar":
        return normalize_arabic(text)

    return normalize_common(text)


def load_sentences(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test sentence file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError(
            f"No test sentences found in: {path}"
        )

    return records


def remove_existing_language_rows(language: str) -> None:
    if not RESULTS_FILE.exists():
        return

    with RESULTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = [
            row
            for row in reader
            if row.get("language") != language
        ]

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


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


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    language_code: str,
) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language=language_code,
        beam_size=5,
        vad_filter=True,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()


def main() -> None:
    args = parse_arguments()

    config = LANGUAGE_CONFIG[args.language]
    language_code = config["code"]
    model_id = MODEL_IDS[args.language]

    if args.run_number < 1:
        raise ValueError(
            "--run-number must be at least 1."
        )

    if args.overwrite_language:
        remove_existing_language_rows(
            args.language
        )

    sentences = load_sentences(
        config["test_file"]
    )

    print("=" * 72)
    print("Batch round-trip WER/CER evaluation")
    print("=" * 72)
    print(f"Language: {args.language}")
    print(f"TTS model: {model_id}")
    print(f"ASR model: {args.asr_model}")
    print(f"Sentence count: {len(sentences)}")
    print()

    print("Loading faster-whisper...")

    asr_model = WhisperModel(
        args.asr_model,
        device="cpu",
        compute_type="int8",
    )

    print("ASR model loaded.")
    print()

    successful_scores = []
    successful_cer_scores = []
    failures = 0

    for sentence in sentences:
        sentence_id = sentence["id"]
        category = sentence["category"]
        reference_text = sentence["text"]

        audio_filename = (
            f"{sentence_id}_run_"
            f"{args.run_number:02d}.wav"
        )

        audio_path = (
            config["audio_dir"]
            / audio_filename
        )

        print(
            f"Evaluating {sentence_id}: "
            f"{audio_filename}"
        )

        success = True
        error_message = ""
        transcript = ""
        normalized_reference = ""
        normalized_transcript = ""
        wer_score = None
        cer_score = None

        try:
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            transcript = transcribe_audio(
                model=asr_model,
                audio_path=audio_path,
                language_code=language_code,
            )

            normalized_reference = normalize_text(
                reference_text,
                language_code,
            )

            normalized_transcript = normalize_text(
                transcript,
                language_code,
            )

            wer_score = wer(
                normalized_reference,
                normalized_transcript,
            )

            cer_score = cer(
                normalized_reference,
                normalized_transcript,
            )

            successful_scores.append(wer_score)
            successful_cer_scores.append(cer_score)

            print(f"  Reference: {reference_text}")
            print(f"  Transcript: {transcript}")
            print(f"  WER: {wer_score:.4f}")
            print(f"  CER: {cer_score:.4f}")

        except Exception as error:
            success = False
            failures += 1
            error_message = (
                f"{type(error).__name__}: {error}"
            )

            print(f"  FAILED: {error_message}")

        result_row = {
            "language": args.language,
            "language_code": language_code,
            "model_id": model_id,
            "sentence_id": sentence_id,
            "category": category,
            "audio_file": (
                str(audio_path)
                if audio_path.exists()
                else ""
            ),
            "reference_text": reference_text,
            "asr_transcript": transcript,
            "normalized_reference": normalized_reference,
            "normalized_transcript": normalized_transcript,
            "wer": (
                round(wer_score, 4)
                if wer_score is not None
                else ""
            ),
            "cer": (
                round(cer_score, 4)
                if cer_score is not None
                else ""
            ),
            "asr_model": args.asr_model,
            "success": success,
            "error_message": error_message,
        }

        append_result(result_row)
        print()

    print("=" * 72)
    print("WER evaluation completed")
    print("=" * 72)

    if successful_scores:
        mean_wer = statistics.mean(
            successful_scores
        )
        median_wer = statistics.median(
            successful_scores
        )
        mean_cer = statistics.mean(
            successful_cer_scores
        )

        pass_count = sum(
            score <= 0.10
            for score in successful_scores
        )

        pass_rate = (
            pass_count
            / len(successful_scores)
            * 100
        )

        print(f"Mean WER: {mean_wer:.4f}")
        print(f"Median WER: {median_wer:.4f}")
        print(f"Mean CER: {mean_cer:.4f}")
        print(
            f"WER target pass rate: "
            f"{pass_rate:.2f}%"
        )

    print(f"Failures: {failures}")
    print(f"Saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()