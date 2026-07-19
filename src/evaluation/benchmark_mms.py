import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write
from transformers import AutoTokenizer, VitsModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANGUAGE_CONFIG = {
    "english": {
        "language_code": "en",
        "model_id": "facebook/mms-tts-eng",
        "test_file": PROJECT_ROOT / "data" / "test_sentences" / "english.json",
        "output_dir": PROJECT_ROOT / "outputs" / "mms" / "english",
    },
    "arabic": {
        "language_code": "ar",
        "model_id": "facebook/mms-tts-ara",
        "test_file": PROJECT_ROOT / "data" / "test_sentences" / "arabic.json",
        "output_dir": PROJECT_ROOT / "outputs" / "mms" / "arabic",
    },
    "hindi": {
        "language_code": "hi",
        "model_id": "facebook/mms-tts-hin",
        "test_file": PROJECT_ROOT / "data" / "test_sentences" / "hindi.json",
        "output_dir": PROJECT_ROOT / "outputs" / "mms" / "hindi",
    },
}

RESULTS_FILE = PROJECT_ROOT / "results" / "raw_runs.csv"

CSV_FIELDS = [
    "timestamp",
    "language",
    "language_code",
    "model_id",
    "sentence_id",
    "category",
    "run_number",
    "input_text",
    "output_file",
    "device",
    "sample_rate_hz",
    "model_load_time_seconds",
    "generation_time_seconds",
    "audio_duration_seconds",
    "rtf",
    "python_version",
    "pytorch_version",
    "platform",
    "success",
    "error_message",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MMS-TTS on all test sentences."
    )

    parser.add_argument(
        "--language",
        choices=LANGUAGE_CONFIG.keys(),
        required=True,
        help="Language to benchmark.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per sentence.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm-up runs before measured inference.",
    )
    parser.add_argument(
    "--overwrite-language",
    action="store_true",
    help=(
        "Remove previous CSV rows for the selected language "
        "before starting the benchmark."
    ),
)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_test_sentences(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError(f"No test sentences found in {path}")

    return records


def append_result(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_FILE.exists()
    needs_header = not file_exists or RESULTS_FILE.stat().st_size == 0

    with RESULTS_FILE.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)

        if needs_header:
            writer.writeheader()

        writer.writerow(row)

def remove_existing_language_rows(language: str) -> None:
    if not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0:
        return

    with RESULTS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row.get("language") != language]

    with RESULTS_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def generate_audio(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    synchronize(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model(**inputs)

    synchronize(device)

    generation_time = time.perf_counter() - start_time

    waveform = (
        output.waveform
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    sample_rate = int(model.config.sampling_rate)

    return waveform, sample_rate, generation_time

def main() -> None:
    args = parse_arguments()
    if args.overwrite_language:
        remove_existing_language_rows(args.language)

    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")

    config = LANGUAGE_CONFIG[args.language]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sentences = load_test_sentences(config["test_file"])
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MMS-TTS batch benchmark")
    print("=" * 72)
    print(f"Language: {args.language}")
    print(f"Model: {config['model_id']}")
    print(f"Device: {device}")
    print(f"Sentences: {len(sentences)}")
    print(f"Warm-up runs: {args.warmup_runs}")
    print(f"Measured runs per sentence: {args.runs}")
    print()

    print("Loading model...")

    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"]
    )

    model = VitsModel.from_pretrained(
        config["model_id"]
    )

    model = model.to(device)
    model.eval()

    synchronize(device)

    load_time = time.perf_counter() - load_start

    print(f"Model loaded in {load_time:.4f} seconds.")
    print()

    warmup_text = sentences[0]["text"]

    for warmup_index in range(1, args.warmup_runs + 1):
        print(
            f"Warm-up run {warmup_index}/"
            f"{args.warmup_runs}..."
        )

        generate_audio(
            model=model,
            tokenizer=tokenizer,
            text=warmup_text,
            device=device,
        )

    print()
    print("Starting measured runs...")
    print()

    successful_runs = 0
    failed_runs = 0

    for sentence in sentences:
        sentence_id = sentence["id"]
        category = sentence["category"]
        text = sentence["text"]

        for run_number in range(1, args.runs + 1):
            output_filename = (
                f"{sentence_id}_run_{run_number:02d}.wav"
            )

            output_path = output_dir / output_filename

            print(
                f"[{sentence_id}] "
                f"Run {run_number}/{args.runs}"
            )

            error_message = ""
            success = True

            try:
                waveform, sample_rate, generation_time = (
                    generate_audio(
                        model=model,
                        tokenizer=tokenizer,
                        text=text,
                        device=device,
                    )
                )

                write(
                    filename=str(output_path),
                    rate=sample_rate,
                    data=waveform,
                )

                audio_duration = len(waveform) / sample_rate

                rtf = (
                    generation_time / audio_duration
                    if audio_duration > 0
                    else None
                )

                successful_runs += 1

                print(
                    f"  Generation: "
                    f"{generation_time:.4f} s"
                )
                print(
                    f"  Audio: "
                    f"{audio_duration:.4f} s"
                )
                print(f"  RTF: {rtf:.4f}")
                print(f"  Saved: {output_path.name}")

            except Exception as error:
                success = False
                failed_runs += 1
                error_message = (
                    f"{type(error).__name__}: {error}"
                )

                generation_time = None
                audio_duration = None
                rtf = None
                sample_rate = None

                print(f"  FAILED: {error_message}")

            result_row = {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "language": args.language,
                "language_code": config["language_code"],
                "model_id": config["model_id"],
                "sentence_id": sentence_id,
                "category": category,
                "run_number": run_number,
                "input_text": text,
                "output_file": (
                    str(output_path)
                    if success
                    else ""
                ),
                "device": str(device),
                "sample_rate_hz": sample_rate,
                "model_load_time_seconds": round(
                    load_time,
                    4,
                ),
                "generation_time_seconds": (
                    round(generation_time, 4)
                    if generation_time is not None
                    else ""
                ),
                "audio_duration_seconds": (
                    round(audio_duration, 4)
                    if audio_duration is not None
                    else ""
                ),
                "rtf": (
                    round(rtf, 4)
                    if rtf is not None
                    else ""
                ),
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
                "platform": platform.platform(),
                "success": success,
                "error_message": error_message,
            }

            append_result(result_row)
            print()

    print("=" * 72)
    print("Benchmark completed")
    print("=" * 72)
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Results CSV: {RESULTS_FILE}")
    print(f"Audio directory: {output_dir}")


if __name__ == "__main__":
    main()

