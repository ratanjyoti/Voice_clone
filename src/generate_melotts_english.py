import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NLTK_DATA_DIR = PROJECT_ROOT / "data" / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))

import numpy as np
import soundfile as sf
import torch
from melo.api import TTS


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "melotts" / "english"
REPORT_DIR = PROJECT_ROOT / "evidence" / "result_snapshots" / "melotts"
CSV_PATH = PROJECT_ROOT / "results" / "raw_runs.csv"


TEST_SENTENCES = [
    {
        "id": "en_01",
        "text": "Hello, welcome to Infinia. How can I help you today?",
    },
    {
        "id": "en_02",
        "text": "Fast response time is important for natural voice conversations.",
    },
    {
        "id": "en_03",
        "text": "My name is Alex, and I will assist you with your request.",
    },
    {
        "id": "en_04",
        "text": "Your order number is five eight two four, and it will arrive on July twenty first.",
    },
    {
        "id": "en_05",
        "text": (
            "Thank you for contacting customer support. "
            "I have reviewed your request and will explain the next steps clearly."
        ),
    },
]


def ensure_nltk_data():
    import nltk

    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    packages = (
        "cmudict",
        "averaged_perceptron_tagger_eng",
        "averaged_perceptron_tagger",
    )

    for package in packages:
        try:
            nltk.data.find(package)
        except LookupError:
            print(f"Downloading NLTK package: {package}")
            nltk.download(
                package,
                download_dir=str(NLTK_DATA_DIR),
                quiet=False,
            )


def wav_stats(path: Path):
    data, sample_rate = sf.read(str(path), always_2d=True)

    if data.size == 0:
        raise RuntimeError("Generated WAV contains no samples")

    finite = np.isfinite(data)
    duration = data.shape[0] / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(data))))
    peak = float(np.max(np.abs(data)))

    return {
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "frames": int(data.shape[0]),
        "duration_seconds": duration,
        "rms": rms,
        "peak": peak,
        "all_finite": bool(finite.all()),
        "size_bytes": int(path.stat().st_size),
    }


def validate_wav(path: Path, stats: dict):
    if not path.exists():
        raise RuntimeError(f"Output WAV was not created: {path}")

    if path.stat().st_size <= 44:
        raise RuntimeError(f"Output WAV is empty or header-only: {path}")

    if stats["duration_seconds"] <= 0:
        raise RuntimeError("Generated WAV duration is not positive")

    if not stats["all_finite"]:
        raise RuntimeError("Generated WAV contains NaN or infinite samples")

    if stats["rms"] <= 0.0005:
        raise RuntimeError(
            f"Generated WAV appears silent; RMS={stats['rms']}"
        )


def append_result_to_csv(result: dict):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pipeline",
        "model",
        "language",
        "sentence_id",
        "text",
        "speaker",
        "device",
        "status",
        "model_load_time_seconds",
        "generation_time_seconds",
        "audio_duration_seconds",
        "rtf",
        "sample_rate",
        "channels",
        "rms",
        "peak",
        "output_path",
        "report_path",
        "error",
    ]

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(
            {
                "pipeline": "melotts",
                "model": result.get("model", ""),
                "language": result.get("language", ""),
                "sentence_id": result.get("sentence_id", ""),
                "text": result.get("text", ""),
                "speaker": result.get("speaker", ""),
                "device": result.get("device", ""),
                "status": result.get("status", ""),
                "model_load_time_seconds": result.get(
                    "model_load_time_seconds", ""
                ),
                "generation_time_seconds": result.get(
                    "synthesis_time_seconds", ""
                ),
                "audio_duration_seconds": result.get(
                    "duration_seconds", ""
                ),
                "rtf": result.get("rtf", ""),
                "sample_rate": result.get("sample_rate", ""),
                "channels": result.get("channels", ""),
                "rms": result.get("rms", ""),
                "peak": result.get("peak", ""),
                "output_path": result.get("output_path", ""),
                "report_path": result.get("report_path", ""),
                "error": result.get("error", ""),
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate five English MeloTTS benchmark WAV files on CPU."
    )
    parser.add_argument(
        "--speaker",
        default="EN-Default",
        help="MeloTTS English speaker name.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MeloTTS English batch benchmark")
    print("=" * 70)
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Device: CPU")
    print(f"Number of sentences: {len(TEST_SENTENCES)}")

    ensure_nltk_data()

    print("\nLoading MeloTTS English model once...")

    load_start = time.perf_counter()
    model = TTS(language="EN", device="cpu")
    model_load_time = time.perf_counter() - load_start

    speaker_ids = dict(model.hps.data.spk2id)

    if args.speaker not in speaker_ids:
        raise KeyError(
            f"Speaker {args.speaker!r} not found. "
            f"Available speakers: {sorted(speaker_ids)}"
        )

    speaker_id = speaker_ids[args.speaker]

    print(f"Model loaded in {model_load_time:.4f} seconds")
    print(f"Available speakers: {sorted(speaker_ids)}")
    print(f"Selected speaker: {args.speaker} ({speaker_id})")

    successful = 0
    failed = 0
    batch_start = time.perf_counter()

    for index, sample in enumerate(TEST_SENTENCES, start=1):
        sentence_id = sample["id"]
        text = sample["text"]

        output_path = OUTPUT_DIR / f"{sentence_id}_melotts.wav"
        report_path = REPORT_DIR / f"{sentence_id}_melotts.json"

        print("\n" + "-" * 70)
        print(f"[{index}/{len(TEST_SENTENCES)}] Generating {sentence_id}")
        print(f"Text: {text}")
        print(f"Output: {output_path}")

        result = {
            "status": "failed",
            "pipeline": "melotts",
            "model": "myshell-ai/MeloTTS-English",
            "language": "english",
            "sentence_id": sentence_id,
            "speaker": args.speaker,
            "speaker_id": int(speaker_id),
            "device": "cpu",
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "text": text,
            "speed": args.speed,
            "output_path": str(output_path),
            "report_path": str(report_path),
            "model_load_time_seconds": model_load_time,
        }

        try:
            synthesis_start = time.perf_counter()

            model.tts_to_file(
                text,
                speaker_id,
                str(output_path),
                speed=args.speed,
                quiet=True,
            )

            synthesis_time = time.perf_counter() - synthesis_start

            if not output_path.exists():
                raise RuntimeError(
                    f"Output WAV was not created: {output_path}"
                )

            stats = wav_stats(output_path)
            validate_wav(output_path, stats)

            rtf = synthesis_time / stats["duration_seconds"]

            result.update(
                {
                    "status": "success",
                    "synthesis_time_seconds": synthesis_time,
                    "rtf": rtf,
                    **stats,
                }
            )

            successful += 1

            print(f"Status: SUCCESS")
            print(
                f"Synthesis time: {synthesis_time:.4f} seconds"
            )
            print(
                f"Audio duration: "
                f"{stats['duration_seconds']:.4f} seconds"
            )
            print(f"RTF: {rtf:.4f}")
            print(f"RMS: {stats['rms']:.8f}")
            print(f"Peak: {stats['peak']:.8f}")

        except Exception as exc:
            failed += 1

            result.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

            print(f"Status: FAILED")
            print(f"Error: {exc}")
            print("Continuing to the next sentence...")

        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        append_result_to_csv(result)

        print(f"Report: {report_path}")

    total_batch_time = time.perf_counter() - batch_start

    summary = {
        "pipeline": "melotts",
        "model": "myshell-ai/MeloTTS-English",
        "language": "english",
        "device": "cpu",
        "speaker": args.speaker,
        "model_load_time_seconds": model_load_time,
        "total_sentences": len(TEST_SENTENCES),
        "successful_sentences": successful,
        "failed_sentences": failed,
        "total_batch_time_seconds": total_batch_time,
        "output_directory": str(OUTPUT_DIR),
        "csv_path": str(CSV_PATH),
    }

    summary_path = REPORT_DIR / "melotts_english_batch_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("Batch generation completed")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total sentences: {len(TEST_SENTENCES)}")
    print(f"Model load time: {model_load_time:.4f} seconds")
    print(f"Batch generation time: {total_batch_time:.4f} seconds")
    print(f"Audio directory: {OUTPUT_DIR}")
    print(f"Results CSV: {CSV_PATH}")
    print(f"Summary JSON: {summary_path}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()