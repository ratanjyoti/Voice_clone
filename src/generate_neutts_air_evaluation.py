import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from neutts import NeuTTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_AUDIO = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "english"
    / "ratan_neutral.wav"
)

REFERENCE_TEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "english"
    / "ratan_neutral.txt"
)

REFERENCE_CODES_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "english"
    / "ratan_neutral.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "neutts"
    / "english"
    / "air_evaluation"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "neutts_air"
)

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "neutts_air_evaluation.csv"
)

MODEL_NAME = "neuphonic/neutts-air"
CODEC_NAME = "neuphonic/neucodec"
SAMPLE_RATE = 24000


TEST_SENTENCES = [
    {
        "id": "air_01_greeting",
        "display_text": (
            "Hello, welcome to Infinia. "
            "I am ready to help you today."
        ),
        "spoken_text": (
            "Hello, welcome to In-fin-ee-uh. "
            "I am ready to help you today."
        ),
        "category": "greeting",
    },
    {
        "id": "air_02_identity",
        "display_text": (
            "My name is Ratan Jyoti, and I enjoy "
            "building practical artificial intelligence systems."
        ),
        "spoken_text": (
            "My name is Ruh tun Jyo tee, and I enjoy "
            "building practical artificial intelligence systems."
        ),
        "category": "identity",
    },
    {
        "id": "air_03_support",
        "display_text": (
            "Thank you for contacting customer support. "
            "I have reviewed your request and will guide "
            "you through the next steps."
        ),
        "spoken_text": (
            "Thank you for contacting customer support. "
            "I have reviewed your request and will guide "
            "you through the next steps."
        ),
        "category": "customer_support",
    },
    {
        "id": "air_04_numbers",
        "display_text": (
            "Your order number is five eight two four, "
            "and it will arrive on July twenty-first."
        ),
        "spoken_text": (
            "Your order number is five eight two four, "
            "and it will arrive on July twenty-first."
        ),
        "category": "numbers_and_date",
    },
    {
        "id": "air_05_expressive",
        "display_text": (
            "That is wonderful news! Everything worked "
            "correctly, and I hope you have a great day."
        ),
        "spoken_text": (
            "That is wonderful news! Everything worked "
            "correctly, and I hope you have a great day."
        ),
        "category": "expressive",
    },
]


def wav_stats(path: Path) -> dict:
    audio, sample_rate = sf.read(
        str(path),
        always_2d=True,
    )

    if audio.size == 0:
        raise RuntimeError(
            f"Generated WAV has no samples: {path}"
        )

    duration = audio.shape[0] / float(sample_rate)
    rms = float(
        np.sqrt(np.mean(np.square(audio)))
    )
    peak = float(np.max(np.abs(audio)))

    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_seconds": duration,
        "rms": rms,
        "peak": peak,
        "all_finite": bool(
            np.isfinite(audio).all()
        ),
        "size_bytes": int(path.stat().st_size),
    }


def validate_wav(path: Path, stats: dict) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Output was not created: {path}"
        )

    if path.stat().st_size <= 44:
        raise RuntimeError(
            f"Output is empty or header-only: {path}"
        )

    if stats["duration_seconds"] <= 0:
        raise RuntimeError(
            "Generated audio duration is invalid."
        )

    if not stats["all_finite"]:
        raise RuntimeError(
            "Generated audio contains NaN or infinity."
        )

    if stats["rms"] <= 0.0005:
        raise RuntimeError(
            f"Generated audio appears silent: "
            f"RMS={stats['rms']}"
        )


def write_csv(results: list[dict]) -> None:
    CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sentence_id",
        "category",
        "display_text",
        "spoken_text",
        "status",
        "model",
        "model_load_time_seconds",
        "reference_encoding_time_seconds",
        "synthesis_time_seconds",
        "duration_seconds",
        "rtf",
        "sample_rate",
        "channels",
        "rms",
        "peak",
        "output_path",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field: result.get(field, "")
                    for field in fieldnames
                }
            )


def main() -> None:
    if not REFERENCE_AUDIO.exists():
        raise FileNotFoundError(
            f"Reference audio missing: "
            f"{REFERENCE_AUDIO}"
        )

    if not REFERENCE_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"Reference transcript missing: "
            f"{REFERENCE_TEXT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_text = (
        REFERENCE_TEXT_PATH
        .read_text(encoding="utf-8")
        .strip()
    )

    print("=" * 72)
    print("NeuTTS Air five-sentence evaluation")
    print("=" * 72)
    print(f"Model: {MODEL_NAME}")
    print(f"Reference: {REFERENCE_AUDIO}")
    print(f"Sentences: {len(TEST_SENTENCES)}")
    print("Device: CPU")

    print("\nLoading NeuTTS Air once...")

    load_start = time.perf_counter()

    tts = NeuTTS(
        backbone_repo=MODEL_NAME,
        backbone_device="cpu",
        codec_repo=CODEC_NAME,
        codec_device="cpu",
    )

    model_load_time = (
        time.perf_counter() - load_start
    )

    print(
        f"Model loaded in "
        f"{model_load_time:.4f} seconds"
    )

    print("\nEncoding reference once...")

    reference_start = time.perf_counter()

    # The WAV is deliberately encoded once for this
    # process so all five sentences use the same codes.
    reference_codes = tts.encode_reference(
        str(REFERENCE_AUDIO)
    )

    reference_encoding_time = (
        time.perf_counter()
        - reference_start
    )

    print(
        f"Reference encoded in "
        f"{reference_encoding_time:.4f} seconds"
    )

    results = []
    successful = 0
    failed = 0
    batch_start = time.perf_counter()

    for index, sample in enumerate(
        TEST_SENTENCES,
        start=1,
    ):
        sentence_id = sample["id"]
        output_path = (
            OUTPUT_DIR
            / f"{sentence_id}.wav"
        )
        report_path = (
            REPORT_DIR
            / f"{sentence_id}.json"
        )

        print("\n" + "-" * 72)
        print(
            f"[{index}/{len(TEST_SENTENCES)}] "
            f"{sentence_id}"
        )
        print(
            f"Displayed: {sample['display_text']}"
        )
        print(
            f"Spoken: {sample['spoken_text']}"
        )

        result = {
            "sentence_id": sentence_id,
            "category": sample["category"],
            "display_text": (
                sample["display_text"]
            ),
            "spoken_text": (
                sample["spoken_text"]
            ),
            "status": "failed",
            "model": MODEL_NAME,
            "codec": CODEC_NAME,
            "device": "cpu",
            "model_load_time_seconds": (
                model_load_time
            ),
            "reference_encoding_time_seconds": (
                reference_encoding_time
            ),
            "reference_audio": str(
                REFERENCE_AUDIO
            ),
            "reference_text": reference_text,
            "output_path": str(output_path),
            "report_path": str(report_path),
        }

        try:
            synthesis_start = time.perf_counter()

            waveform = tts.infer(
                sample["spoken_text"],
                reference_codes,
                reference_text,
            )

            synthesis_time = (
                time.perf_counter()
                - synthesis_start
            )

            waveform = np.asarray(
                waveform,
                dtype=np.float32,
            ).squeeze()

            if waveform.size == 0:
                raise RuntimeError(
                    "NeuTTS returned empty audio."
                )

            sf.write(
                str(output_path),
                waveform,
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            stats = wav_stats(output_path)
            validate_wav(output_path, stats)

            rtf = (
                synthesis_time
                / stats["duration_seconds"]
            )

            result.update(
                {
                    "status": "success",
                    "synthesis_time_seconds": (
                        synthesis_time
                    ),
                    "rtf": rtf,
                    **stats,
                }
            )

            successful += 1

            print("Status: SUCCESS")
            print(
                f"Synthesis: "
                f"{synthesis_time:.4f}s"
            )
            print(
                f"Duration: "
                f"{stats['duration_seconds']:.4f}s"
            )
            print(f"RTF: {rtf:.4f}")

        except Exception as exc:
            failed += 1

            result.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )

            print("Status: FAILED")
            print(f"Error: {exc}")
            print(
                "Continuing to the next sentence."
            )

        report_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        results.append(result)

    batch_time = (
        time.perf_counter() - batch_start
    )

    write_csv(results)

    summary = {
        "status": (
            "success"
            if failed == 0
            else "partial"
        ),
        "model": MODEL_NAME,
        "codec": CODEC_NAME,
        "device": "cpu",
        "total_sentences": len(
            TEST_SENTENCES
        ),
        "successful": successful,
        "failed": failed,
        "model_load_time_seconds": (
            model_load_time
        ),
        "reference_encoding_time_seconds": (
            reference_encoding_time
        ),
        "batch_time_seconds": batch_time,
        "output_directory": str(OUTPUT_DIR),
        "csv_path": str(CSV_PATH),
    }

    summary_path = (
        REPORT_DIR
        / "neutts_air_evaluation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("NeuTTS Air evaluation completed")
    print("=" * 72)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"CSV: {CSV_PATH}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()