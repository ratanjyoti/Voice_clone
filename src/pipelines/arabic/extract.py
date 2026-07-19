from __future__ import annotations
import io

import csv
import json
import math
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "arabic"
    / "professional_msa"
)

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
METADATA_PATH = DATA_ROOT / "metadata_all.csv"
SUMMARY_PATH = DATA_ROOT / "extraction_summary.json"

TARGET_SAMPLE_RATE = 24000
SILENCE_THRESHOLD = 0.01


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_audio_array(
    audio_value: object,
) -> tuple[np.ndarray, int]:
    if not isinstance(audio_value, dict):
        raise TypeError(
            "Unexpected audio value type: "
            f"{type(audio_value)}"
        )

    audio_bytes = audio_value.get("bytes")
    audio_path = audio_value.get("path")

    if audio_bytes:
        waveform, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=True,
        )
    elif audio_path:
        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
    else:
        raise ValueError(
            "Audio item contains neither bytes nor path."
        )

    if waveform.size == 0:
        raise ValueError(
            "Decoded audio is empty."
        )

    # soundfile returns shape:
    # samples × channels
    if waveform.shape[1] > 1:
        waveform = np.mean(
            waveform,
            axis=1,
        )
    else:
        waveform = waveform[:, 0]

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).squeeze()

    if waveform.ndim != 1:
        raise ValueError(
            "Expected mono waveform, found shape "
            f"{waveform.shape}"
        )

    return waveform, int(sample_rate)

def linear_resample(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32)

    if waveform.size == 0:
        return waveform.astype(np.float32)

    source_duration = waveform.size / source_rate
    target_length = max(
        1,
        int(round(source_duration * target_rate)),
    )

    old_positions = np.linspace(
        0.0,
        1.0,
        num=waveform.size,
        endpoint=False,
    )
    new_positions = np.linspace(
        0.0,
        1.0,
        num=target_length,
        endpoint=False,
    )

    return np.interp(
        new_positions,
        old_positions,
        waveform,
    ).astype(np.float32)


def calculate_stats(
    waveform: np.ndarray,
    sample_rate: int,
) -> dict:
    absolute = np.abs(waveform)

    peak = (
        float(np.max(absolute))
        if waveform.size
        else 0.0
    )

    rms = (
        float(np.sqrt(np.mean(np.square(waveform))))
        if waveform.size
        else 0.0
    )

    hard_clip_samples = int(
        np.sum(absolute >= 1.0)
    )

    silence_ratio = (
        float(np.mean(absolute < SILENCE_THRESHOLD))
        if waveform.size
        else 1.0
    )

    duration = (
        float(waveform.size / sample_rate)
        if sample_rate > 0
        else 0.0
    )

    return {
        "duration_seconds": duration,
        "sample_rate": int(sample_rate),
        "channels": 1,
        "peak": peak,
        "rms": rms,
        "hard_clip_samples": hard_clip_samples,
        "silence_ratio": silence_ratio,
    }


def evaluate_usability(
    stats: dict,
    transcript: str,
) -> tuple[bool, str]:
    reasons = []

    if not transcript.strip():
        reasons.append("empty transcript")

    duration = stats["duration_seconds"]

    if duration < 2.5:
        reasons.append("duration below 2.5 seconds")

    if duration > 15.0:
        reasons.append("duration above 15 seconds")

    if stats["rms"] <= 0.01:
        reasons.append("RMS too low")

    if stats["hard_clip_samples"] > 0:
        reasons.append("hard clipping detected")

    if stats["silence_ratio"] >= 0.30:
        reasons.append("silence ratio too high")

    return not reasons, "; ".join(reasons)


def find_text_column(column_names: list[str]) -> str:
    candidates = [
        "text",
        "sentence",
        "transcription",
        "transcript",
        "normalized_text",
    ]

    lowered = {
        name.lower(): name
        for name in column_names
    }

    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    raise KeyError(
        "Could not find a transcript column. "
        f"Available columns: {column_names}"
    )


def main() -> None:
    ensure_directories()

    print("=" * 72)
    print("Arabic professional voice dataset extraction")
    print("=" * 72)

    dataset = load_dataset(
        "NightPrince/Arabic-professional-voice",
        split="train",
    )

    if "audio" not in dataset.column_names:
        raise KeyError(
            f"No audio column found. Columns: {dataset.column_names}"
        )

    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False),
    )

    text_column = find_text_column(
        list(dataset.column_names)
    )

    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"Transcript column: {text_column}")

    fieldnames = [
        "dataset_row",
        "file_name",
        "transcription",
        "speaker_name",
        "duration_seconds",
        "sample_rate",
        "channels",
        "peak",
        "rms",
        "hard_clip_samples",
        "silence_ratio",
        "usable",
        "rejection_reason",
        "error",
    ]

    rows: list[dict] = []
    success_count = 0
    usable_count = 0
    failed_count = 0

    for index, item in enumerate(dataset):
        file_name = f"clip_{index:06d}.wav"
        output_path = PROCESSED_DIR / file_name

        record = {
            "dataset_row": index,
            "file_name": file_name,
            "transcription": str(
                item.get(text_column, "")
            ).strip(),
            "speaker_name": "arabic_professional_male",
            "usable": "No",
            "rejection_reason": "",
            "error": "",
        }

        try:
            waveform, source_rate = get_audio_array(
                item["audio"]
            )

            if not np.isfinite(waveform).all():
                raise ValueError(
                    "Audio contains NaN or infinite samples."
                )

            waveform = linear_resample(
                waveform,
                source_rate,
                TARGET_SAMPLE_RATE,
            )

            waveform = np.clip(
                waveform,
                -1.0,
                1.0,
            )

            sf.write(
                str(output_path),
                waveform,
                TARGET_SAMPLE_RATE,
                subtype="PCM_16",
            )

            saved_audio, saved_rate = sf.read(
                str(output_path),
                dtype="float32",
                always_2d=False,
            )

            saved_audio = np.asarray(
                saved_audio,
                dtype=np.float32,
            ).squeeze()

            stats = calculate_stats(
                saved_audio,
                saved_rate,
            )

            usable, reason = evaluate_usability(
                stats,
                record["transcription"],
            )

            record.update(stats)
            record["usable"] = "Yes" if usable else "No"
            record["rejection_reason"] = reason

            success_count += 1

            if usable:
                usable_count += 1

        except Exception as exc:
            failed_count += 1
            record["error"] = str(exc)
            record["rejection_reason"] = "extraction failed"

            print(
                f"FAILED row {index}: {exc}"
            )

        rows.append(record)

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(dataset)}"
            )

    with METADATA_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "NightPrince/Arabic-professional-voice",
        "total_rows": len(dataset),
        "successfully_extracted": success_count,
        "usable_clips": usable_count,
        "rejected_or_failed": len(dataset) - usable_count,
        "failed_extractions": failed_count,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "processed_directory": str(PROCESSED_DIR),
        "metadata_path": str(METADATA_PATH),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("Extraction completed")
    print("=" * 72)
    print(
        f"Successfully extracted: {success_count}"
    )
    print(f"Usable clips: {usable_count}")
    print(f"Failed extractions: {failed_count}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise