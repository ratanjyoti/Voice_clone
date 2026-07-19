from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "english" / "melotts" / "final"
RESULT_PATH = PROJECT_ROOT / "results" / "raw" / "melotts_english_evaluation.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary" / "melotts_english_summary.csv"
COMPARISON_PATH = PROJECT_ROOT / "results" / "summary" / "english_model_comparison.csv"
LOG_SNAPSHOT_DIR = PROJECT_ROOT / "evidence" / "result_snapshots" / "melotts"

PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def audio_stats(path: Path) -> dict:
    audio, sr = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)

    return {
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "audio_duration": float(len(mono) / sr),
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
        "peak": float(abs_audio.max()),
        "clipping": bool(np.any(abs_audio >= 0.999)),
        "clipping_samples": int(np.sum(abs_audio >= 0.999)),
        "silence_ratio": float(np.mean(abs_audio < 0.001)),
    }


def transcribe(model: WhisperModel, path: Path) -> str:
    segments, _info = model.transcribe(
        str(path),
        language="en",
        beam_size=5,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def round_value(value: float, places: int = 3) -> float:
    return round(float(value), places)


def write_summary(rows: list[dict]) -> dict:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = len(rows)
    summary = {
        "model": "MeloTTS",
        "language": "English",
        "samples": samples,
        "average_WER_percent": round_value(sum(float(row["WER_percent"]) for row in rows) / samples, 2),
        "average_RTF": round_value(sum(float(row["RTF"]) for row in rows) / samples, 3),
        "average_generation_time": round_value(sum(float(row["generation_time"]) for row in rows) / samples, 3),
        "average_audio_duration": round_value(sum(float(row["audio_duration"]) for row in rows) / samples, 3),
        "max_peak": round_value(max(float(row["peak"]) for row in rows), 3),
        "total_clipping_samples": int(sum(int(row["clipping_samples"]) for row in rows)),
        "status": "Final English winner if manual listening confirms naturalness",
    }
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def write_comparison(summary: dict) -> None:
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison = [
        {
            "model": "MMS-TTS",
            "language": "English",
            "WER_percent": "8.0",
            "RTF": "<0.3",
            "status": "Fast stable baseline, robotic voice",
        },
        {
            "model": "NeuTTS",
            "language": "English",
            "WER_percent": "12.67",
            "RTF": "Not finalized",
            "status": "Failed WER target / unstable final generation",
        },
        {
            "model": "MeloTTS",
            "language": "English",
            "WER_percent": summary["average_WER_percent"],
            "RTF": summary["average_RTF"],
            "status": "Final English winner if naturalness confirmed",
        },
    ]
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)


def main() -> None:
    summary_json = OUTPUT_DIR / "generation_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing generation summary: {summary_json}")

    items = json.loads(summary_json.read_text(encoding="utf-8"))
    if not items:
        raise RuntimeError("Generation summary is empty")

    print("Loading Whisper model for English WER...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in items:
        output_path = Path(item["output_path"])
        if not output_path.exists():
            candidate = OUTPUT_DIR / output_path.name
            if candidate.exists():
                output_path = candidate
            else:
                raise FileNotFoundError(f"Missing generated audio: {output_path}")

        expected_text = item["text"]
        asr_text = transcribe(model, output_path)
        normalized_expected = normalize_text(expected_text)
        normalized_asr = normalize_text(asr_text)
        wer_score = wer(normalized_expected, normalized_asr)
        stats = audio_stats(output_path)

        row = {
            "test_id": item["test_id"],
            "model": "MeloTTS",
            "language": "English",
            "expected_text": expected_text,
            "asr_text": asr_text,
            "normalized_expected": normalized_expected,
            "normalized_asr": normalized_asr,
            "WER": round_value(wer_score, 6),
            "WER_percent": round_value(wer_score * 100, 2),
            "generation_time": round_value(item.get("generation_time", 0.0), 6),
            "RTF": round_value(item.get("RTF", 0.0), 6),
            "output_path": str(output_path),
            **stats,
        }
        rows.append(row)
        (LOG_SNAPSHOT_DIR / f"{item['test_id']}_evaluation.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False, indent=2))

    fieldnames = list(rows[0].keys())
    with RESULT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = write_summary(rows)
    write_comparison(summary)

    print(f"Evaluation saved to: {RESULT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Comparison saved to: {COMPARISON_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
