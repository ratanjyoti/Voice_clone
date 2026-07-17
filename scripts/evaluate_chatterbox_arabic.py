from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_PATH = PROJECT_ROOT / "data" / "test_sentences" / "arabic_voice_clone_tests.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chatterbox" / "arabic"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = PROJECT_ROOT / "evidence" / "result_snapshots" / "chatterbox" / "arabic" / "chatterbox_arabic_summary.json"
MODEL_NAME = "ResembleAI/chatterbox multilingual"
DEVICE = "cpu"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_PUNCTUATION = re.compile(r"[،؛؟.!«»\"'ـ,:;?\-()\[\]{}]")
HUMAN_REVIEW_FIELDS = [
    "similarity_score_1_to_5",
    "naturalness_score_1_to_5",
    "pronunciation_score_1_to_5",
    "metallic",
    "missing_words",
    "repeated_words",
    "accepted",
    "reviewer_notes",
]


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


def load_tests() -> dict[str, dict[str, Any]]:
    data = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
    return {test["id"]: test for test in data["tests"]}


def read_generation_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sidecar_for(audio_path: Path) -> dict[str, Any]:
    path = audio_path.with_suffix(".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def audio_stats(audio_path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    return {
        "duration": duration,
        "sample_rate": int(sample_rate),
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0,
        "peak": float(abs_audio.max()) if mono.size else 0.0,
        "clipping_samples": int(np.sum(abs_audio >= 0.999)) if mono.size else 0,
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if mono.size else 1.0,
    }


def transcribe(model: WhisperModel, audio_path: Path) -> tuple[str, float]:
    segments, info = model.transcribe(
        str(audio_path),
        language="ar",
        beam_size=5,
        vad_filter=True,
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    return transcript, float(info.language_probability)


def evaluate_items(items: list[dict[str, Any]], output_csv: Path, write_summary: bool = False) -> list[dict[str, Any]]:
    tests = load_tests()
    print("Loading Whisper small on CPU...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    rows: list[dict[str, Any]] = []

    for item in items:
        audio_path = Path(item["output_path"])
        test_id = item.get("test_id") or audio_path.stem
        if test_id not in tests and audio_path.stem.startswith("ar_clone_01"):
            test_id = "ar_clone_01_greeting"
        reference_text = item.get("text") or tests.get(test_id, {}).get("text", "")
        sidecar = sidecar_for(audio_path)
        stats = audio_stats(audio_path)
        print(f"Evaluating {audio_path.name}")
        transcript, language_probability = transcribe(model, audio_path)
        normalized_reference = normalize_arabic(reference_text)
        normalized_hypothesis = normalize_arabic(transcript)
        score = wer(normalized_reference, normalized_hypothesis) if normalized_reference else math.nan

        parameters = sidecar.get("parameters", {})
        generation_time = item.get("generation_time") or sidecar.get("generation_time", "")
        rtf = item.get("rtf") or sidecar.get("rtf", "")
        reference_path = item.get("reference_file") or sidecar.get("reference_path", "")
        reference_name = item.get("reference_name") or sidecar.get("reference_name", "")
        seed = item.get("seed") or sidecar.get("seed", "")

        row = {
            "test_id": test_id,
            "output_path": str(audio_path),
            "model": sidecar.get("model", MODEL_NAME),
            "device": DEVICE,
            "reference_name": reference_name,
            "reference_used": reference_path,
            "seed": seed,
            "temperature": item.get("temperature") or parameters.get("temperature", ""),
            "cfg_weight": item.get("cfg_weight") or parameters.get("cfg_weight", ""),
            "exaggeration": item.get("exaggeration") or parameters.get("exaggeration", ""),
            "repetition_penalty": item.get("repetition_penalty") or parameters.get("repetition_penalty", ""),
            "min_p": item.get("min_p") or parameters.get("min_p", ""),
            "top_p": item.get("top_p") or parameters.get("top_p", ""),
            "arabic_language_probability": language_probability,
            "asr_transcript": transcript,
            "reference_text": reference_text,
            "normalized_reference_text": normalized_reference,
            "normalized_hypothesis": normalized_hypothesis,
            "wer": score,
            "wer_percent": score * 100 if not math.isnan(score) else math.nan,
            "duration": stats["duration"],
            "generation_time": generation_time,
            "rtf": rtf,
            "sample_rate": stats["sample_rate"],
            "rms": stats["rms"],
            "peak": stats["peak"],
            "clipping_samples": stats["clipping_samples"],
            "silence_ratio": stats["silence_ratio"],
        }
        for field in HUMAN_REVIEW_FIELDS:
            row[field] = ""
        rows.append(row)
        print(f"WER {score * 100:.2f}% | Arabic probability {language_probability:.4f}")

    write_csv(output_csv, rows)
    if write_summary:
        write_final_summary(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_final_summary(rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if finite_float(row.get("wer")) is not None]
    wers = [float(row["wer"]) for row in successful]
    generation_times = [finite_float(row.get("generation_time")) for row in successful]
    durations = [finite_float(row.get("duration")) for row in successful]
    rtfs = [finite_float(row.get("rtf")) for row in successful]
    generation_times = [x for x in generation_times if x is not None]
    durations = [x for x in durations if x is not None]
    rtfs = [x for x in rtfs if x is not None]
    reference_used = rows[0].get("reference_used", "") if rows else ""

    summary = {
        "total_samples": len(rows),
        "successful_samples": len(successful),
        "failed_samples": len(rows) - len(successful),
        "average_wer": statistics.mean(wers) if wers else None,
        "median_wer": statistics.median(wers) if wers else None,
        "minimum_wer": min(wers) if wers else None,
        "maximum_wer": max(wers) if wers else None,
        "average_generation_time": statistics.mean(generation_times) if generation_times else None,
        "average_duration": statistics.mean(durations) if durations else None,
        "average_rtf": statistics.mean(rtfs) if rtfs else None,
        "samples_passing_arabic_wer_le_15_percent": sum(1 for row in successful if float(row["wer"]) <= 0.15),
        "samples_passing_clipping_eq_0": sum(1 for row in rows if int(row.get("clipping_samples", 1)) == 0),
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "reference_used": reference_used,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def final_items() -> list[dict[str, Any]]:
    tests = load_tests()
    return [
        {"test_id": test_id, "text": test["text"], "output_path": str(OUTPUT_DIR / f"{test_id}.wav")}
        for test_id, test in tests.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Arabic Chatterbox samples with Faster Whisper.")
    parser.add_argument("--mode", choices=["final", "parameter-tests", "reference-comparison"], default="final")
    args = parser.parse_args()

    if args.mode == "parameter-tests":
        items = read_generation_rows(RESULTS_DIR / "chatterbox_arabic_parameter_tests.csv")
        rows = evaluate_items(items, RESULTS_DIR / "chatterbox_arabic_parameter_evaluation.csv")
        best = min(rows, key=lambda row: float(row["wer"]))
        print("Automatic best candidate:")
        print(json.dumps(best, ensure_ascii=False, indent=2))
    elif args.mode == "reference-comparison":
        items = read_generation_rows(RESULTS_DIR / "chatterbox_arabic_reference_generation.csv")
        rows = evaluate_items(items, RESULTS_DIR / "chatterbox_arabic_reference_comparison.csv")
        best = min(rows, key=lambda row: float(row["wer"]))
        print("Lowest automatic WER reference candidate, not a final voice-quality decision:")
        print(json.dumps(best, ensure_ascii=False, indent=2))
    else:
        evaluate_items(final_items(), RESULTS_DIR / "chatterbox_arabic_evaluation.csv", write_summary=True)
        print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
