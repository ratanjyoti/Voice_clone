from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xtts" / "arabic"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = PROJECT_ROOT / "evidence" / "result_snapshots" / "xtts" / "arabic"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
HUMAN_FIELDS = [
    "similarity_score_1_to_5",
    "naturalness_score_1_to_5",
    "pronunciation_score_1_to_5",
    "metallic",
    "missing_words",
    "repeated_words",
    "accepted",
    "reviewer_notes",
]
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_PUNCTUATION = re.compile(r"[،؛؟.!«»\"'ـ,:;?\-()\[\]{}]")


def normalize_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text)
    replacements = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي", "ة": "ه"}
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = ARABIC_PUNCTUATION.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_tests() -> dict[str, dict[str, Any]]:
    return {test["id"]: test for test in json.loads(TESTS_PATH.read_text(encoding="utf-8"))["tests"]}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sidecar(path: Path) -> dict[str, Any]:
    candidate = path.with_suffix(".json")
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def audio_stats(path: Path) -> dict[str, Any]:
    audio, sr = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)
    return {
        "duration": float(len(mono) / sr) if sr else 0.0,
        "sample_rate": int(sr),
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0,
        "peak": float(abs_audio.max()) if mono.size else 0.0,
        "clipping_count": int(np.sum(abs_audio >= 0.999)) if mono.size else 0,
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if mono.size else 1.0,
    }


def transcribe(model: WhisperModel, path: Path) -> tuple[str, float]:
    segments, info = model.transcribe(str(path), language="ar", beam_size=5, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip(), float(info.language_probability)


def infer_test_id(path: Path, row: dict[str, Any]) -> str:
    if row.get("test_id"):
        return str(row["test_id"])
    if "greeting" in path.stem:
        return "ar_clone_01_greeting"
    return path.stem


def parse_params(row: dict[str, Any], sc: dict[str, Any]) -> str:
    keys = ["temperature", "top_k", "top_p", "repetition_penalty", "length_penalty", "speed"]
    params = {key: row.get(key, sc.get(key, "")) for key in keys if row.get(key, sc.get(key, "")) != ""}
    return json.dumps(params, ensure_ascii=False)


def evaluate(rows_in: list[dict[str, Any]], output_csv: Path, add_human: bool = False, write_summary: bool = False) -> list[dict[str, Any]]:
    tests = load_tests()
    print("Loading Whisper small on CPU...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    rows: list[dict[str, Any]] = []
    for item in rows_in:
        output_path = Path(item["output_path"])
        test_id = infer_test_id(output_path, item)
        expected = item.get("text") or tests.get(test_id, {}).get("text", "")
        category = item.get("category") or tests.get(test_id, {}).get("category", "")
        sc = sidecar(output_path)
        base = {
            "model": item.get("model") or sc.get("model", MODEL_NAME),
            "model_version": item.get("model_version") or sc.get("model_version", ""),
            "test_id": test_id,
            "category": category,
            "expected_text": expected,
            "output_path": str(output_path),
            "reference_strategy": item.get("reference_strategy") or sc.get("reference_strategy", ""),
            "reference_paths": item.get("reference_paths") or sc.get("reference_paths", ""),
            "parameters": parse_params(item, sc),
            "seed": item.get("seed") or sc.get("seed", ""),
            "generation_time": item.get("generation_time") or sc.get("generation_time", ""),
            "rtf": item.get("rtf") or sc.get("rtf", ""),
            "status": item.get("status") or sc.get("status", ""),
            "error": item.get("error") or sc.get("error", ""),
            "configuration_id": item.get("configuration_id") or sc.get("configuration_id", ""),
        }
        try:
            if base["status"] != "success":
                raise RuntimeError(base["error"] or "generation did not report success")
            stats = audio_stats(output_path)
            print(f"Evaluating {output_path.name}")
            asr_text, prob = transcribe(model, output_path)
            normalized_expected = normalize_arabic(expected)
            normalized_predicted = normalize_arabic(asr_text)
            score = wer(normalized_expected, normalized_predicted) if normalized_expected else math.nan
            base.update(
                {
                    "asr_text": asr_text,
                    "normalized_expected": normalized_expected,
                    "normalized_predicted": normalized_predicted,
                    "arabic_probability": prob,
                    "wer": score,
                    "wer_percentage": score * 100 if math.isfinite(score) else math.nan,
                    "duration": stats["duration"],
                    "rms": stats["rms"],
                    "peak": stats["peak"],
                    "clipping_count": stats["clipping_count"],
                    "silence_ratio": stats["silence_ratio"],
                    "sample_rate": stats["sample_rate"],
                    "status": "success",
                }
            )
            print(f"WER {base['wer_percentage']:.2f}% | Arabic probability {prob:.4f}")
        except Exception as exc:
            base.update(
                {
                    "asr_text": "",
                    "normalized_expected": normalize_arabic(expected),
                    "normalized_predicted": "",
                    "arabic_probability": "",
                    "wer": math.nan,
                    "wer_percentage": math.nan,
                    "duration": "",
                    "rms": "",
                    "peak": "",
                    "clipping_count": "",
                    "silence_ratio": "",
                    "sample_rate": "",
                    "status": "failed",
                    "error": base.get("error") or f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"FAILED evaluation for {output_path}: {base['error']}")
        if add_human:
            for field in HUMAN_FIELDS:
                base[field] = ""
        rows.append(base)
    write_csv(output_csv, rows)
    if write_summary:
        write_final_summary(rows)
    return rows


def finite(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def write_final_summary(rows: list[dict[str, Any]]) -> None:
    success = [row for row in rows if row.get("status") == "success" and finite(row.get("wer")) is not None]
    wers = [float(row["wer"]) for row in success]
    generation = [finite(row.get("generation_time")) for row in success]
    durations = [finite(row.get("duration")) for row in success]
    rtfs = [finite(row.get("rtf")) for row in success]
    generation = [x for x in generation if x is not None]
    durations = [x for x in durations if x is not None]
    rtfs = [x for x in rtfs if x is not None]
    first = rows[0] if rows else {}
    torch_version = ""
    version_log = PROJECT_ROOT / "evidence" / "terminal_logs" / "xtts_coqui_import_verification.txt"
    if version_log.exists():
        for line in version_log.read_text(encoding="utf-8", errors="replace").replace("\x00", "").replace("\ufeff", "").replace("�", "").splitlines():
            line = line.strip()
            if line.startswith("torch:"):
                torch_version = line.split(":", 1)[1].strip()
                break
    if not torch_version:
        try:
            import torch
            torch_version = torch.__version__
        except Exception:
            torch_version = ""
    package_version = first.get("model_version", "")
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(rows),
        "successful_samples": len(success),
        "failed_samples": len(rows) - len(success),
        "average_wer": statistics.mean(wers) if wers else None,
        "median_wer": statistics.median(wers) if wers else None,
        "minimum_wer": min(wers) if wers else None,
        "maximum_wer": max(wers) if wers else None,
        "average_generation_time": statistics.mean(generation) if generation else None,
        "average_duration": statistics.mean(durations) if durations else None,
        "average_rtf": statistics.mean(rtfs) if rtfs else None,
        "number_passing_wer_le_15_percent": sum(1 for row in success if float(row["wer"]) <= 0.15),
        "number_failing_wer": sum(1 for row in success if float(row["wer"]) > 0.15),
        "number_with_zero_clipping": sum(1 for row in rows if str(row.get("clipping_count")) not in ("", "nan") and int(float(row.get("clipping_count"))) == 0),
        "model_name": MODEL_NAME,
        "package_version": package_version,
        "torch_version": torch_version,
        "device": "cpu",
        "reference_strategy": first.get("reference_strategy", ""),
        "reference_paths": first.get("reference_paths", ""),
        "selected_parameters": first.get("parameters", ""),
        "seed": first.get("seed", ""),
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "xtts_arabic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def automatic_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        w = finite(row.get("wer"))
        clip = finite(row.get("clipping_count"))
        rtf = finite(row.get("rtf"))
        if row.get("status") == "success" and w is not None and clip == 0:
            candidates.append((w, rtf if rtf is not None else float("inf"), row))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]


def final_items() -> list[dict[str, Any]]:
    tests = load_tests()
    rows = []
    for test in tests.values():
        path = OUTPUT_DIR / f"{test['id']}.wav"
        sc = sidecar(path)
        rows.append({**sc, "output_path": str(path), "test_id": test["id"], "text": test["text"], "category": test["category"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Arabic XTTS-v2 samples with Faster Whisper.")
    parser.add_argument("--mode", choices=["reference-comparison", "parameter-tests", "final"], default="final")
    args = parser.parse_args()
    if args.mode == "reference-comparison":
        rows = evaluate(read_csv(RESULTS_DIR / "xtts_arabic_reference_generation.csv"), RESULTS_DIR / "xtts_arabic_reference_comparison.csv")
        winner = automatic_winner(rows)
        print("automatic reference winner:")
        print(json.dumps(winner, ensure_ascii=False, indent=2) if winner else "None")
    elif args.mode == "parameter-tests":
        rows = evaluate(read_csv(RESULTS_DIR / "xtts_arabic_parameter_tests.csv"), RESULTS_DIR / "xtts_arabic_parameter_evaluation.csv")
        winner = automatic_winner(rows)
        print("automatic parameter winner:")
        print(json.dumps(winner, ensure_ascii=False, indent=2) if winner else "None")
    else:
        evaluate(final_items(), RESULTS_DIR / "xtts_arabic_evaluation.csv", add_human=True, write_summary=True)
        print(f"summary: {SUMMARY_DIR / 'xtts_arabic_summary.json'}")


if __name__ == "__main__":
    main()






