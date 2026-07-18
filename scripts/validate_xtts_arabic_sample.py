from __future__ import annotations

import csv
import json
import math
import re
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
AUDIO_PATH = PROJECT_ROOT / "outputs" / "xtts" / "arabic" / "ar_clone_01_greeting.wav"
SIDECAR_PATH = AUDIO_PATH.with_suffix(".json")
CSV_PATH = PROJECT_ROOT / "results" / "xtts_arabic_initial_test.csv"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_PUNCTUATION = re.compile(r"[،؛؟.!«»\"'ـ,:;?\-()\[\]{}]")


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


def load_test() -> dict[str, Any]:
    data = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
    return data["tests"][0]


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


def write_csv(row: dict[str, Any]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(AUDIO_PATH)
    test = load_test()
    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8")) if SIDECAR_PATH.exists() else {}
    stats = audio_stats(AUDIO_PATH)

    print("Loading Whisper small on CPU...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(AUDIO_PATH), language="ar", beam_size=5, vad_filter=True)
    predicted = " ".join(segment.text.strip() for segment in segments).strip()
    expected = test["text"]
    normalized_expected = normalize_arabic(expected)
    normalized_predicted = normalize_arabic(predicted)
    score = wer(normalized_expected, normalized_predicted)

    row = {
        "test_id": test["id"],
        "expected_text": expected,
        "asr_text": predicted,
        "normalized_expected": normalized_expected,
        "normalized_predicted": normalized_predicted,
        "arabic_probability": float(info.language_probability),
        "wer": score,
        "wer_percentage": score * 100,
        "duration": stats["duration"],
        "generation_time": sidecar.get("generation_time", ""),
        "rtf": sidecar.get("rtf", ""),
        "rms": stats["rms"],
        "peak": stats["peak"],
        "clipping_count": stats["clipping_count"],
        "silence_ratio": stats["silence_ratio"],
        "reference_file": sidecar.get("reference_file", ""),
        "seed": sidecar.get("seed", ""),
        "model": sidecar.get("model", "tts_models/multilingual/multi-dataset/xtts_v2"),
        "device": sidecar.get("device", "cpu"),
        "output_path": str(AUDIO_PATH),
    }
    write_csv(row)

    print(f"Arabic probability: {float(info.language_probability):.4f}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Normalized expected: {normalized_expected}")
    print(f"Normalized predicted: {normalized_predicted}")
    print(f"WER: {score:.4f}")
    print(f"WER percentage: {score * 100:.2f}%")
    print(f"Pass target WER <= 15%: {score <= 0.15}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
