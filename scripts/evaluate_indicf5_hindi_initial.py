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
AUDIO_PATH = PROJECT_ROOT / "outputs" / "indicf5" / "hindi" / "hi_clone_01_greeting.wav"
CSV_PATH = PROJECT_ROOT / "results" / "indicf5_hindi_initial_test.csv"
DEVANAGARI_MARKS = re.compile(r"[\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963]")
PUNCTUATION = re.compile(r"[।،؛؟.!«»\"'ـ,:;?\-()\[\]{}]")

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = DEVANAGARI_MARKS.sub("", text)
    text = PUNCTUATION.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def audio_stats(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)
    return {
        "audio_duration": float(len(mono) / sample_rate) if sample_rate else 0.0,
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0,
        "peak": float(abs_audio.max()) if mono.size else 0.0,
        "clipping": bool(np.any(abs_audio >= 0.999)) if mono.size else False,
        "clipping_samples": int(np.sum(abs_audio >= 0.999)) if mono.size else 0,
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if mono.size else 1.0,
        "all_finite": bool(np.isfinite(mono).all()),
    }

def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

def main() -> None:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Missing IndicF5 greeting WAV: {AUDIO_PATH}")
    sidecar = json.loads(AUDIO_PATH.with_suffix(".json").read_text(encoding="utf-8"))
    expected = sidecar["expected_text"]
    stats = audio_stats(AUDIO_PATH)
    print("Loading Faster Whisper small on CPU...")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    print(f"Transcribing {AUDIO_PATH.name}...")
    segments, _ = whisper.transcribe(str(AUDIO_PATH), language="hi", beam_size=5, vad_filter=False)
    asr_text = " ".join(segment.text.strip() for segment in segments).strip()
    normalized_expected = normalize_text(expected)
    normalized_asr = normalize_text(asr_text)
    wer_value = wer(normalized_expected, normalized_asr) if normalized_expected else math.nan
    generation_time = float(sidecar.get("generation_time", 0.0))
    duration = stats["audio_duration"]
    row = {
        "test_id": sidecar.get("test_id", AUDIO_PATH.stem),
        "expected_text": expected,
        "ASR_text": asr_text,
        "normalized_expected": normalized_expected,
        "normalized_ASR": normalized_asr,
        "WER": wer_value,
        "WER_percent": wer_value * 100 if math.isfinite(wer_value) else math.nan,
        "generation_time": generation_time,
        "audio_duration": duration,
        "RTF": generation_time / duration if duration else math.inf,
        "peak": stats["peak"],
        "clipping": stats["clipping"],
        "clipping_samples": stats["clipping_samples"],
        "silence_ratio": stats["silence_ratio"],
        "rms": stats["rms"],
        "all_finite": stats["all_finite"],
        "model": "IndicF5",
        "device": "cpu",
        "reference_audio": sidecar.get("reference_audio", ""),
        "reference_transcript": sidecar.get("reference_transcript", ""),
        "status": "completed",
        "error": "",
    }
    write_csv(CSV_PATH, row)
    print(json.dumps(row, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
