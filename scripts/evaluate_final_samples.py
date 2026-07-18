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
RESULTS_DIR = PROJECT_ROOT / "results"
SNAPSHOT_DIR = PROJECT_ROOT / "evidence" / "result_snapshots"

CONFIGS = {
    "mms_english": {
        "language": "english",
        "language_code": "en",
        "model": "MMS-TTS",
        "output_dir": PROJECT_ROOT / "outputs" / "mms" / "english" / "final",
        "csv": RESULTS_DIR / "mms_english_evaluation.csv",
    },
    "neutts_english": {
        "language": "english",
        "language_code": "en",
        "model": "NeuTTS",
        "output_dir": PROJECT_ROOT / "outputs" / "neutts" / "english" / "final",
        "csv": RESULTS_DIR / "neutts_english_evaluation.csv",
    },
    "mms_hindi": {
        "language": "hindi",
        "language_code": "hi",
        "model": "MMS-TTS",
        "output_dir": PROJECT_ROOT / "outputs" / "mms" / "hindi" / "final",
        "csv": RESULTS_DIR / "mms_hindi_evaluation.csv",
    },
    "chatterbox_hindi": {
        "language": "hindi",
        "language_code": "hi",
        "model": "Chatterbox",
        "output_dir": PROJECT_ROOT / "outputs" / "chatterbox" / "hindi" / "final",
        "csv": RESULTS_DIR / "chatterbox_hindi_evaluation.csv",
    },
}

DEVANAGARI_MARKS = re.compile(r"[\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963]")
PUNCTUATION = re.compile(r"[।،؛؟.!«»\"'ـ,:;?\-()\[\]{}]")


def normalize_text(text: str, language_code: str) -> str:
    text = text.strip().lower()
    if language_code == "hi":
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
        "peak": float(abs_audio.max()) if mono.size else 0.0,
        "clipping": bool(np.any(abs_audio >= 0.999)) if mono.size else False,
        "clipping_samples": int(np.sum(abs_audio >= 0.999)) if mono.size else 0,
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if mono.size else 1.0,
    }


def read_sidecar(audio_path: Path) -> dict[str, Any]:
    sidecar_path = audio_path.with_suffix(".json")
    if sidecar_path.exists():
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    return {}


def transcribe(model: WhisperModel, audio_path: Path, language_code: str) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language=language_code,
        beam_size=5,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_config(key: str, whisper: WhisperModel) -> list[dict[str, Any]]:
    cfg = CONFIGS[key]
    rows: list[dict[str, Any]] = []
    for audio_path in sorted(cfg["output_dir"].glob("*.wav")):
        sidecar = read_sidecar(audio_path)
        expected = sidecar.get("expected_text", "")
        stats = audio_stats(audio_path)
        print(f"Evaluating {key}: {audio_path.name}")
        asr_text = transcribe(whisper, audio_path, cfg["language_code"])
        normalized_expected = normalize_text(expected, cfg["language_code"])
        normalized_asr = normalize_text(asr_text, cfg["language_code"])
        wer_value = wer(normalized_expected, normalized_asr) if normalized_expected else math.nan
        generation_time = sidecar.get("generation_time", "")
        duration = stats["audio_duration"]
        rtf = sidecar.get("rtf")
        if (rtf is None or rtf == "") and generation_time != "" and duration:
            rtf = float(generation_time) / duration

        rows.append(
            {
                "test_id": sidecar.get("test_id", audio_path.stem),
                "expected_text": expected,
                "ASR_text": asr_text,
                "WER": wer_value,
                "WER_percent": wer_value * 100 if not math.isnan(wer_value) else math.nan,
                "generation_time": generation_time,
                "audio_duration": duration,
                "RTF": rtf,
                "peak": stats["peak"],
                "clipping": stats["clipping"],
                "clipping_samples": stats["clipping_samples"],
                "silence_ratio": stats["silence_ratio"],
                "language": cfg["language"],
                "model": cfg["model"],
                "output_path": str(audio_path),
            }
        )
    write_csv(cfg["csv"], rows)
    return rows


def summarize(rows: list[dict[str, Any]], language: str, path: Path) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    payload: dict[str, Any] = {
        "language": language,
        "wer_target": 0.10,
        "models": {},
    }
    for model, model_rows in by_model.items():
        wers = [float(row["WER"]) for row in model_rows if math.isfinite(float(row["WER"]))]
        rtfs = [float(row["RTF"]) for row in model_rows if row.get("RTF") not in ("", None)]
        payload["models"][model] = {
            "total_samples": len(model_rows),
            "successful_samples": len(wers),
            "average_WER": statistics.mean(wers) if wers else None,
            "average_WER_percent": statistics.mean(wers) * 100 if wers else None,
            "WER_pass": statistics.mean(wers) <= 0.10 if wers else False,
            "average_RTF": statistics.mean(rtfs) if rtfs else None,
            "clipping_pass_count": sum(not bool(row["clipping"]) for row in model_rows),
            "failed_generations": sum(not Path(str(row["output_path"])).exists() for row in model_rows),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final English/Hindi samples with Faster Whisper.")
    parser.add_argument(
        "--target",
        choices=["english", "hindi", "all"],
        default="all",
    )
    args = parser.parse_args()

    print("Loading Faster Whisper small on CPU...")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    all_rows: list[dict[str, Any]] = []

    if args.target in ("english", "all"):
        rows = evaluate_config("mms_english", whisper) + evaluate_config("neutts_english", whisper)
        summarize(rows, "English", SNAPSHOT_DIR / "english_model_summary.json")
        all_rows.extend(rows)

    if args.target in ("hindi", "all"):
        rows = evaluate_config("mms_hindi", whisper) + evaluate_config("chatterbox_hindi", whisper)
        summarize(rows, "Hindi", SNAPSHOT_DIR / "hindi_model_summary.json")
        all_rows.extend(rows)

    if all_rows:
        write_csv(RESULTS_DIR / "english_hindi_final_evaluation_all.csv", all_rows)

    print("Final sample ASR/WER evaluation complete.")


if __name__ == "__main__":
    main()
