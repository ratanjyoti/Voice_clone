from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import wer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "english" / "neutts_improved" / "final"
RAW_PATH = PROJECT_ROOT / "results" / "raw" / "neutts_english_improved_evaluation.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary" / "neutts_english_improved_summary.csv"
COMPARISON_PATH = PROJECT_ROOT / "results" / "summary" / "english_model_comparison.csv"

PUNCT = re.compile(r"[^\w\s]")

DIGITS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

ORDINALS = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
    "11th": "eleventh",
    "12th": "twelfth",
    "13th": "thirteenth",
    "14th": "fourteenth",
    "15th": "fifteenth",
    "16th": "sixteenth",
    "17th": "seventeenth",
    "18th": "eighteenth",
    "19th": "nineteenth",
    "20th": "twentieth",
    "21st": "twenty first",
    "22nd": "twenty second",
    "23rd": "twenty third",
    "24th": "twenty fourth",
    "25th": "twenty fifth",
    "26th": "twenty sixth",
    "27th": "twenty seventh",
    "28th": "twenty eighth",
    "29th": "twenty ninth",
    "30th": "thirtieth",
    "31st": "thirty first",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    for token, replacement in ORDINALS.items():
        text = re.sub(rf"\b{token}\b", replacement, text)
    text = re.sub(
        r"\b\d+\b",
        lambda match: " ".join(DIGITS[digit] for digit in match.group(0)),
        text,
    )
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
        "peak": float(abs_audio.max()) if abs_audio.size else 0.0,
        "clipping": bool(np.any(abs_audio >= 0.999)),
        "clipping_samples": int(np.sum(abs_audio >= 0.999)),
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if abs_audio.size else 1.0,
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


def load_sidecars() -> list[dict]:
    sidecars = []
    for path in sorted(OUTPUT_DIR.glob("air_*.json")):
        if path.name.endswith("_failure.json"):
            continue
        sidecars.append(json.loads(path.read_text(encoding="utf-8")))
    if not sidecars:
        raise FileNotFoundError(f"No NeuTTS improved sidecars found in {OUTPUT_DIR}")
    return sidecars


def write_summary(rows: list[dict]) -> dict:
    samples = len(rows)
    failed = sum(1 for row in rows if row["status"] != "success")
    successful = [row for row in rows if row["status"] == "success"]
    denominator = len(successful) or 1
    summary = {
        "model": "NeuTTS-improved",
        "language": "English",
        "samples": samples,
        "successful_samples": len(successful),
        "failed_samples": failed,
        "average_WER_percent": round_value(
            sum(float(row["WER_percent"]) for row in rows) / samples,
            2,
        ),
        "average_RTF": round_value(
            sum(float(row["RTF"]) for row in successful) / denominator,
            3,
        ),
        "average_generation_time": round_value(
            sum(float(row["generation_time"]) for row in successful) / denominator,
            3,
        ),
        "average_audio_duration": round_value(
            sum(float(row["audio_duration"]) for row in successful) / denominator,
            3,
        ),
        "max_peak": round_value(max((float(row["peak"]) for row in rows), default=0.0), 3),
        "total_clipping_samples": int(sum(int(row["clipping_samples"]) for row in rows)),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def write_comparison(summary: dict) -> None:
    decision = (
        "NeuTTS improved is English winner pending final MOS confirmation"
        if summary["average_WER_percent"] <= 10
        and summary["total_clipping_samples"] == 0
        and summary["failed_samples"] == 0
        else "NeuTTS remains naturalness winner only, not final reproducible winner"
    )
    rows = [
        {
            "model": "MMS-TTS",
            "average_WER_percent": 8.0,
            "average_RTF": "<0.3",
            "average_generation_time": "",
            "average_audio_duration": "",
            "max_peak": "",
            "total_clipping_samples": 0,
            "manual_naturalness_rank": "robotic",
            "stability_status": "stable",
            "final_decision": "Automatic WER/RTF baseline, not naturalness winner",
        },
        {
            "model": "MeloTTS",
            "average_WER_percent": 9.67,
            "average_RTF": 1.428,
            "average_generation_time": "",
            "average_audio_duration": "",
            "max_peak": "",
            "total_clipping_samples": 0,
            "manual_naturalness_rank": "pending",
            "stability_status": "stable",
            "final_decision": "Current reproducible English winner unless NeuTTS improved passes",
        },
        {
            "model": "NeuTTS-original",
            "average_WER_percent": 12.67,
            "average_RTF": "14.708",
            "average_generation_time": "82.079",
            "average_audio_duration": "",
            "max_peak": "0.821",
            "total_clipping_samples": 0,
            "manual_naturalness_rank": "most human in manual listening",
            "stability_status": "final standardized run previously hung",
            "final_decision": "Naturalness winner only, not final reproducible winner",
        },
        {
            "model": "NeuTTS-improved",
            "average_WER_percent": summary["average_WER_percent"],
            "average_RTF": summary["average_RTF"],
            "average_generation_time": summary["average_generation_time"],
            "average_audio_duration": summary["average_audio_duration"],
            "max_peak": summary["max_peak"],
            "total_clipping_samples": summary["total_clipping_samples"],
            "manual_naturalness_rank": "pending",
            "stability_status": "stable" if summary["failed_samples"] == 0 else "failed samples present",
            "final_decision": decision,
        },
    ]
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sidecars = load_sidecars()
    needs_asr = any(item.get("status") == "success" for item in sidecars)
    model = None
    if needs_asr:
        print("Loading Faster Whisper small for NeuTTS improved English evaluation...")
        from faster_whisper import WhisperModel

        model = WhisperModel("small", device="cpu", compute_type="int8")

    rows = []
    for item in sidecars:
        output_value = item.get("output_path") or ""
        wav_path = Path(output_value) if output_value else OUTPUT_DIR / f"{item['test_id']}.wav"
        if not wav_path.exists():
            wav_path = OUTPUT_DIR / f"{item['test_id']}.wav"

        if item.get("status") != "success" or not wav_path.exists():
            row = {
                "test_id": item["test_id"],
                "model": "NeuTTS-improved",
                "language": "English",
                "expected_text": item["expected_text"],
                "tts_input_text": item["tts_input_text"],
                "ASR_text": "",
                "normalized_expected": normalize_text(item["expected_text"]),
                "normalized_asr": "",
                "WER": 1.0,
                "WER_percent": 100.0,
                "generation_time": item.get("generation_time", 0.0),
                "RTF": item.get("RTF", 0.0),
                "status": "failed",
                "error": item.get("error", "missing wav"),
                "output_path": str(wav_path),
                "sample_rate": item.get("sample_rate", 0),
                "channels": 0,
                "audio_duration": item.get("audio_duration", 0.0),
                "rms": item.get("rms", 0.0),
                "peak": item.get("peak", 0.0),
                "clipping": item.get("clipping", False),
                "clipping_samples": item.get("clipping_samples", 0),
                "silence_ratio": item.get("silence_ratio", 1.0),
            }
        else:
            stats = audio_stats(wav_path)
            asr_text = transcribe(model, wav_path)
            normalized_expected = normalize_text(item["expected_text"])
            normalized_asr = normalize_text(asr_text)
            wer_score = wer(normalized_expected, normalized_asr)
            row = {
                "test_id": item["test_id"],
                "model": "NeuTTS-improved",
                "language": "English",
                "expected_text": item["expected_text"],
                "tts_input_text": item["tts_input_text"],
                "ASR_text": asr_text,
                "normalized_expected": normalized_expected,
                "normalized_asr": normalized_asr,
                "WER": round_value(wer_score, 6),
                "WER_percent": round_value(wer_score * 100, 2),
                "generation_time": round_value(item.get("generation_time", 0.0), 6),
                "RTF": round_value(item.get("RTF", 0.0), 6),
                "status": "success",
                "error": "",
                "output_path": str(wav_path),
                **stats,
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = write_summary(rows)
    write_comparison(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


