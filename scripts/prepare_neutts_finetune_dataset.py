from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "finetune" / "neutts_english"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_PATH = DATA_DIR / "metadata_template.csv"
MANIFEST_PATH = DATA_DIR / "manifest_clean.csv"
VALIDATION_CSV = PROJECT_ROOT / "results" / "diagnostics" / "neutts_finetune_dataset_validation.csv"
SUMMARY_JSON = PROJECT_ROOT / "evidence" / "result_snapshots" / "neutts_finetune_dataset_summary.json"

MIN_PROTOTYPE_SECONDS = 30 * 60
RECOMMENDED_SECONDS = 2 * 60 * 60
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 12.0
TARGET_SAMPLE_RATE = 24000


def audio_stats(path: Path) -> dict:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_seconds": duration,
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0,
        "peak": float(abs_audio.max()) if abs_audio.size else 0.0,
        "clipping": bool(np.any(abs_audio >= 0.999)),
        "clipping_samples": int(np.sum(abs_audio >= 0.999)),
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if abs_audio.size else 1.0,
    }


def read_metadata() -> dict[str, dict]:
    if not METADATA_PATH.exists():
        return {}
    with METADATA_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["file_name"]: row for row in csv.DictReader(handle)}


def validate_row(path: Path, metadata: dict[str, dict]) -> dict:
    row = metadata.get(path.name, {})
    transcript = (row.get("transcript") or "").strip()
    speaker_id = (row.get("speaker_id") or "").strip()
    split = (row.get("split") or "train").strip()
    issues = []

    try:
        stats = audio_stats(path)
    except Exception as exc:
        stats = {
            "sample_rate": 0,
            "channels": 0,
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "clipping": False,
            "clipping_samples": 0,
            "silence_ratio": 1.0,
        }
        issues.append(f"audio_read_error: {exc}")

    if not transcript:
        issues.append("missing_transcript")
    if not speaker_id:
        issues.append("missing_speaker_id")
    if stats["sample_rate"] != TARGET_SAMPLE_RATE:
        issues.append("sample_rate_must_be_24000")
    if stats["channels"] != 1:
        issues.append("audio_must_be_mono")
    if stats["duration_seconds"] < MIN_CLIP_SECONDS:
        issues.append("clip_too_short")
    if stats["duration_seconds"] > MAX_CLIP_SECONDS:
        issues.append("clip_too_long")
    if stats["clipping_samples"] > 0:
        issues.append("clipping_detected")
    if stats["silence_ratio"] > 0.6:
        issues.append("too_much_silence")

    return {
        "file_name": path.name,
        "path": str(path),
        "transcript": transcript,
        "speaker_id": speaker_id,
        "split": split,
        **stats,
        "status": "accepted" if not issues else "rejected",
        "issues": "; ".join(issues),
    }


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata()
    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    rows = [validate_row(path, metadata) for path in wav_paths]

    referenced = set(metadata)
    present = {path.name for path in wav_paths}
    for missing in sorted(referenced - present):
        meta = metadata[missing]
        rows.append(
            {
                "file_name": missing,
                "path": str(AUDIO_DIR / missing),
                "transcript": (meta.get("transcript") or "").strip(),
                "speaker_id": (meta.get("speaker_id") or "").strip(),
                "split": (meta.get("split") or "train").strip(),
                "sample_rate": 0,
                "channels": 0,
                "duration_seconds": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "clipping": False,
                "clipping_samples": 0,
                "silence_ratio": 1.0,
                "status": "rejected",
                "issues": "missing_audio_file",
            }
        )

    accepted = [row for row in rows if row["status"] == "accepted"]
    rejected = [row for row in rows if row["status"] != "accepted"]
    total_seconds = sum(float(row["duration_seconds"]) for row in accepted)
    speakers = sorted({row["speaker_id"] for row in accepted if row["speaker_id"]})

    fieldnames = [
        "file_name",
        "path",
        "transcript",
        "speaker_id",
        "split",
        "sample_rate",
        "channels",
        "duration_seconds",
        "rms",
        "peak",
        "clipping",
        "clipping_samples",
        "silence_ratio",
        "status",
        "issues",
    ]
    with VALIDATION_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "path", "transcript", "speaker_id", "split", "duration_seconds"])
        writer.writeheader()
        for row in accepted:
            writer.writerow({key: row[key] for key in ["file_name", "path", "transcript", "speaker_id", "split", "duration_seconds"]})

    summary = {
        "audio_dir": str(AUDIO_DIR),
        "metadata_path": str(METADATA_PATH),
        "total_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "total_accepted_seconds": round(total_seconds, 3),
        "total_accepted_minutes": round(total_seconds / 60, 3),
        "speakers": speakers,
        "prototype_ready": total_seconds >= MIN_PROTOTYPE_SECONDS and len(speakers) == 1 and not rejected,
        "recommended_real_improvement_ready": total_seconds >= RECOMMENDED_SECONDS and len(speakers) == 1 and not rejected,
        "training_started": False,
        "training_status": "not_started_scaffold_only",
        "blockers": [] if accepted else ["No accepted WAV clips found."],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
