from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHATTERBOX_CSV = PROJECT_ROOT / "results" / "chatterbox_arabic_evaluation.csv"
XTTS_CSV = PROJECT_ROOT / "results" / "xtts_arabic_evaluation.csv"
OUT_CSV = PROJECT_ROOT / "results" / "arabic_model_comparison.csv"
OUT_JSON = PROJECT_ROOT / "evidence" / "result_snapshots" / "arabic_model_comparison_summary.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def summarize(name: str, rows: list[dict[str, str]], reference_strategy: str, license_note: str, model_size: str, streaming: str) -> dict[str, object]:
    successful = [row for row in rows if (number(row, "wer", "wer_percentage", "wer_percent") is not None)]
    wers = []
    for row in successful:
        if row.get("wer") not in (None, ""):
            wers.append(float(row["wer"]))
        else:
            wers.append(float(row.get("wer_percentage") or row.get("wer_percent")) / 100)
    gen = [number(row, "generation_time") for row in successful]
    rtfs = [number(row, "rtf") for row in successful]
    clips = [number(row, "clipping_count", "clipping_samples") for row in rows]
    gen = [x for x in gen if x is not None]
    rtfs = [x for x in rtfs if x is not None]
    clips = [x for x in clips if x is not None]
    pass_count = sum(1 for w in wers if w <= 0.15)
    clip_pass = sum(1 for c in clips if c == 0)
    return {
        "model": name,
        "successful_samples": len(successful),
        "failed_samples": len(rows) - len(successful),
        "average_wer": statistics.mean(wers) if wers else None,
        "median_wer": statistics.median(wers) if wers else None,
        "maximum_wer": max(wers) if wers else None,
        "wer_pass_count": pass_count,
        "average_generation_time": statistics.mean(gen) if gen else None,
        "average_rtf": statistics.mean(rtfs) if rtfs else None,
        "clipping_pass_count": clip_pass,
        "model_size_if_known": model_size,
        "cpu_feasibility": "CPU evaluation completed; production real-time suitability depends on RTF and hardware",
        "streaming_support_if_verified": streaming,
        "voice_cloning_support": "verified zero-shot reference audio cloning",
        "reference_strategy": reference_strategy,
        "license_note": license_note,
        "meets_average_wer_target": bool(wers and statistics.mean(wers) <= 0.15),
        "meets_4_of_5_pass_target": pass_count >= 4,
        "meets_no_clipping_target": clip_pass == len(rows),
        "meets_no_failed_target": len(successful) == len(rows),
    }


def winner(chatterbox: dict[str, object], xtts: dict[str, object]) -> str:
    candidates = []
    for row in [chatterbox, xtts]:
        if (
            row["meets_average_wer_target"]
            and row["meets_4_of_5_pass_target"]
            and row["meets_no_clipping_target"]
            and row["meets_no_failed_target"]
        ):
            candidates.append(row)
    if not candidates:
        return "No automatic production winner"
    candidates.sort(key=lambda r: (float(r["average_wer"]), float(r["average_rtf"])))
    return str(candidates[0]["model"])


def main() -> None:
    chatterbox_rows = read_rows(CHATTERBOX_CSV)
    xtts_rows = read_rows(XTTS_CSV)
    chatterbox = summarize(
        "Chatterbox multilingual",
        chatterbox_rows,
        "standard reference",
        "Existing case-study model; deployment licensing still requires project review.",
        "not recorded in this run",
        "not verified",
    )
    xtts = summarize(
        "XTTS-v2",
        xtts_rows,
        "multi reference: short + standard + long",
        "CPML; commercial/deployment use requires additional license review.",
        "model.pth 1.87 GB plus config/vocab/speakers",
        "not verified",
    )
    rows = [chatterbox, xtts]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": rows,
        "automatic_winner": winner(chatterbox, xtts),
        "winner_rules": [
            "meets Arabic WER target",
            "at least 4/5 samples pass",
            "no clipping",
            "no failed samples",
            "lower average WER",
            "lower average RTF",
        ],
        "human_quality_fields_compared": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
