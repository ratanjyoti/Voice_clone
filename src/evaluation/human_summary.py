from __future__ import annotations

import csv
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "results" / "human_listener_scores.csv"
OUTPUT = PROJECT_ROOT / "results" / "human_evaluation_summary.csv"


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    rows: list[dict[str, str]] = []
    if INPUT.exists():
        with INPUT.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        language = row.get("language", "").strip() or "unknown"
        sample = row.get("anonymous_sample_id", "").strip() or "unknown"
        groups.setdefault((language, sample), []).append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "language",
        "anonymous_sample_id",
        "average_naturalness_MOS",
        "average_similarity_judgment",
        "average_pronunciation",
        "acceptance_rate",
        "reviewer_count",
        "sample_count",
        "status",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        if not groups:
            writer.writerow(
                {
                    "language": "all",
                    "anonymous_sample_id": "",
                    "average_naturalness_MOS": "",
                    "average_similarity_judgment": "",
                    "average_pronunciation": "",
                    "acceptance_rate": "",
                    "reviewer_count": 0,
                    "sample_count": 0,
                    "status": "pending_real_listener_scores",
                }
            )
            return

        for (language, sample), group_rows in sorted(groups.items()):
            naturalness = [
                value
                for value in (
                    to_float(row.get("naturalness_score_1_to_5", ""))
                    for row in group_rows
                )
                if value is not None
            ]
            similarity = [
                value
                for value in (
                    to_float(row.get("speaker_similarity_score_1_to_5", ""))
                    for row in group_rows
                )
                if value is not None
            ]
            pronunciation = [
                value
                for value in (
                    to_float(row.get("pronunciation_score_1_to_5", ""))
                    for row in group_rows
                )
                if value is not None
            ]
            accepted = [
                row.get("accepted", "").strip().lower() in {"yes", "true", "1"}
                for row in group_rows
                if row.get("accepted", "").strip()
            ]
            reviewers = {
                row.get("reviewer_id", "").strip()
                for row in group_rows
                if row.get("reviewer_id", "").strip()
            }
            writer.writerow(
                {
                    "language": language,
                    "anonymous_sample_id": sample,
                    "average_naturalness_MOS": statistics.mean(naturalness)
                    if naturalness
                    else "",
                    "average_similarity_judgment": statistics.mean(similarity)
                    if similarity
                    else "",
                    "average_pronunciation": statistics.mean(pronunciation)
                    if pronunciation
                    else "",
                    "acceptance_rate": statistics.mean(accepted) if accepted else "",
                    "reviewer_count": len(reviewers),
                    "sample_count": len(group_rows),
                    "status": "complete",
                }
            )


if __name__ == "__main__":
    main()
