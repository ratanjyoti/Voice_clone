import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_RESULTS_FILE = PROJECT_ROOT / "results" / "raw_runs.csv"
SUMMARY_FILE = PROJECT_ROOT / "results" / "mms_summary.csv"


SUMMARY_FIELDS = [
    "language",
    "language_code",
    "model_id",
    "device",
    "successful_runs",
    "failed_runs",
    "mean_generation_time_seconds",
    "median_generation_time_seconds",
    "min_generation_time_seconds",
    "max_generation_time_seconds",
    "mean_audio_duration_seconds",
    "mean_rtf",
    "median_rtf",
    "min_rtf",
    "max_rtf",
    "mean_model_load_time_seconds",
    "latency_target_pass_rate_percent",
    "rtf_target_pass_rate_percent",
]


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def rounded(value: float | None, digits: int = 4):
    if value is None:
        return ""
    return round(value, digits)


def load_rows() -> list[dict]:
    if not RAW_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Raw results file not found: {RAW_RESULTS_FILE}"
        )

    with RAW_RESULTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def group_rows(rows: list[dict]) -> dict:
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row.get("language", ""),
            row.get("language_code", ""),
            row.get("model_id", ""),
            row.get("device", ""),
        )
        grouped[key].append(row)

    return grouped


def summarize_group(key: tuple, rows: list[dict]) -> dict:
    language, language_code, model_id, device = key

    successful_rows = [
        row for row in rows
        if parse_bool(row.get("success", ""))
    ]

    failed_rows = [
        row for row in rows
        if not parse_bool(row.get("success", ""))
    ]

    generation_times = [
        value
        for row in successful_rows
        if (
            value := parse_float(
                row.get("generation_time_seconds", "")
            )
        ) is not None
    ]

    audio_durations = [
        value
        for row in successful_rows
        if (
            value := parse_float(
                row.get("audio_duration_seconds", "")
            )
        ) is not None
    ]

    rtfs = [
        value
        for row in successful_rows
        if (
            value := parse_float(row.get("rtf", ""))
        ) is not None
    ]

    load_times = [
        value
        for row in successful_rows
        if (
            value := parse_float(
                row.get("model_load_time_seconds", "")
            )
        ) is not None
    ]

    latency_pass_count = sum(
        1 for value in generation_times
        if value < 2.0
    )

    rtf_pass_count = sum(
        1 for value in rtfs
        if value <= 0.5
    )

    latency_pass_rate = (
        latency_pass_count / len(generation_times) * 100
        if generation_times
        else None
    )

    rtf_pass_rate = (
        rtf_pass_count / len(rtfs) * 100
        if rtfs
        else None
    )

    return {
        "language": language,
        "language_code": language_code,
        "model_id": model_id,
        "device": device,
        "successful_runs": len(successful_rows),
        "failed_runs": len(failed_rows),
        "mean_generation_time_seconds": rounded(
            mean_or_none(generation_times)
        ),
        "median_generation_time_seconds": rounded(
            median_or_none(generation_times)
        ),
        "min_generation_time_seconds": rounded(
            min(generation_times) if generation_times else None
        ),
        "max_generation_time_seconds": rounded(
            max(generation_times) if generation_times else None
        ),
        "mean_audio_duration_seconds": rounded(
            mean_or_none(audio_durations)
        ),
        "mean_rtf": rounded(mean_or_none(rtfs)),
        "median_rtf": rounded(median_or_none(rtfs)),
        "min_rtf": rounded(
            min(rtfs) if rtfs else None
        ),
        "max_rtf": rounded(
            max(rtfs) if rtfs else None
        ),
        "mean_model_load_time_seconds": rounded(
            mean_or_none(load_times)
        ),
        "latency_target_pass_rate_percent": rounded(
            latency_pass_rate,
            2,
        ),
        "rtf_target_pass_rate_percent": rounded(
            rtf_pass_rate,
            2,
        ),
    }


def print_summary(summary_rows: list[dict]) -> None:
    print("=" * 110)
    print("MMS-TTS benchmark summary")
    print("=" * 110)

    header = (
        f"{'Language':<10}"
        f"{'Runs':>8}"
        f"{'Mean latency':>16}"
        f"{'Median latency':>18}"
        f"{'Mean RTF':>12}"
        f"{'Latency pass':>16}"
        f"{'RTF pass':>12}"
    )

    print(header)
    print("-" * 110)

    for row in summary_rows:
        runs = (
            f"{row['successful_runs']}/"
            f"{row['successful_runs'] + row['failed_runs']}"
        )

        print(
            f"{row['language']:<10}"
            f"{runs:>8}"
            f"{row['mean_generation_time_seconds']:>16}"
            f"{row['median_generation_time_seconds']:>18}"
            f"{row['mean_rtf']:>12}"
            f"{str(row['latency_target_pass_rate_percent']) + '%':>16}"
            f"{str(row['rtf_target_pass_rate_percent']) + '%':>12}"
        )

    print("=" * 110)


def main() -> None:
    rows = load_rows()

    if not rows:
        raise ValueError("The raw benchmark CSV contains no rows.")

    grouped_rows = group_rows(rows)

    summary_rows = [
        summarize_group(key, grouped_rows[key])
        for key in sorted(grouped_rows)
    ]

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print_summary(summary_rows)
    print(f"\nSummary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()