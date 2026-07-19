import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "results" / "wer_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "wer_summary.csv"

OUTPUT_FIELDS = [
    "language",
    "language_code",
    "model_id",
    "asr_model",
    "successful_samples",
    "failed_samples",
    "mean_wer",
    "median_wer",
    "min_wer",
    "max_wer",
    "mean_cer",
    "median_cer",
    "wer_target_pass_rate_percent",
]


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def rounded(value: float | None):
    if value is None:
        return ""

    return round(value, 4)


def load_rows() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"WER results not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main() -> None:
    rows = load_rows()

    grouped = defaultdict(list)

    for row in rows:
        key = (
            row.get("language", ""),
            row.get("language_code", ""),
            row.get("model_id", ""),
            row.get("asr_model", ""),
        )

        grouped[key].append(row)

    summary_rows = []

    for key, group_rows in sorted(grouped.items()):
        language, language_code, model_id, asr_model = key

        successful = [
            row
            for row in group_rows
            if parse_bool(row.get("success", ""))
        ]

        failed = [
            row
            for row in group_rows
            if not parse_bool(row.get("success", ""))
        ]

        wer_values = [
            value
            for row in successful
            if (
                value := parse_float(
                    row.get("wer", "")
                )
            ) is not None
        ]

        cer_values = [
            value
            for row in successful
            if (
                value := parse_float(
                    row.get("cer", "")
                )
            ) is not None
        ]

        pass_count = sum(
            value <= 0.10
            for value in wer_values
        )

        pass_rate = (
            pass_count / len(wer_values) * 100
            if wer_values
            else None
        )

        summary_rows.append(
            {
                "language": language,
                "language_code": language_code,
                "model_id": model_id,
                "asr_model": asr_model,
                "successful_samples": len(successful),
                "failed_samples": len(failed),
                "mean_wer": rounded(
                    statistics.mean(wer_values)
                    if wer_values
                    else None
                ),
                "median_wer": rounded(
                    statistics.median(wer_values)
                    if wer_values
                    else None
                ),
                "min_wer": rounded(
                    min(wer_values)
                    if wer_values
                    else None
                ),
                "max_wer": rounded(
                    max(wer_values)
                    if wer_values
                    else None
                ),
                "mean_cer": rounded(
                    statistics.mean(cer_values)
                    if cer_values
                    else None
                ),
                "median_cer": rounded(
                    statistics.median(cer_values)
                    if cer_values
                    else None
                ),
                "wer_target_pass_rate_percent": (
                    round(pass_rate, 2)
                    if pass_rate is not None
                    else ""
                ),
            }
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 90)
    print("WER/CER summary")
    print("=" * 90)

    for row in summary_rows:
        print(
            f"{row['language']:<10} "
            f"Mean WER: {str(row['mean_wer']):<8} "
            f"Mean CER: {str(row['mean_cer']):<8} "
            f"Pass rate: "
            f"{row['wer_target_pass_rate_percent']}%"
        )

    print("=" * 90)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()