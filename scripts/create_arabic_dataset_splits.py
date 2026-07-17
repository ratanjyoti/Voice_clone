from __future__ import annotations

import csv
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "training"
    / "arabic"
    / "professional_msa"
)

METADATA_PATH = DATA_ROOT / "metadata_all.csv"
SPLITS_DIR = DATA_ROOT / "splits"
SUMMARY_PATH = SPLITS_DIR / "split_summary.json"

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METADATA_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file_handle:
        reader = csv.DictReader(file_handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if row.get("usable", "").strip() == "Yes"
        ]

    if not rows:
        raise ValueError("No usable rows were found.")

    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(rows)

    total = len(rows)
    train_count = int(total * TRAIN_RATIO)
    validation_count = int(
        total * VALIDATION_RATIO
    )

    train_rows = rows[:train_count]

    validation_rows = rows[
        train_count:
        train_count + validation_count
    ]

    test_rows = rows[
        train_count + validation_count:
    ]

    split_fieldnames = fieldnames + ["split"]

    for row in train_rows:
        row["split"] = "train"

    for row in validation_rows:
        row["split"] = "validation"

    for row in test_rows:
        row["split"] = "test"

    write_csv(
        SPLITS_DIR / "train.csv",
        train_rows,
        split_fieldnames,
    )

    write_csv(
        SPLITS_DIR / "validation.csv",
        validation_rows,
        split_fieldnames,
    )

    write_csv(
        SPLITS_DIR / "test.csv",
        test_rows,
        split_fieldnames,
    )

    summary = {
        "random_seed": RANDOM_SEED,
        "total_usable_clips": total,
        "train_clips": len(train_rows),
        "validation_clips": len(validation_rows),
        "test_clips": len(test_rows),
        "train_ratio": len(train_rows) / total,
        "validation_ratio": len(validation_rows) / total,
        "test_ratio": len(test_rows) / total,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("Arabic dataset splits created")
    print("=" * 60)
    print(f"Total usable: {total}")
    print(f"Train: {len(train_rows)}")
    print(f"Validation: {len(validation_rows)}")
    print(f"Test: {len(test_rows)}")
    print(f"Seed: {RANDOM_SEED}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()