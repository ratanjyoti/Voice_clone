import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test_sentences"

FILES = {
    "English": TEST_DATA_DIR / "english.json",
    "Arabic": TEST_DATA_DIR / "arabic.json",
    "Hindi": TEST_DATA_DIR / "hindi.json",
}


def validate_file(language: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{language} file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"{language}: JSON root must be a list.")

    if len(records) == 0:
        raise ValueError(f"{language}: no test sentences found.")

    seen_ids = set()

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{language}: item {index} must be an object.")

        for field in ("id", "category", "text"):
            value = record.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{language}: item {index} has an invalid '{field}'."
                )

        if record["id"] in seen_ids:
            raise ValueError(
                f"{language}: duplicate sentence ID '{record['id']}'."
            )

        seen_ids.add(record["id"])

    print(f"{language}: {len(records)} valid test sentences")

    for record in records:
        print(
            f"  {record['id']} | "
            f"{record['category']} | "
            f"{record['text']}"
        )


def main() -> None:
    print("Validating multilingual test data")
    print("=" * 60)

    for language, path in FILES.items():
        validate_file(language, path)
        print()

    print("All test sentence files are valid.")


if __name__ == "__main__":
    main()