from __future__ import annotations

import json
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.english_tts_normalizer import normalize_for_tts


CASES = {
    "year_digits": "The access code expires in 2065.",
    "time": "The callback is scheduled for 3:30 today.",
    "natural_number": "Please send 25 copies.",
    "currency": "The balance is $5.",
    "acronym": "AI systems should explain decisions.",
}


def main() -> None:
    results = []
    for test_id, text in CASES.items():
        normalized = normalize_for_tts(text)
        row = {
            "test_id": test_id,
            "expected_text": normalized.expected_text,
            "tts_input_text": normalized.tts_input_text,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    checks = {
        "2065": "two zero six five" in results[0]["tts_input_text"],
        "3:30": "three thirty" in results[1]["tts_input_text"],
        "25": "twenty five" in results[2]["tts_input_text"],
        "$5": "five dollars" in results[3]["tts_input_text"],
        "AI": "A I" in results[4]["tts_input_text"],
        "expected_preserved": all(
            row["expected_text"] == CASES[row["test_id"]]
            for row in results
        ),
    }
    print(json.dumps({"checks": checks}, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
