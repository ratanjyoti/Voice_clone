from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANG_ROOT = PROJECT_ROOT / "languages"
LANGUAGES = ["english", "hindi", "arabic", "shared"]

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".venv-arabic-data",
    ".venv-indicf5",
    ".venv-melotts",
    ".venv-neutts",
    ".venv-openvoice",
    ".venv-silma",
    ".venv-speaker-eval",
    ".venv-speaker-similarity",
    ".venv-ui",
    ".venv-xtts-arabic",
    "models",
    "downloads",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

THIRD_PARTY_ROOTS = {"external"}
GENERATED_LANGUAGE_ROOT = "languages"

LANGUAGE_PATTERNS = {
    "english": ["english", "_english", "english_", " en_", "en_", "mms_english", "neutts", "melotts", "openvoice", "ratan_20s", "ratan_neutral", "ratan_conversational", "ratan_expressive"],
    "hindi": ["hindi", "_hindi", "hindi_", " hi_", "hi_", "indicf5", "ratan_hindi", "chatterbox_hindi"],
    "arabic": ["arabic", "_arabic", "arabic_", " ar_", "ar_", "xtts", "professional_msa", "chatterbox_arabic"],
}

CURATED_COMMANDS = {
    "english": [
        "python scripts\\build_language_reports.py",
        ".\\.venv\\Scripts\\python.exe -u scripts\\generate_final_mms.py --language english",
        ".\\.venv-neutts\\Scripts\\python.exe -u scripts\\generate_final_neutts_english.py",
        ".\\.venv\\Scripts\\python.exe -u scripts\\evaluate_final_samples.py --target english",
    ],
    "hindi": [
        "python scripts\\build_language_reports.py",
        ".\\.venv\\Scripts\\python.exe -u scripts\\generate_final_mms.py --language hindi",
        ".\\.venv\\Scripts\\python.exe -u scripts\\generate_final_chatterbox_hindi.py",
        ".\\.venv\\Scripts\\python.exe -u scripts\\evaluate_final_samples.py --target hindi",
        ".\\.venv-indicf5\\Scripts\\python.exe -u src\\generate_indicf5_hindi_test.py",
    ],
    "arabic": [
        "python scripts\\build_language_reports.py",
        ".\\.venv-xtts-arabic\\Scripts\\python.exe -u src\\generate_xtts_arabic.py",
        ".\\.venv-xtts-arabic\\Scripts\\python.exe -u scripts\\evaluate_xtts_arabic.py",
        ".\\.venv\\Scripts\\python.exe -u src\\generate_chatterbox_arabic.py",
        ".\\.venv\\Scripts\\python.exe -u scripts\\evaluate_chatterbox_arabic.py",
    ],
}

PRIMARY_FILES = {
    "english": [
        "docs/english_results.md",
        "results/english_model_comparison.csv",
        "results/mms_english_evaluation.csv",
        "results/neutts_english_evaluation.csv",
        "scripts/generate_final_mms.py",
        "scripts/generate_final_neutts_english.py",
        "src/generate_mms.py",
        "src/generate_neutts_english.py",
        "data/test_sentences/english_voice_clone_tests.json",
    ],
    "hindi": [
        "docs/hindi_results.md",
        "results/hindi_model_comparison.csv",
        "results/mms_hindi_evaluation.csv",
        "results/chatterbox_hindi_evaluation.csv",
        "results/indicf5_hindi_initial_test.csv",
        "scripts/generate_final_mms.py",
        "scripts/generate_final_chatterbox_hindi.py",
        "scripts/evaluate_indicf5_hindi_initial.py",
        "src/generate_chatterbox_hindi.py",
        "src/generate_indicf5_hindi_test.py",
        "data/test_sentences/hindi_voice_clone_tests.json",
    ],
    "arabic": [
        "docs/arabic_results.md",
        "results/arabic_model_comparison.csv",
        "results/xtts_arabic_evaluation.csv",
        "results/chatterbox_arabic_evaluation.csv",
        "scripts/evaluate_xtts_arabic.py",
        "scripts/evaluate_chatterbox_arabic.py",
        "src/generate_xtts_arabic.py",
        "src/generate_chatterbox_arabic.py",
        "data/test_sentences/arabic_voice_clone_tests.json",
    ],
}


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def should_skip(path: Path) -> bool:
    parts = set(path.relative_to(PROJECT_ROOT).parts)
    if parts & EXCLUDED_PARTS:
        return True
    if path.parts and GENERATED_LANGUAGE_ROOT in path.relative_to(PROJECT_ROOT).parts:
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False


def iter_project_files() -> Iterable[Path]:
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        relative_parts = set(root_path.relative_to(PROJECT_ROOT).parts) if root_path != PROJECT_ROOT else set()
        if relative_parts & EXCLUDED_PARTS:
            dirs[:] = []
            continue
        if GENERATED_LANGUAGE_ROOT in relative_parts:
            dirs[:] = []
            continue
        dirs[:] = [
            dirname for dirname in dirs
            if dirname not in EXCLUDED_PARTS and dirname != GENERATED_LANGUAGE_ROOT
        ]
        for filename in files:
            path = root_path / filename
            if should_skip(path):
                continue
            yield path


def classify_language(path: Path) -> str:
    relative = rel(path).lower().replace("/", " ").replace("-", "_")
    parts = path.relative_to(PROJECT_ROOT).parts

    if parts and parts[0] in THIRD_PARTY_ROOTS:
        return "shared"

    scores: dict[str, int] = {}
    for language, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if pattern.strip() in relative:
                score += 1
        scores[language] = score

    best_language = max(scores, key=scores.get)
    if scores[best_language] == 0:
        return "shared"
    return best_language


def classify_kind(path: Path) -> str:
    first = path.relative_to(PROJECT_ROOT).parts[0]
    suffix = path.suffix.lower()
    if first == "src":
        return "source"
    if first == "scripts":
        return "script"
    if first == "results":
        return "result"
    if first == "outputs":
        return "output"
    if first == "data":
        return "data"
    if first == "docs":
        return "doc"
    if first == "report":
        return "report"
    if first == "evidence":
        return "evidence"
    if suffix in {".md", ".txt"}:
        return "doc"
    if suffix in {".csv", ".json"}:
        return "data"
    return "project"


def note_for(path: Path, language: str) -> str:
    text = rel(path).lower()
    if language == "shared":
        if text.startswith("external/"):
            return "third-party source; do not edit unless intentionally patching dependency"
        return "shared project file used across languages"
    if text.startswith("results/"):
        return "metric/result evidence for this language"
    if text.startswith("outputs/"):
        return "generated audio or sidecar evidence for this language"
    if text.startswith("data/test_sentences"):
        return "benchmark text set for this language"
    if text.startswith("data/reference_audio"):
        return "reference voice/audio for this language"
    if text.startswith("src/") or text.startswith("scripts/"):
        return "implementation or evaluation code for this language"
    if text.startswith("docs/"):
        return "human-readable documentation for this language"
    if text.startswith("evidence/"):
        return "execution proof/log/snapshot for this language"
    return "language-related file"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["language", "kind", "path", "notes"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], limit: int = 30) -> str:
    lines = ["| Kind | Path | Notes |", "| --- | --- | --- |"]
    for row in rows[:limit]:
        lines.append(f"| {row['kind']} | `{row['path']}` | {row['notes']} |")
    if len(rows) > limit:
        lines.append(f"| ... | See `files.csv` | {len(rows) - limit} more files |")
    return "\n".join(lines)


def write_language_readme(language: str, rows: list[dict[str, str]]) -> None:
    title = language.title() if language != "shared" else "Shared"
    by_kind: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)

    primary = [path for path in PRIMARY_FILES.get(language, []) if (PROJECT_ROOT / path).exists()]
    commands = CURATED_COMMANDS.get(language, ["python scripts\\build_language_reports.py"])

    lines = [
        f"# {title} Workspace",
        "",
        "This folder is an index for language-specific work. The actual source/evidence files stay in their original locations so existing scripts and reports keep working.",
        "",
        "## Start here",
    ]
    if primary:
        for path in primary:
            lines.append(f"- `{path}`")
    else:
        lines.append("- See `files.csv` for shared files.")

    lines += ["", "## Common commands"]
    for command in commands:
        lines.append(f"```powershell\n{command}\n```")

    lines += ["", "## File groups"]
    for kind in sorted(by_kind):
        lines.append(f"- {kind}: {len(by_kind[kind])} files")

    lines += ["", "## Most relevant files", markdown_table(rows), "", "## Full manifest", "See `files.csv`.", ""]
    (LANG_ROOT / language / "README.md").write_text("\n".join(lines), encoding="utf-8")

    for kind, kind_rows in by_kind.items():
        kind_path = LANG_ROOT / language / f"{kind}_files.md"
        kind_lines = [f"# {title} {kind.title()} Files", "", markdown_table(kind_rows, limit=200), ""]
        kind_path.write_text("\n".join(kind_lines), encoding="utf-8")


def write_top_readme(rows_by_language: dict[str, list[dict[str, str]]]) -> None:
    lines = [
        "# Language Workspaces",
        "",
        "Use these folders when you want to work on one language without searching the whole repository.",
        "",
        "The files are indexed here, not moved. That keeps existing generation/evaluation scripts working.",
        "",
        "## Folders",
        "- `english/`: English models, data, outputs, results, and docs.",
        "- `hindi/`: Hindi models, data, outputs, results, and docs.",
        "- `arabic/`: Arabic models, data, outputs, results, and docs.",
        "- `shared/`: cross-language scripts, configs, reports, dashboard files, and third-party repository pointers.",
        "",
        "## Counts",
        "| Workspace | Files |",
        "| --- | --- |",
    ]
    for language in LANGUAGES:
        lines.append(f"| {language} | {len(rows_by_language.get(language, []))} |")
    lines += [
        "",
        "## Rebuild",
        "```powershell",
        "python scripts\\build_language_workspaces.py",
        "```",
        "",
    ]
    (LANG_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows_by_language: dict[str, list[dict[str, str]]] = {language: [] for language in LANGUAGES}
    for path in iter_project_files():
        language = classify_language(path)
        row = {
            "language": language,
            "kind": classify_kind(path),
            "path": rel(path),
            "notes": note_for(path, language),
        }
        rows_by_language[language].append(row)

    for language, rows in rows_by_language.items():
        rows.sort(key=lambda row: (row["kind"], row["path"]))
        folder = LANG_ROOT / language
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "files.csv", rows)
        write_language_readme(language, rows)

    write_top_readme(rows_by_language)
    print("Built language workspaces:")
    for language in LANGUAGES:
        print(f"- languages/{language}: {len(rows_by_language[language])} indexed files")


if __name__ == "__main__":
    main()

