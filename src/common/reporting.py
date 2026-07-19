from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SNAPSHOT_DIR = PROJECT_ROOT / "evidence" / "result_snapshots"
DOCS_DIR = PROJECT_ROOT / "docs"

STANDARD_COLUMNS = [
    "language",
    "model",
    "test_id",
    "expected_text",
    "ASR_text",
    "WER_percent",
    "generation_time",
    "audio_duration",
    "RTF",
    "peak",
    "clipping",
    "clipping_samples",
    "speaker_similarity",
    "manual_naturalness",
    "manual_pronunciation",
    "status",
    "output_path",
    "notes",
]

LANGUAGE_OUTPUTS = {
    "English": RESULTS_DIR / "english_model_comparison.csv",
    "Hindi": RESULTS_DIR / "hindi_model_comparison.csv",
    "Arabic": RESULTS_DIR / "arabic_model_comparison.csv",
}
FINAL_OUTPUT = RESULTS_DIR / "final_model_summary.csv"

SOURCE_FILES = {
    "English": [
        RESULTS_DIR / "mms_english_evaluation.csv",
        RESULTS_DIR / "neutts_english_evaluation.csv",
        RESULTS_DIR / "english_speaker_similarity.csv",
        SNAPSHOT_DIR / "english_model_summary.json",
        SNAPSHOT_DIR / "english_speaker_similarity_summary.json",
        RESULTS_DIR / "human_listener_scores.csv",
        RESULTS_DIR / "human_evaluation_summary.csv",
    ],
    "Hindi": [
        RESULTS_DIR / "mms_hindi_evaluation.csv",
        RESULTS_DIR / "chatterbox_hindi_evaluation.csv",
        RESULTS_DIR / "indicf5_hindi_initial_test.csv",
        RESULTS_DIR / "hindi_speaker_similarity.csv",
        RESULTS_DIR / "hindi_greeting_model_comparison.csv",
        SNAPSHOT_DIR / "hindi_model_summary.json",
        SNAPSHOT_DIR / "hindi_speaker_similarity_summary.json",
        SNAPSHOT_DIR / "indicf5_hindi_initial_generation_success.json",
        RESULTS_DIR / "human_listener_scores.csv",
        RESULTS_DIR / "human_evaluation_summary.csv",
    ],
    "Arabic": [
        RESULTS_DIR / "xtts_arabic_evaluation.csv",
        RESULTS_DIR / "chatterbox_arabic_evaluation.csv",
        RESULTS_DIR / "arabic_speaker_similarity.csv",
        SNAPSHOT_DIR / "arabic_model_comparison_summary.json",
        SNAPSHOT_DIR / "arabic_speaker_similarity_summary.json",
        SNAPSHOT_DIR / "xtts" / "arabic" / "xtts_arabic_summary.json",
        SNAPSHOT_DIR / "chatterbox" / "arabic" / "chatterbox_arabic_summary.json",
        RESULTS_DIR / "human_listener_scores.csv",
        RESULTS_DIR / "human_evaluation_summary.csv",
    ],
}

INPUT_CSVS = {
    "English": [
        (RESULTS_DIR / "mms_english_evaluation.csv", "MMS-TTS"),
        (RESULTS_DIR / "neutts_english_evaluation.csv", "NeuTTS"),
    ],
    "Hindi": [
        (RESULTS_DIR / "mms_hindi_evaluation.csv", "MMS-TTS"),
        (RESULTS_DIR / "chatterbox_hindi_evaluation.csv", "Chatterbox"),
        (RESULTS_DIR / "indicf5_hindi_initial_test.csv", "IndicF5"),
    ],
    "Arabic": [
        (RESULTS_DIR / "xtts_arabic_evaluation.csv", "XTTS-v2"),
        (RESULTS_DIR / "chatterbox_arabic_evaluation.csv", "Chatterbox"),
    ],
}

MODEL_STRENGTHS = {
    ("English", "MMS-TTS"): "Fast CPU baseline, passed average English WER target, stable/no clipping in final samples.",
    ("English", "NeuTTS"): "Voice-cloning candidate with final samples available and no clipping in this run.",
    ("Hindi", "MMS-TTS"): "Fastest Hindi baseline with no clipping in final samples.",
    ("Hindi", "Chatterbox"): "Best Hindi WER among the five-sample Hindi models tested and provides voice cloning.",
    ("Hindi", "IndicF5"): "Hindi-specialized model became runnable after authenticated access and CPU compatibility fixes; one sample had no clipping.",
    ("Arabic", "XTTS-v2"): "Best Arabic automatic WER result and no clipping in final samples.",
    ("Arabic", "Chatterbox"): "Arabic voice-cloning candidate with higher speaker cosine than XTTS-v2 in the available summary, though still below the 0.75 target.",
}

MODEL_WEAKNESSES = {
    ("English", "MMS-TTS"): "Not a voice-cloning pipeline; speaker similarity is not applicable.",
    ("English", "NeuTTS"): "Missed the 10% average WER target, speaker cosine target, and CPU latency/RTF target.",
    ("Hindi", "MMS-TTS"): "Missed Hindi WER target and is not a cloning pipeline.",
    ("Hindi", "Chatterbox"): "Missed Hindi WER target, very slow on CPU, and some final samples clipped.",
    ("Hindi", "IndicF5"): "Only one greeting sample was evaluated; WER was worse than Chatterbox and CPU RTF was very high.",
    ("Arabic", "XTTS-v2"): "Failed CPU latency/RTF target and speaker cosine target; licensing needs review.",
    ("Arabic", "Chatterbox"): "Failed Arabic WER target and was slower than XTTS-v2 in the final Arabic benchmark.",
}

WINNERS = {
    "English": "MMS-TTS is the automatic English baseline winner because it passed average WER and CPU RTF; it is not a cloning winner.",
    "Hindi": "No production winner. Chatterbox is the best tested Hindi cloning candidate, but it fails WER/latency and has clipping; MMS is faster but not cloning; IndicF5 should not expand beyond the initial sample yet.",
    "Arabic": "XTTS-v2 is the automatic Arabic WER winner, with clear caveats for CPU latency, RTF, speaker similarity, and licensing.",
}


def rel(path: Path | str) -> str:
    if not path:
        return ""
    p = Path(str(path))
    try:
        if p.is_absolute():
            return str(p.relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        pass
    return str(path).replace("/", "\\")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def average(values: list[Any]) -> float | None:
    floats = [v for v in (as_float(value) for value in values) if v is not None]
    return mean(floats) if floats else None


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    number = as_float(value)
    if number is not None:
        return f"{number:.3f}".rstrip("0").rstrip(".")
    return str(value)


def boolish(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def normalize_model(language: str, fallback: str, row_model: str = "") -> str:
    model = row_model or fallback
    text = model.lower()
    if language == "Arabic" and "xtts" in text:
        return "XTTS-v2"
    if "chatterbox" in text:
        return "Chatterbox"
    if "neutts" in text:
        return "NeuTTS"
    if "mms" in text:
        return "MMS-TTS"
    if "indicf5" in text:
        return "IndicF5"
    return fallback or model


def output_key(path: str) -> str:
    return rel(path).lower().replace("\\", "/")


def load_speaker_similarity() -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for path in [
        RESULTS_DIR / "english_speaker_similarity.csv",
        RESULTS_DIR / "hindi_speaker_similarity.csv",
        RESULTS_DIR / "arabic_speaker_similarity.csv",
    ]:
        for row in read_csv(path):
            language = str(row.get("language", "")).strip().title()
            model = normalize_model(language, row.get("model", ""), row.get("model", ""))
            score = row.get("average_similarity") or row.get("maximum_similarity") or row.get("similarity_to_reference")
            generated = row.get("generated_file", "")
            if generated and score not in (None, ""):
                mapping[(language, output_key(generated))] = str(score)
            if score not in (None, ""):
                mapping.setdefault((language, model), str(score))
    return mapping


def load_manual_scores() -> dict[tuple[str, str], dict[str, str]]:
    scores: dict[tuple[str, str], dict[str, str]] = {}
    rows = read_csv(RESULTS_DIR / "human_listener_scores.csv")
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        language = str(row.get("language", "")).strip().title()
        sample_id = row.get("anonymous_sample_id", "")
        for source, target in [
            ("naturalness_score_1_to_5", "manual_naturalness"),
            ("pronunciation_score_1_to_5", "manual_pronunciation"),
        ]:
            value = as_float(row.get(source))
            if language and sample_id and value is not None:
                grouped[(language, sample_id)][target].append(value)
    for key, values in grouped.items():
        scores[key] = {metric: fmt(mean(metric_values)) for metric, metric_values in values.items() if metric_values}
    return scores


def row_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def build_rows() -> tuple[dict[str, list[dict[str, str]]], dict[str, list[str]]]:
    speaker = load_speaker_similarity()
    manual = load_manual_scores()
    rows_by_language: dict[str, list[dict[str, str]]] = {language: [] for language in INPUT_CSVS}
    sources_used: dict[str, list[str]] = {language: [] for language in INPUT_CSVS}

    for language, inputs in INPUT_CSVS.items():
        for path, fallback_model in inputs:
            input_rows = read_csv(path)
            if not input_rows:
                continue
            sources_used[language].append(rel(path))
            for raw in input_rows:
                model = normalize_model(language, fallback_model, raw.get("model", ""))
                output_path = row_value(raw, "output_path")
                if output_path:
                    sidecar_path = Path(output_path)
                    if not sidecar_path.is_absolute():
                        sidecar_path = PROJECT_ROOT / sidecar_path
                    sidecar_path = sidecar_path.with_suffix(".json")
                    if sidecar_path.exists() and rel(sidecar_path) not in sources_used[language]:
                        sources_used[language].append(rel(sidecar_path))
                key = (language, output_key(output_path))
                speaker_score = speaker.get(key) or speaker.get((language, model), "")
                manual_key = (language, Path(output_path).stem if output_path else raw.get("test_id", ""))
                manual_scores = manual.get(manual_key, {})

                wer_percent = row_value(raw, "WER_percent", "wer_percent", "wer_percentage")
                if wer_percent == "":
                    wer = row_value(raw, "WER", "wer")
                    wer_float = as_float(wer)
                    wer_percent = fmt(wer_float * 100) if wer_float is not None else ""

                clipping = row_value(raw, "clipping")
                clipping_samples = row_value(raw, "clipping_samples", "clipping_count")
                if clipping == "" and clipping_samples != "":
                    count = as_float(clipping_samples)
                    clipping = str(count is not None and count > 0)

                notes: list[str] = [f"source={rel(path)}"]
                if model == "IndicF5":
                    notes.append("initial one-sample test only; do not compare as a five-sample benchmark")
                if model == "MMS-TTS":
                    notes.append("fixed-speaker baseline, not voice cloning")
                if speaker_score == "" and model == "MMS-TTS":
                    notes.append("speaker similarity not applicable")
                if not manual_scores:
                    notes.append("manual MOS pending")

                rows_by_language[language].append(
                    {
                        "language": language,
                        "model": model,
                        "test_id": row_value(raw, "test_id"),
                        "expected_text": row_value(raw, "expected_text", "reference_text"),
                        "ASR_text": row_value(raw, "ASR_text", "asr_text", "asr_transcript"),
                        "WER_percent": fmt(wer_percent),
                        "generation_time": fmt(row_value(raw, "generation_time")),
                        "audio_duration": fmt(row_value(raw, "audio_duration", "duration")),
                        "RTF": fmt(row_value(raw, "RTF", "rtf")),
                        "peak": fmt(row_value(raw, "peak")),
                        "clipping": clipping,
                        "clipping_samples": row_value(raw, "clipping_samples", "clipping_count"),
                        "speaker_similarity": fmt(speaker_score),
                        "manual_naturalness": manual_scores.get("manual_naturalness", ""),
                        "manual_pronunciation": manual_scores.get("manual_pronunciation", ""),
                        "status": row_value(raw, "status") or "completed",
                        "output_path": rel(output_path),
                        "notes": "; ".join(notes),
                    }
                )

    for language, paths in SOURCE_FILES.items():
        for path in paths:
            if path.exists() and rel(path) not in sources_used[language]:
                sources_used[language].append(rel(path))
    return rows_by_language, sources_used


def preserve_existing_target(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            header = next(csv.reader(file), [])
    except StopIteration:
        return
    if header == STANDARD_COLUMNS:
        return
    legacy = path.with_name(path.stem + ".legacy_pre_language_report" + path.suffix)
    if not legacy.exists():
        shutil.copy2(path, legacy)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preserve_existing_target(path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in STANDARD_COLUMNS})


def model_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    stats: dict[str, dict[str, Any]] = {}
    for model, model_rows in grouped.items():
        clipping_count = 0
        clipping_known = 0
        for row in model_rows:
            clipped = boolish(row.get("clipping"))
            if clipped is not None:
                clipping_known += 1
                clipping_count += int(clipped)
            elif as_float(row.get("clipping_samples")) is not None:
                clipping_known += 1
                clipping_count += int(as_float(row.get("clipping_samples")) or 0 > 0)
        stats[model] = {
            "samples": len(model_rows),
            "average_WER_percent": average([row["WER_percent"] for row in model_rows]),
            "average_generation_time": average([row["generation_time"] for row in model_rows]),
            "average_RTF": average([row["RTF"] for row in model_rows]),
            "clipping_count": clipping_count,
            "clipping_known": clipping_known,
            "speaker_similarity": average([row["speaker_similarity"] for row in model_rows]),
            "manual_naturalness": average([row["manual_naturalness"] for row in model_rows]),
            "manual_pronunciation": average([row["manual_pronunciation"] for row in model_rows]),
            "statuses": sorted({row.get("status", "") for row in model_rows if row.get("status", "")}),
        }
    return stats


def markdown_table(headers: list[str], data: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def language_doc(language: str, rows: list[dict[str, str]], sources: list[str]) -> str:
    stats = model_stats(rows)
    model_names = sorted(stats)
    summary_rows = []
    for model in model_names:
        item = stats[model]
        clip = f"{item['clipping_count']}/{item['clipping_known']}" if item["clipping_known"] else "NA"
        summary_rows.append([
            model,
            str(item["samples"]),
            fmt(item["average_WER_percent"]),
            fmt(item["average_generation_time"]),
            fmt(item["average_RTF"]),
            clip,
            fmt(item["speaker_similarity"]),
            fmt(item["manual_naturalness"]) or "pending",
            fmt(item["manual_pronunciation"]) or "pending",
        ])

    lines = [
        f"# {language} Results",
        "",
        "This report is rebuilt from existing CSV/JSON evidence by `python scripts/build_language_reports.py`. Missing values are left blank or marked `NA`; no metrics are invented.",
        "",
        "## Models tested",
    ]
    for model in model_names:
        lines.append(f"- {model}")
    lines += [
        "",
        "## Summary",
        markdown_table(
            ["Model", "Samples", "Avg WER %", "Avg gen s", "Avg RTF", "Clipping", "Speaker sim", "MOS", "Pronunciation"],
            summary_rows,
        ),
        "",
        "## Strengths",
    ]
    for model in model_names:
        lines.append(f"- {model}: {MODEL_STRENGTHS.get((language, model), 'Recorded in evidence; review detailed rows for behavior.')}")
    lines += ["", "## Weaknesses"]
    for model in model_names:
        lines.append(f"- {model}: {MODEL_WEAKNESSES.get((language, model), 'No extra weakness note recorded beyond the metrics table.')}")
    lines += [
        "",
        "## Manual listening status",
        "Manual MOS/listener fields are pending unless values appear in the summary table. Current `human_listener_scores.csv` does not contain real reviewer scores for these samples.",
        "",
        "## Winner",
        WINNERS.get(language, "No winner recorded."),
        "",
        "## Evidence paths",
    ]
    for source in sources:
        lines.append(f"- `{source}`")
    lines += ["", "## Consolidated rows", f"See `results\\{language.lower()}_model_comparison.csv`." , ""]
    return "\n".join(lines)


def final_doc(rows_by_language: dict[str, list[dict[str, str]]], sources_by_language: dict[str, list[str]]) -> str:
    data = []
    for language, rows in rows_by_language.items():
        stats = model_stats(rows)
        for model, item in sorted(stats.items()):
            data.append([
                language,
                model,
                str(item["samples"]),
                fmt(item["average_WER_percent"]),
                fmt(item["average_generation_time"]),
                fmt(item["average_RTF"]),
                fmt(item["speaker_similarity"]),
                fmt(item["manual_naturalness"]) or "pending",
            ])
    lines = [
        "# Final Recommendation",
        "",
        "This is the human-readable cross-language view rebuilt from existing evidence. Manual MOS remains pending; recommendations below are automatic-metric recommendations only.",
        "",
        markdown_table(["Language", "Model", "Samples", "Avg WER %", "Avg gen s", "Avg RTF", "Speaker sim", "MOS"], data),
        "",
        "## Recommendations",
        f"- English: {WINNERS['English']}",
        f"- Hindi: {WINNERS['Hindi']}",
        f"- Arabic: {WINNERS['Arabic']}",
        "",
        "## Important caveats",
        "- Manual listener scores are not complete, so MOS-based production approval is still pending.",
        "- MMS-TTS is a fixed-speaker baseline and should not be treated as a voice-cloning result.",
        "- IndicF5 has only a one-sample Hindi result in this evidence set.",
        "- CPU latency/RTF is not representative of optimized GPU deployment.",
        "",
        "## Source files used",
    ]
    for language, sources in sources_by_language.items():
        lines.append(f"### {language}")
        for source in sources:
            lines.append(f"- `{source}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows_by_language, sources_by_language = build_rows()
    all_rows: list[dict[str, str]] = []
    for language, rows in rows_by_language.items():
        write_csv(LANGUAGE_OUTPUTS[language], rows)
        all_rows.extend(rows)
        (DOCS_DIR / f"{language.lower()}_results.md").write_text(
            language_doc(language, rows, sources_by_language[language]),
            encoding="utf-8",
        )
    write_csv(FINAL_OUTPUT, all_rows)
    (DOCS_DIR / "final_recommendation.md").write_text(
        final_doc(rows_by_language, sources_by_language),
        encoding="utf-8",
    )
    print("Built language comparison CSVs and Markdown reports.")
    for language, path in LANGUAGE_OUTPUTS.items():
        print(f"{language}: {rel(path)} ({len(rows_by_language[language])} rows)")
    print(f"Final: {rel(FINAL_OUTPUT)} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()


