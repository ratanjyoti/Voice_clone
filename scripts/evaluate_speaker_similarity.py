from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PROJECT_ROOT / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
HF_CACHE = PROJECT_ROOT / "models" / "huggingface"
RESULTS_DIR = PROJECT_ROOT / "results"
SNAPSHOT_DIR = PROJECT_ROOT / "evidence" / "result_snapshots"
SIMILARITY_THRESHOLD = 0.75

os.environ.setdefault("HF_HOME", str(HF_CACHE))


@dataclass(frozen=True)
class SimilarityJob:
    language: str
    model: str
    generated_root: Path
    references: dict[str, Path]
    output_csv: Path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, original_sample_rate = torchaudio.load(str(path))

    if waveform.numel() == 0:
        raise ValueError(f"empty audio: {path}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=original_sample_rate,
            new_freq=sample_rate,
        )

    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite audio samples: {path}")

    return waveform


def embed_file(
    classifier: EncoderClassifier,
    path: Path,
    cache: dict[Path, torch.Tensor],
) -> torch.Tensor:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]

    waveform = load_audio(resolved)
    with torch.inference_mode():
        embedding = classifier.encode_batch(waveform)

    embedding = embedding.detach().float().cpu().reshape(1, -1)
    if not torch.isfinite(embedding).all():
        raise ValueError(f"non-finite embedding: {path}")

    norm = float(torch.linalg.vector_norm(embedding).item())
    if norm <= 0:
        raise ValueError(f"zero-norm embedding: {path}")

    cache[resolved] = embedding
    return embedding


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(F.cosine_similarity(first, second, dim=1).item())


def generated_wavs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.wav") if path.is_file())


def evaluate_job(
    classifier: EncoderClassifier,
    job: SimilarityJob,
    cache: dict[Path, torch.Tensor],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    references = {
        name: path for name, path in job.references.items() if path.exists()
    }

    for generated in generated_wavs(job.generated_root):
        row: dict[str, object] = {
            "language": job.language,
            "model": job.model,
            "generated_file": rel(generated),
            "status": "success",
            "error": "",
        }

        try:
            generated_embedding = embed_file(classifier, generated, cache)
            scores: list[float] = []

            for reference_name, reference_path in job.references.items():
                column = f"similarity_to_{reference_name}"
                if reference_name not in references:
                    row[column] = ""
                    row["status"] = "missing_reference"
                    row["error"] = f"missing reference: {rel(reference_path)}"
                    continue

                reference_embedding = embed_file(classifier, reference_path, cache)
                score = cosine(generated_embedding, reference_embedding)
                row[column] = round(score, 6)
                scores.append(score)

            if scores:
                row["average_similarity"] = round(mean(scores), 6)
                row["maximum_similarity"] = round(max(scores), 6)
                row["passes_0_75"] = max(scores) >= SIMILARITY_THRESHOLD
            else:
                row["average_similarity"] = ""
                row["maximum_similarity"] = ""
                row["passes_0_75"] = False
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            for reference_name in job.references:
                row.setdefault(f"similarity_to_{reference_name}", "")
            row["average_similarity"] = ""
            row["maximum_similarity"] = ""
            row["passes_0_75"] = False

        rows.append(row)

    if not rows:
        rows.append(
            {
                "language": job.language,
                "model": job.model,
                "generated_file": rel(job.generated_root),
                "status": "no_generated_files",
                "error": f"no wav files found under {rel(job.generated_root)}",
                **{f"similarity_to_{name}": "" for name in job.references},
                "average_similarity": "",
                "maximum_similarity": "",
                "passes_0_75": False,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def successful_scores(rows: list[dict[str, object]], model: str | None = None) -> list[float]:
    scores: list[float] = []
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        if row.get("status") != "success":
            continue
        value = row.get("maximum_similarity")
        if isinstance(value, (int, float)):
            scores.append(float(value))
    return scores


def summarize_arabic(rows: list[dict[str, object]]) -> dict[str, object]:
    xtts_scores = successful_scores(rows, "XTTS-v2")
    chatterbox_scores = successful_scores(rows, "Chatterbox")
    all_scores = successful_scores(rows)
    passing = sum(1 for score in all_scores if score >= SIMILARITY_THRESHOLD)

    return {
        "metric": "SpeechBrain ECAPA-TDNN cosine similarity",
        "threshold": SIMILARITY_THRESHOLD,
        "xtts_average_similarity": round(mean(xtts_scores), 6) if xtts_scores else None,
        "chatterbox_average_similarity": round(mean(chatterbox_scores), 6)
        if chatterbox_scores
        else None,
        "minimum_similarity": round(min(all_scores), 6) if all_scores else None,
        "maximum_similarity": round(max(all_scores), 6) if all_scores else None,
        "samples_passing_0_75": passing,
        "successful_samples": len(all_scores),
        "total_rows": len(rows),
        "models": {
            "XTTS-v2": {
                "average_similarity": round(mean(xtts_scores), 6) if xtts_scores else None,
                "passing_samples": sum(score >= SIMILARITY_THRESHOLD for score in xtts_scores),
                "successful_samples": len(xtts_scores),
            },
            "Chatterbox": {
                "average_similarity": round(mean(chatterbox_scores), 6)
                if chatterbox_scores
                else None,
                "passing_samples": sum(score >= SIMILARITY_THRESHOLD for score in chatterbox_scores),
                "successful_samples": len(chatterbox_scores),
            },
        },
    }


def summarize_language(rows: list[dict[str, object]], language: str) -> dict[str, object]:
    models = sorted({str(row.get("model")) for row in rows})
    summary: dict[str, object] = {
        "language": language,
        "metric": "SpeechBrain ECAPA-TDNN cosine similarity",
        "threshold": SIMILARITY_THRESHOLD,
        "minimum_similarity": None,
        "maximum_similarity": None,
        "samples_passing_0_75": 0,
        "successful_samples": 0,
        "total_rows": len(rows),
        "models": {},
    }

    all_scores = successful_scores(rows)
    if all_scores:
        summary["minimum_similarity"] = round(min(all_scores), 6)
        summary["maximum_similarity"] = round(max(all_scores), 6)
        summary["samples_passing_0_75"] = sum(
            score >= SIMILARITY_THRESHOLD for score in all_scores
        )
        summary["successful_samples"] = len(all_scores)

    for model in models:
        scores = successful_scores(rows, model)
        summary["models"][model] = {
            "average_similarity": round(mean(scores), 6) if scores else None,
            "passing_samples": sum(score >= SIMILARITY_THRESHOLD for score in scores),
            "successful_samples": len(scores),
        }

    return summary


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def build_jobs() -> list[SimilarityJob]:
    arabic_refs = {
        "short": PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "arabic"
        / "professional_msa"
        / "arabic_reference_short.wav",
        "standard": PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "arabic"
        / "professional_msa"
        / "arabic_reference_standard.wav",
        "long": PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "arabic"
        / "professional_msa"
        / "arabic_reference_long.wav",
    }

    english_refs = {
        "reference": PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "english"
        / "ratan_20s_norm.wav"
    }

    hindi_refs = {
        "reference": PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "ratan_reference_22050_mono.wav"
    }

    return [
        SimilarityJob(
            language="Arabic",
            model="XTTS-v2",
            generated_root=PROJECT_ROOT / "outputs" / "xtts" / "arabic",
            references=arabic_refs,
            output_csv=RESULTS_DIR / "arabic_speaker_similarity.csv",
        ),
        SimilarityJob(
            language="Arabic",
            model="Chatterbox",
            generated_root=PROJECT_ROOT / "outputs" / "chatterbox" / "arabic",
            references=arabic_refs,
            output_csv=RESULTS_DIR / "arabic_speaker_similarity.csv",
        ),
        SimilarityJob(
            language="English",
            model="NeuTTS",
            generated_root=PROJECT_ROOT / "outputs" / "neutts" / "english" / "air_evaluation",
            references=english_refs,
            output_csv=RESULTS_DIR / "english_speaker_similarity.csv",
        ),
        SimilarityJob(
            language="Hindi",
            model="Chatterbox",
            generated_root=PROJECT_ROOT / "outputs" / "chatterbox" / "hindi",
            references=hindi_refs,
            output_csv=RESULTS_DIR / "hindi_speaker_similarity.csv",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate voice-cloning speaker cosine similarity."
    )
    parser.add_argument(
        "--language",
        choices=["all", "arabic", "english", "hindi"],
        default="all",
        help="Subset to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_language = args.language.lower()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    print("Loading SpeechBrain ECAPA-TDNN speaker embedding model on CPU...")
    start = time.perf_counter()
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_CACHE),
        run_opts={"device": "cpu"},
    )
    print(f"Model loaded in {time.perf_counter() - start:.2f} seconds.")

    jobs = build_jobs()
    if selected_language != "all":
        jobs = [job for job in jobs if job.language.lower() == selected_language]

    cache: dict[Path, torch.Tensor] = {}
    grouped: dict[str, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []

    for job in jobs:
        print(f"Evaluating {job.language} / {job.model}: {rel(job.generated_root)}")
        rows = evaluate_job(classifier, job, cache)
        grouped.setdefault(job.language, []).extend(rows)
        all_rows.extend(rows)

    if "Arabic" in grouped:
        write_csv(RESULTS_DIR / "arabic_speaker_similarity.csv", grouped["Arabic"])
        write_json(
            SNAPSHOT_DIR / "arabic_speaker_similarity_summary.json",
            summarize_arabic(grouped["Arabic"]),
        )

    if "English" in grouped:
        write_csv(RESULTS_DIR / "english_speaker_similarity.csv", grouped["English"])
        write_json(
            SNAPSHOT_DIR / "english_speaker_similarity_summary.json",
            summarize_language(grouped["English"], "English"),
        )

    if "Hindi" in grouped:
        write_csv(RESULTS_DIR / "hindi_speaker_similarity.csv", grouped["Hindi"])
        write_json(
            SNAPSHOT_DIR / "hindi_speaker_similarity_summary.json",
            summarize_language(grouped["Hindi"], "Hindi"),
        )

    if all_rows:
        write_csv(RESULTS_DIR / "all_language_speaker_similarity.csv", all_rows)

    print("Speaker similarity evaluation complete.")


if __name__ == "__main__":
    main()
