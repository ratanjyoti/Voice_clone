import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from openvoice.api import ToneColorConverter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = (
    PROJECT_ROOT
    / "checkpoints_v2"
    / "converter"
    / "config.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints_v2"
    / "converter"
    / "checkpoint.pth"
)

SOURCE_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "openvoice"
    / "embeddings"
    / "mms_english_en_01_source_se.pth"
)

TARGET_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "openvoice"
    / "embeddings"
    / "ratan_target_se.pth"
)

CLONED_AUDIO_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "openvoice"
    / "english"
    / "en_01_openvoice_cloned.wav"
)

CLONED_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "openvoice"
    / "embeddings"
    / "en_01_cloned_se.pth"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "openvoice_english_similarity.json"
)


def verify_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest().upper()


def load_embedding(path: Path) -> torch.Tensor:
    embedding = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(embedding, torch.Tensor):
        raise TypeError(
            f"Expected tensor in {path}, "
            f"received {type(embedding).__name__}"
        )

    embedding = embedding.detach().float().cpu()

    if not torch.isfinite(embedding).all():
        raise ValueError(
            f"Embedding contains NaN or infinity: {path}"
        )

    if torch.linalg.vector_norm(embedding).item() <= 0:
        raise ValueError(
            f"Embedding has zero norm: {path}"
        )

    return embedding


def cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    first_flat = first.reshape(1, -1)
    second_flat = second.reshape(1, -1)

    if first_flat.shape != second_flat.shape:
        raise ValueError(
            "Embedding shapes do not match: "
            f"{first_flat.shape} vs {second_flat.shape}"
        )

    score = F.cosine_similarity(
        first_flat,
        second_flat,
        dim=1,
    )

    return float(score.item())


def main() -> None:
    required_files = (
        (CONFIG_PATH, "Converter configuration"),
        (CHECKPOINT_PATH, "Converter checkpoint"),
        (SOURCE_EMBEDDING_PATH, "Source embedding"),
        (TARGET_EMBEDDING_PATH, "Target embedding"),
        (CLONED_AUDIO_PATH, "Cloned audio"),
    )

    for path, label in required_files:
        verify_file(path, label)

    CLONED_EMBEDDING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("OpenVoice embedding similarity evaluation")
    print("=" * 76)

    load_start = time.perf_counter()

    converter = ToneColorConverter(
        str(CONFIG_PATH),
        device="cpu",
    )

    converter.load_ckpt(
        str(CHECKPOINT_PATH)
    )

    converter.model.eval()

    converter_load_seconds = (
        time.perf_counter() - load_start
    )

    extraction_start = time.perf_counter()

    with torch.inference_mode():
        converter.extract_se(
            str(CLONED_AUDIO_PATH),
            se_save_path=str(
                CLONED_EMBEDDING_PATH
            ),
        )

    extraction_seconds = (
        time.perf_counter() - extraction_start
    )

    verify_file(
        CLONED_EMBEDDING_PATH,
        "Cloned output embedding",
    )

    source_embedding = load_embedding(
        SOURCE_EMBEDDING_PATH
    )

    target_embedding = load_embedding(
        TARGET_EMBEDDING_PATH
    )

    cloned_embedding = load_embedding(
        CLONED_EMBEDDING_PATH
    )

    source_to_target = cosine_similarity(
        source_embedding,
        target_embedding,
    )

    cloned_to_target = cosine_similarity(
        cloned_embedding,
        target_embedding,
    )

    source_to_cloned = cosine_similarity(
        source_embedding,
        cloned_embedding,
    )

    improvement = (
        cloned_to_target - source_to_target
    )

    moved_toward_target = (
        cloned_to_target > source_to_target
    )

    target_pass = cloned_to_target >= 0.75

    report = {
        "status": "success",
        "metric_type": (
            "OpenVoice internal speaker-embedding cosine similarity"
        ),
        "device": "cpu",
        "converter_load_time_seconds": round(
            converter_load_seconds,
            4,
        ),
        "cloned_embedding_extraction_seconds": round(
            extraction_seconds,
            4,
        ),
        "embedding_shape": list(
            cloned_embedding.shape
        ),
        "source_to_target_cosine": round(
            source_to_target,
            6,
        ),
        "cloned_to_target_cosine": round(
            cloned_to_target,
            6,
        ),
        "source_to_cloned_cosine": round(
            source_to_cloned,
            6,
        ),
        "similarity_improvement": round(
            improvement,
            6,
        ),
        "moved_toward_target": moved_toward_target,
        "target_threshold": 0.75,
        "target_passed": target_pass,
        "source_embedding_sha256": calculate_sha256(
            SOURCE_EMBEDDING_PATH
        ),
        "target_embedding_sha256": calculate_sha256(
            TARGET_EMBEDDING_PATH
        ),
        "cloned_embedding_sha256": calculate_sha256(
            CLONED_EMBEDDING_PATH
        ),
        "note": (
            "This is an OpenVoice internal embedding metric. "
            "An independent speaker-verification model should "
            "also be used for the final benchmark."
        ),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        f"Converter load time: "
        f"{converter_load_seconds:.4f} seconds"
    )
    print(
        f"Cloned embedding extraction: "
        f"{extraction_seconds:.4f} seconds"
    )
    print()
    print(
        f"Source → target cosine: "
        f"{source_to_target:.6f}"
    )
    print(
        f"Cloned → target cosine: "
        f"{cloned_to_target:.6f}"
    )
    print(
        f"Source → cloned cosine: "
        f"{source_to_cloned:.6f}"
    )
    print(
        f"Similarity improvement: "
        f"{improvement:.6f}"
    )
    print(
        f"Moved toward target: "
        f"{moved_toward_target}"
    )
    print(
        f"Passed 0.75 threshold: "
        f"{target_pass}"
    )
    print()
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()


