import hashlib
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="stft with return_complex=False is deprecated.*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated.*",
    category=UserWarning,
)

import torch
from openvoice.api import ToneColorConverter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "checkpoints_v2" / "converter" / "config.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints_v2" / "converter" / "checkpoint.pth"
REFERENCE_WAV = PROJECT_ROOT / "data" / "reference_audio" / "ratan_reference_22050_mono.wav"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "openvoice" / "embeddings" / "ratan_target_se.pth"
REPORT_PATH = PROJECT_ROOT / "evidence" / "result_snapshots" / "openvoice_target_embedding.json"

EXPECTED_REFERENCE_SHA256 = "58FCC93C94D09471712F5D34A7A8E596C2F6B653A68A72283B95A75F81DE8C4C"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def summarize_tensor(tensor: torch.Tensor) -> dict:
    cpu_tensor = tensor.detach().cpu()
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "min": float(cpu_tensor.min().item()),
        "max": float(cpu_tensor.max().item()),
        "mean": float(cpu_tensor.mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(cpu_tensor).item()),
        "all_finite": bool(torch.isfinite(cpu_tensor).all().item()),
        "all_zero": bool(torch.count_nonzero(cpu_tensor).item() == 0),
    }


def main() -> None:
    print("OpenVoice target speaker embedding extraction")
    print("=" * 72)
    print(f"Config: {CONFIG_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Reference WAV: {REFERENCE_WAV}")
    print(f"Output embedding: {OUTPUT_PATH}")

    for path in (CONFIG_PATH, CHECKPOINT_PATH, REFERENCE_WAV):
        if not path.exists():
            raise FileNotFoundError(path)

    reference_sha256 = sha256_file(REFERENCE_WAV)
    print(f"Reference SHA256: {reference_sha256}")
    if reference_sha256 != EXPECTED_REFERENCE_SHA256:
        raise ValueError(
            "Reference WAV SHA256 does not match the validated recording."
        )

    load_start = time.perf_counter()
    converter = ToneColorConverter(str(CONFIG_PATH), device="cpu")
    converter.load_ckpt(str(CHECKPOINT_PATH))
    load_time = time.perf_counter() - load_start
    print(f"Converter loaded in {load_time:.4f} seconds.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    extraction_start = time.perf_counter()
    returned_embedding = converter.extract_se(
        str(REFERENCE_WAV),
        se_save_path=str(OUTPUT_PATH),
    )
    extraction_time = time.perf_counter() - extraction_start

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Embedding was not saved: {OUTPUT_PATH}")

    saved_embedding = torch.load(OUTPUT_PATH, map_location="cpu")
    returned_cpu = returned_embedding.detach().cpu()
    returned_and_saved_match = bool(torch.allclose(returned_cpu, saved_embedding))

    summary = summarize_tensor(saved_embedding)
    embedding_sha256 = sha256_file(OUTPUT_PATH)
    embedding_size = OUTPUT_PATH.stat().st_size

    if not summary["all_finite"]:
        raise ValueError("Embedding contains non-finite values.")
    if summary["all_zero"]:
        raise ValueError("Embedding is all zero.")
    if summary["l2_norm"] <= 0:
        raise ValueError("Embedding norm is not greater than zero.")
    if not returned_and_saved_match:
        raise ValueError("Returned embedding does not match saved embedding.")
    if embedding_size <= 0:
        raise ValueError("Embedding file is empty.")

    report = {
        "status": "success",
        "device": "cpu",
        "config_path": str(CONFIG_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "reference_wav": str(REFERENCE_WAV),
        "reference_sha256": reference_sha256,
        "embedding_path": str(OUTPUT_PATH),
        "embedding_sha256": embedding_sha256,
        "embedding_size_bytes": embedding_size,
        "converter_load_time_seconds": round(load_time, 4),
        "embedding_extraction_time_seconds": round(extraction_time, 4),
        "returned_and_saved_match": returned_and_saved_match,
        **summary,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Target embedding extracted successfully.")
    print(f"Extraction time: {extraction_time:.4f} seconds")
    print(f"Embedding shape: {summary['shape']}")
    print(f"Embedding dtype: {summary['dtype']}")
    print(f"Embedding min: {summary['min']:.8f}")
    print(f"Embedding max: {summary['max']:.8f}")
    print(f"Embedding mean: {summary['mean']:.8f}")
    print(f"Embedding norm: {summary['l2_norm']:.8f}")
    print(f"All finite: {summary['all_finite']}")
    print(f"All zero: {summary['all_zero']}")
    print(f"Returned/saved match: {returned_and_saved_match}")
    print(f"SHA256: {embedding_sha256}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

