import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_AUDIO = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "ratan_reference_22050_mono.wav"
)

MMS_AUDIO = (
    PROJECT_ROOT
    / "outputs"
    / "mms"
    / "english"
    / "en_01_run_01.wav"
)

CLONED_AUDIO = (
    PROJECT_ROOT
    / "outputs"
    / "openvoice"
    / "english"
    / "en_01_openvoice_cloned.wav"
)

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

MODEL_CACHE = (
    PROJECT_ROOT
    / "models"
    / "speechbrain"
    / "spkrec-ecapa-voxceleb"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "ecapa_english_speaker_similarity.json"
)


def verify_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty: {path}"
        )


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest().upper()


def load_audio(
    path: Path,
    target_sample_rate: int = 16000,
) -> tuple[torch.Tensor, dict]:
    waveform, sample_rate = torchaudio.load(
        str(path)
    )

    original_channels = waveform.shape[0]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    original_sample_rate = sample_rate

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )

        sample_rate = target_sample_rate

    if waveform.numel() == 0:
        raise ValueError(
            f"Audio contains no samples: {path}"
        )

    if not torch.isfinite(waveform).all():
        raise ValueError(
            f"Audio contains NaN or infinity: {path}"
        )

    peak = float(
        waveform.abs().max().item()
    )

    rms = float(
        torch.sqrt(
            torch.mean(
                waveform.square()
            )
        ).item()
    )

    duration = (
        waveform.shape[-1] / sample_rate
    )

    return waveform, {
        "path": str(path),
        "original_sample_rate_hz": (
            original_sample_rate
        ),
        "evaluation_sample_rate_hz": sample_rate,
        "original_channels": original_channels,
        "evaluation_channels": 1,
        "duration_seconds": round(
            duration,
            6,
        ),
        "rms": round(rms, 8),
        "peak": round(peak, 8),
        "sha256": calculate_sha256(path),
    }


def extract_embedding(
    classifier: EncoderClassifier,
    waveform: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        embedding = classifier.encode_batch(
            waveform
        )

    embedding = (
        embedding
        .detach()
        .float()
        .cpu()
        .reshape(1, -1)
    )

    if not torch.isfinite(embedding).all():
        raise ValueError(
            "Embedding contains NaN or infinity."
        )

    norm = float(
        torch.linalg.vector_norm(
            embedding
        ).item()
    )

    if norm <= 0:
        raise ValueError(
            "Embedding has zero norm."
        )

    return embedding


def cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    score = F.cosine_similarity(
        first,
        second,
        dim=1,
    )

    return float(score.item())


def main() -> None:
    required_files = (
        (REFERENCE_AUDIO, "Reference audio"),
        (MMS_AUDIO, "Original MMS audio"),
        (CLONED_AUDIO, "OpenVoice cloned audio"),
    )

    for path, label in required_files:
        verify_file(path, label)

    MODEL_CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("Independent ECAPA speaker-similarity evaluation")
    print("=" * 78)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchAudio: {torchaudio.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model: {MODEL_SOURCE}")
    print("Device: CPU")
    print()

    model_load_start = time.perf_counter()

    classifier = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=str(MODEL_CACHE),
        run_opts={
            "device": "cpu",
        },
    )

    model_load_seconds = (
        time.perf_counter()
        - model_load_start
    )

    print(
        f"ECAPA model loaded in "
        f"{model_load_seconds:.4f} seconds."
    )

    reference_waveform, reference_audio_info = (
        load_audio(REFERENCE_AUDIO)
    )

    mms_waveform, mms_audio_info = (
        load_audio(MMS_AUDIO)
    )

    cloned_waveform, cloned_audio_info = (
        load_audio(CLONED_AUDIO)
    )

    extraction_start = time.perf_counter()

    reference_embedding = extract_embedding(
        classifier,
        reference_waveform,
    )

    mms_embedding = extract_embedding(
        classifier,
        mms_waveform,
    )

    cloned_embedding = extract_embedding(
        classifier,
        cloned_waveform,
    )

    extraction_seconds = (
        time.perf_counter()
        - extraction_start
    )

    reference_to_mms = cosine_similarity(
        reference_embedding,
        mms_embedding,
    )

    reference_to_cloned = cosine_similarity(
        reference_embedding,
        cloned_embedding,
    )

    mms_to_cloned = cosine_similarity(
        mms_embedding,
        cloned_embedding,
    )

    improvement = (
        reference_to_cloned
        - reference_to_mms
    )

    moved_toward_reference = (
        reference_to_cloned
        > reference_to_mms
    )

    threshold = 0.75

    report = {
        "status": "success",
        "metric_type": (
            "Independent SpeechBrain ECAPA-TDNN "
            "cosine similarity"
        ),
        "model_source": MODEL_SOURCE,
        "model_cache": str(MODEL_CACHE),
        "device": "cpu",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchaudio_version": (
            torchaudio.__version__
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "model_load_time_seconds": round(
            model_load_seconds,
            4,
        ),
        "embedding_extraction_time_seconds": round(
            extraction_seconds,
            4,
        ),
        "embedding_shape": list(
            reference_embedding.shape
        ),
        "audio": {
            "reference": reference_audio_info,
            "original_mms": mms_audio_info,
            "openvoice_cloned": cloned_audio_info,
        },
        "scores": {
            "reference_to_original_mms": round(
                reference_to_mms,
                6,
            ),
            "reference_to_openvoice_cloned": round(
                reference_to_cloned,
                6,
            ),
            "original_mms_to_openvoice_cloned": round(
                mms_to_cloned,
                6,
            ),
            "improvement_over_original_mms": round(
                improvement,
                6,
            ),
        },
        "moved_toward_reference": (
            moved_toward_reference
        ),
        "requested_threshold": threshold,
        "threshold_passed": (
            reference_to_cloned >= threshold
        ),
        "important_note": (
            "A fixed cosine threshold is not universally "
            "calibrated across all speaker-verification "
            "models and recording conditions. Human A/B "
            "judgment remains required."
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

    print()
    print(
        "Reference → original MMS: "
        f"{reference_to_mms:.6f}"
    )
    print(
        "Reference → OpenVoice cloned: "
        f"{reference_to_cloned:.6f}"
    )
    print(
        "Original MMS → OpenVoice cloned: "
        f"{mms_to_cloned:.6f}"
    )
    print(
        "Improvement over original MMS: "
        f"{improvement:.6f}"
    )
    print(
        "Moved toward reference: "
        f"{moved_toward_reference}"
    )
    print(
        f"Passed requested 0.75 threshold: "
        f"{reference_to_cloned >= threshold}"
    )
    print()
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
    