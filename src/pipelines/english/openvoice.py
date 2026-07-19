import hashlib
import json
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="stft with return_complex=False is deprecated.*",
    category=UserWarning,
)

import numpy as np
import soundfile as sf
import torch
from openvoice.api import ToneColorConverter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "checkpoints_v2" / "converter" / "config.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints_v2" / "converter" / "checkpoint.pth"
SOURCE_WAV = PROJECT_ROOT / "outputs" / "mms" / "english" / "en_01_run_01.wav"
SOURCE_TEXT = "Hello, welcome to Infinia. How can I help you today?"
SOURCE_SE_PATH = PROJECT_ROOT / "outputs" / "openvoice" / "embeddings" / "mms_english_en_01_source_se.pth"
TARGET_SE_PATH = PROJECT_ROOT / "outputs" / "openvoice" / "embeddings" / "ratan_target_se.pth"
OUTPUT_WAV = PROJECT_ROOT / "outputs" / "openvoice" / "english" / "en_01_openvoice_cloned.wav"
REPORT_PATH = PROJECT_ROOT / "evidence" / "result_snapshots" / "openvoice_english_single_conversion.json"
WATERMARK_MESSAGE = "@MyShell"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def audio_stats(path: Path) -> dict:
    data, sample_rate = sf.read(str(path), always_2d=False)
    samples = np.asarray(data, dtype=np.float64)
    if samples.ndim > 1:
        flat = samples.reshape(-1)
    else:
        flat = samples
    duration = len(samples) / float(sample_rate)
    finite = bool(np.isfinite(flat).all())
    rms = float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    clipping_ratio = float(np.mean(np.abs(flat) >= 1.0)) if flat.size else 0.0
    return {
        "path": str(path),
        "sample_rate": int(sample_rate),
        "channels": int(samples.shape[1]) if samples.ndim > 1 else 1,
        "num_samples": int(len(samples)),
        "duration_seconds": duration,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "all_finite": finite,
    }


def tensor_summary(path: Path) -> dict:
    tensor = torch.load(path, map_location="cpu")
    cpu = tensor.detach().cpu()
    return {
        "path": str(path),
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "all_finite": bool(torch.isfinite(cpu).all().item()),
        "all_zero": bool(torch.count_nonzero(cpu).item() == 0),
        "l2_norm": float(torch.linalg.vector_norm(cpu).item()),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    pipeline_start = time.perf_counter()
    print("OpenVoice single English conversion")
    print("=" * 72)
    print(f"Source WAV: {SOURCE_WAV}")
    print(f"Source text: {SOURCE_TEXT}")
    print(f"Target embedding: {TARGET_SE_PATH}")
    print(f"Output WAV: {OUTPUT_WAV}")

    for path in (CONFIG_PATH, CHECKPOINT_PATH, SOURCE_WAV, TARGET_SE_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    source_audio = audio_stats(SOURCE_WAV)
    print(f"Source duration: {source_audio['duration_seconds']:.4f} seconds")

    load_start = time.perf_counter()
    converter = ToneColorConverter(str(CONFIG_PATH), device="cpu")
    converter.load_ckpt(str(CHECKPOINT_PATH))
    converter_load_time = time.perf_counter() - load_start
    print(f"Converter loaded in {converter_load_time:.4f} seconds.")

    SOURCE_SE_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_start = time.perf_counter()
    source_se_returned = converter.extract_se(str(SOURCE_WAV), se_save_path=str(SOURCE_SE_PATH))
    source_embedding_time = time.perf_counter() - source_start
    print("Source embedding extracted successfully.")
    print(f"Source embedding extraction time: {source_embedding_time:.4f} seconds")

    source_se_saved = torch.load(SOURCE_SE_PATH, map_location="cpu")
    require(
        torch.allclose(source_se_returned.detach().cpu(), source_se_saved),
        "Returned source embedding does not match saved source embedding.",
    )

    target_se = torch.load(TARGET_SE_PATH, map_location="cpu")
    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)

    conversion_start = time.perf_counter()
    converter.convert(
        str(SOURCE_WAV),
        source_se_saved,
        target_se,
        output_path=str(OUTPUT_WAV),
        message=WATERMARK_MESSAGE,
    )
    conversion_time = time.perf_counter() - conversion_start
    total_time = time.perf_counter() - pipeline_start
    print("Single English OpenVoice conversion completed.")
    print(f"Conversion time: {conversion_time:.4f} seconds")
    print(f"Total processing time: {total_time:.4f} seconds")

    require(OUTPUT_WAV.exists(), f"Output WAV was not created: {OUTPUT_WAV}")
    output_audio = audio_stats(OUTPUT_WAV)

    require(output_audio["duration_seconds"] > 0, "Output duration is not greater than zero.")
    require(output_audio["rms"] > 0.005, "Output RMS is too low.")
    require(output_audio["peak"] <= 1.0, "Output peak exceeds 1.0.")
    require(output_audio["clipping_ratio"] <= 0.001, "Output clipping ratio exceeds threshold.")
    require(output_audio["all_finite"], "Output contains NaN or infinite samples.")

    conversion_rtf = conversion_time / output_audio["duration_seconds"]
    total_pipeline_rtf = total_time / output_audio["duration_seconds"]

    hashes = {
        "source_wav_sha256": sha256_file(SOURCE_WAV),
        "source_embedding_sha256": sha256_file(SOURCE_SE_PATH),
        "target_embedding_sha256": sha256_file(TARGET_SE_PATH),
        "cloned_output_sha256": sha256_file(OUTPUT_WAV),
    }

    report = {
        "status": "success",
        "language": "english",
        "sentence_id": "en_01",
        "source_text": SOURCE_TEXT,
        "device": "cpu",
        "watermark_message": WATERMARK_MESSAGE,
        "config_path": str(CONFIG_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "source_embedding_path": str(SOURCE_SE_PATH),
        "target_embedding_path": str(TARGET_SE_PATH),
        "output_wav": str(OUTPUT_WAV),
        "converter_load_time_seconds": round(converter_load_time, 4),
        "source_embedding_extraction_time_seconds": round(source_embedding_time, 4),
        "conversion_time_seconds": round(conversion_time, 4),
        "total_processing_time_seconds": round(total_time, 4),
        "source_audio_duration_seconds": round(source_audio["duration_seconds"], 6),
        "output_audio_duration_seconds": round(output_audio["duration_seconds"], 6),
        "conversion_rtf": round(conversion_rtf, 6),
        "total_pipeline_rtf_excluding_mms_generation": round(total_pipeline_rtf, 6),
        "source_audio": source_audio,
        "output_audio": output_audio,
        "source_embedding": tensor_summary(SOURCE_SE_PATH),
        "target_embedding": tensor_summary(TARGET_SE_PATH),
        **hashes,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Signal validation passed.")
    print(f"Output duration: {output_audio['duration_seconds']:.4f} seconds")
    print(f"Output RMS: {output_audio['rms']:.8f}")
    print(f"Output peak: {output_audio['peak']:.8f}")
    print(f"Output clipping ratio: {output_audio['clipping_ratio']:.8f}")
    print(f"Conversion RTF: {conversion_rtf:.6f}")
    print(f"Total pipeline RTF excluding MMS generation: {total_pipeline_rtf:.6f}")
    print(f"Source WAV SHA256: {hashes['source_wav_sha256']}")
    print(f"Source embedding SHA256: {hashes['source_embedding_sha256']}")
    print(f"Target embedding SHA256: {hashes['target_embedding_sha256']}")
    print(f"Cloned output SHA256: {hashes['cloned_output_sha256']}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
