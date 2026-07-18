from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel


MODEL_ID = "ai4bharat/IndicF5"
DEVICE = "cpu"
TEST_ID = "hi_clone_01_greeting"
TESTS_PATH = PROJECT_ROOT / "data" / "test_sentences" / "hindi_voice_clone_tests.json"
REFERENCE_AUDIO = (
    PROJECT_ROOT / "data" / "reference_audio" / "indicf5" / "ratan_reference_indicf5.wav"
)
REFERENCE_TRANSCRIPT = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "indicf5"
    / "ratan_reference_transcript.txt"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "indicf5" / "hindi"
OUTPUT_WAV = OUTPUT_DIR / f"{TEST_ID}.wav"
OUTPUT_JSON = OUTPUT_DIR / f"{TEST_ID}.json"
TARGET_PEAK = 0.95
FALLBACK_SAMPLE_RATE = 24000


class EagerCompiledModule(torch.nn.Module):
    """CPU stand-in for torch.compile that preserves compiled state-dict keys."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._orig_mod(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("_orig_mod"), name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_greeting() -> dict[str, str]:
    tests = json.loads(TESTS_PATH.read_text(encoding="utf-8"))["tests"]
    for test in tests:
        if test["id"] == TEST_ID:
            return test
    raise KeyError(f"Missing benchmark test id: {TEST_ID}")


def to_numpy_audio(output: Any) -> tuple[np.ndarray, int]:
    sample_rate = FALLBACK_SAMPLE_RATE
    audio = output

    if isinstance(output, dict):
        audio = (
            output.get("audio")
            or output.get("waveform")
            or output.get("samples")
            or output.get("wav")
        )
        sample_rate = int(output.get("sample_rate") or output.get("sampling_rate") or sample_rate)
    elif isinstance(output, (tuple, list)) and output:
        audio = output[0]
        if len(output) > 1 and isinstance(output[1], (int, float)):
            sample_rate = int(output[1])

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()

    waveform = np.asarray(audio, dtype=np.float32).squeeze()
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono/1D audio after squeeze, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("IndicF5 returned empty audio")
    if not np.isfinite(waveform).all():
        raise ValueError("IndicF5 returned NaN or infinite samples")

    return waveform, sample_rate


def safe_normalize(waveform: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    raw_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if raw_peak > TARGET_PEAK:
        waveform = waveform * (TARGET_PEAK / raw_peak)
        normalized = True
    else:
        normalized = False
    saved_peak_target = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    return waveform.astype(np.float32), raw_peak, saved_peak_target, normalized


def saved_audio_stats(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_seconds": duration,
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0,
        "saved_peak": float(abs_audio.max()) if mono.size else 0.0,
        "clipping_count": int(np.sum(abs_audio >= 0.999)) if mono.size else 0,
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if mono.size else 1.0,
        "all_finite": bool(np.isfinite(mono).all()),
        "sha256": sha256(path),
    }


def main() -> None:
    if not REFERENCE_AUDIO.exists():
        raise FileNotFoundError(f"Missing reference audio: {REFERENCE_AUDIO}")
    if not REFERENCE_TRANSCRIPT.exists():
        raise FileNotFoundError(f"Missing reference transcript: {REFERENCE_TRANSCRIPT}")

    torch.manual_seed(20260718)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test = load_greeting()
    ref_text = REFERENCE_TRANSCRIPT.read_text(encoding="utf-8", errors="replace").strip()
    if not ref_text or ref_text.startswith("PLACEHOLDER"):
        raise RuntimeError("Exact reference transcript is missing or still a placeholder.")

    print("=" * 72)
    print("IndicF5 Hindi one-sample generation")
    print("=" * 72)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Test: {test['id']}")
    print(f"Text: {test['text']}")
    print(f"Reference audio: {REFERENCE_AUDIO}")
    print(f"Reference transcript: {REFERENCE_TRANSCRIPT}")

    load_start = time.perf_counter()
    print("Loading IndicF5 model...")
    print("CPU mode: disabling torch.compile during model construction")

    original_torch_compile = getattr(torch, "compile", None)

    try:
        if DEVICE == "cpu" and original_torch_compile is not None:
            def eager_compile(module, *args, **kwargs):
                print(
                    "Using eager torch.compile stand-in for:",
                    module.__class__.__name__,
                )
                return EagerCompiledModule(module)

            torch.compile = eager_compile

        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )

    finally:
        if original_torch_compile is not None:
            torch.compile = original_torch_compile

    print("IndicF5 model loaded successfully")
    if hasattr(model, "to"):
        model = model.to(DEVICE)
    if hasattr(model, "eval"):
        model.eval()
    model_load_time = time.perf_counter() - load_start
    print(f"Model loaded in {model_load_time:.3f}s")

    reference_processing_start = time.perf_counter()
    ref_audio_path = str(REFERENCE_AUDIO)
    reference_processing_time = time.perf_counter() - reference_processing_start

    generation_start = time.perf_counter()
    with torch.inference_mode():
        generated = model(
            test["text"],
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )
    generation_time = time.perf_counter() - generation_start

    waveform, sample_rate = to_numpy_audio(generated)
    waveform, raw_peak, normalized_peak, normalized = safe_normalize(waveform)
    if not np.isfinite(waveform).all():
        raise ValueError("Normalized waveform contains NaN or infinity")

    sf.write(str(OUTPUT_WAV), waveform, sample_rate, subtype="PCM_16")
    stats = saved_audio_stats(OUTPUT_WAV)
    if not stats["all_finite"]:
        raise ValueError("Saved WAV contains non-finite samples")

    rtf = generation_time / stats["duration_seconds"] if stats["duration_seconds"] else math.inf
    sidecar = {
        "status": "success",
        "model": "IndicF5",
        "model_id": MODEL_ID,
        "device": DEVICE,
        "test_id": test["id"],
        "category": test["category"],
        "expected_text": test["text"],
        "reference_audio": str(REFERENCE_AUDIO.relative_to(PROJECT_ROOT)),
        "reference_transcript": str(REFERENCE_TRANSCRIPT.relative_to(PROJECT_ROOT)),
        "reference_text": ref_text,
        "model_load_time": model_load_time,
        "reference_processing_time": reference_processing_time,
        "generation_time": generation_time,
        "audio_duration": stats["duration_seconds"],
        "rtf": rtf,
        "raw_peak": raw_peak,
        "normalized_peak": normalized_peak,
        "normalized": normalized,
        "output_path": str(OUTPUT_WAV.relative_to(PROJECT_ROOT)),
        **stats,
    }
    OUTPUT_JSON.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Generation completed.")
    print(f"WAV: {OUTPUT_WAV}")
    print(f"Sidecar: {OUTPUT_JSON}")
    print(f"Generation time: {generation_time:.3f}s")
    print(f"Duration: {stats['duration_seconds']:.3f}s")
    print(f"RTF: {rtf:.3f}")
    print(f"Peak: {stats['saved_peak']:.6f}")
    print(f"Clipping count: {stats['clipping_count']}")


if __name__ == "__main__":
    main()


