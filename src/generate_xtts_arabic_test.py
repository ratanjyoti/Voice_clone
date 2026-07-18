from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "ar"
DEVICE = "cpu"
SEED = 42
CACHE_DIR = PROJECT_ROOT / "models" / "coqui"
TESTS_PATH = PROJECT_ROOT / "data" / "test_sentences" / "arabic_voice_clone_tests.json"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_audio" / "arabic" / "professional_msa" / "arabic_reference_standard.wav"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "xtts" / "arabic" / "ar_clone_01_greeting.wav"
SIDECAR_PATH = OUTPUT_PATH.with_suffix(".json")

os.environ.setdefault("TTS_HOME", str(CACHE_DIR))
os.environ.setdefault("COQUI_TOS_AGREED", "1")
for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
    os.environ[proxy_name] = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_stats(waveform, sample_rate: int) -> dict[str, Any]:
    import numpy as np

    abs_wave = np.abs(waveform)
    duration = float(len(waveform) / sample_rate) if sample_rate else 0.0
    return {
        "duration": duration,
        "sample_rate": int(sample_rate),
        "channels": 1,
        "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) if waveform.size else 0.0,
        "peak": float(abs_wave.max()) if waveform.size else 0.0,
        "clipping_count": int(np.sum(abs_wave >= 0.999)) if waveform.size else 0,
        "silence_ratio": float(np.mean(abs_wave < 0.001)) if waveform.size else 1.0,
    }


def normalize_waveform(waveform, target_peak: float = 0.95):
    import numpy as np

    waveform = np.asarray(waveform, dtype=np.float32).squeeze()
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1).astype(np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform after squeeze/mono conversion, got shape {waveform.shape}")
    if not np.isfinite(waveform).all():
        raise ValueError("Generated waveform contains NaN or infinite samples")
    raw_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if raw_peak > 0:
        waveform = waveform * (target_peak / raw_peak)
    return waveform.astype(np.float32), raw_peak


def load_greeting() -> dict[str, Any]:
    data = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
    return data["tests"][0]


def write_sidecar(payload: dict[str, Any]) -> None:
    SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch
    import TTS as tts_pkg
    from TTS.api import TTS

    test = load_greeting()
    sidecar: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_version": getattr(tts_pkg, "__version__", None),
        "language": LANGUAGE,
        "device": DEVICE,
        "test_id": test.get("id"),
        "category": test.get("category"),
        "text": test.get("text"),
        "reference_file": str(REFERENCE_PATH),
        "seed": SEED,
        "output_path": str(OUTPUT_PATH),
        "status": "failed",
        "error": "",
    }

    try:
        if not REFERENCE_PATH.exists():
            raise FileNotFoundError(REFERENCE_PATH)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        print(f"Loading XTTS model: {MODEL_NAME}")
        load_start = time.perf_counter()
        api = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
        model_load_seconds = time.perf_counter() - load_start
        sidecar["model_load_time"] = model_load_seconds
        sample_rate = int(getattr(getattr(api, "synthesizer", None), "output_sample_rate", 24000) or 24000)
        print(f"Model loaded in {model_load_seconds:.3f}s")
        print(f"Generating {test['id']} from {REFERENCE_PATH}")

        generation_start = time.perf_counter()
        generated = api.tts(
            text=test["text"],
            speaker_wav=str(REFERENCE_PATH),
            language=LANGUAGE,
            split_sentences=False,
        )
        generation_seconds = time.perf_counter() - generation_start

        waveform, raw_peak = normalize_waveform(generated)
        stats = audio_stats(waveform, sample_rate)
        if stats["duration"] <= 0:
            raise ValueError("Generated waveform has zero duration")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(OUTPUT_PATH), waveform, sample_rate, subtype="PCM_16")
        saved_audio, saved_sr = sf.read(str(OUTPUT_PATH), always_2d=True)
        saved_mono = saved_audio.mean(axis=1).astype(np.float32)
        saved_stats = audio_stats(saved_mono, int(saved_sr))
        sha256 = sha256_file(OUTPUT_PATH)
        rtf = generation_seconds / saved_stats["duration"] if saved_stats["duration"] > 0 else math.nan

        sidecar.update(
            {
                "status": "success",
                "model_load_time": model_load_seconds,
                "generation_time": generation_seconds,
                "duration": saved_stats["duration"],
                "rtf": rtf,
                "sample_rate": saved_stats["sample_rate"],
                "channels": saved_stats["channels"],
                "rms": saved_stats["rms"],
                "raw_peak": raw_peak,
                "normalized_peak": stats["peak"],
                "saved_peak": saved_stats["peak"],
                "clipping_count": saved_stats["clipping_count"],
                "silence_ratio": saved_stats["silence_ratio"],
                "sha256": sha256,
            }
        )
        print(f"Saved: {OUTPUT_PATH}")
        print(f"Duration: {saved_stats['duration']:.3f}s")
        print(f"Generation time: {generation_seconds:.3f}s")
        print(f"RTF: {rtf:.3f}")
        print(f"RMS: {saved_stats['rms']:.6f}")
        print(f"Peak: {saved_stats['peak']:.6f}")
        print(f"Clipping count: {saved_stats['clipping_count']}")
        print(f"Silence ratio: {saved_stats['silence_ratio']:.6f}")
        print(f"SHA256: {sha256}")
        return 0
    except Exception as exc:
        sidecar["status"] = "failed"
        sidecar["error"] = f"{type(exc).__name__}: {exc}"
        sidecar["traceback"] = traceback.format_exc()
        print("XTTS initial generation: FAIL")
        print(sidecar["error"])
        print(sidecar["traceback"])
        return 1
    finally:
        write_sidecar(sidecar)


if __name__ == "__main__":
    raise SystemExit(main())
