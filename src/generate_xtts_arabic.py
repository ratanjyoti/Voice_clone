from __future__ import annotations

import argparse
import csv
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
CACHE_DIR = PROJECT_ROOT / "models" / "coqui"
TESTS_PATH = PROJECT_ROOT / "data" / "test_sentences" / "arabic_voice_clone_tests.json"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_audio" / "arabic" / "professional_msa"
REFERENCES = {
    "short": REFERENCE_DIR / "arabic_reference_short.wav",
    "standard": REFERENCE_DIR / "arabic_reference_standard.wav",
    "long": REFERENCE_DIR / "arabic_reference_long.wav",
}
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xtts" / "arabic"
RESULTS_DIR = PROJECT_ROOT / "results"
SNAPSHOT_DIR = PROJECT_ROOT / "evidence" / "result_snapshots" / "xtts" / "arabic"
SUPPORTED_PARAM_KEYS = ["temperature", "top_k", "top_p", "repetition_penalty", "length_penalty", "speed"]
DEFAULT_PARAMETERS = {
    "temperature": 0.75,
    "top_k": 50,
    "top_p": 0.85,
    "repetition_penalty": 2.0,
    "length_penalty": 1.0,
    "speed": 1.0,
}
PARAMETER_CONFIGS = [
    {"configuration_id": "A", "temperature": 0.65, "top_p": 0.85, "top_k": 50, "repetition_penalty": 2.0, "speed": 1.0},
    {"configuration_id": "B", "temperature": 0.75, "top_p": 0.85, "top_k": 50, "repetition_penalty": 2.0, "speed": 1.0},
    {"configuration_id": "C", "temperature": 0.65, "top_p": 0.80, "top_k": 40, "repetition_penalty": 5.0, "speed": 1.0},
    {"configuration_id": "D", "installed_defaults": True},
]

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


def load_tests() -> list[dict[str, Any]]:
    return json.loads(TESTS_PATH.read_text(encoding="utf-8"))["tests"]


def reference_paths(strategy: str) -> list[Path]:
    if strategy == "multi":
        return [REFERENCES["short"], REFERENCES["standard"], REFERENCES["long"]]
    return [REFERENCES[strategy]]


def to_mono_float32(waveform: Any):
    import numpy as np

    arr = np.asarray(waveform, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected mono 1D waveform, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Generated waveform contains NaN or infinite values")
    return arr.astype(np.float32)


def normalize_peak(waveform, target_peak: float = 0.95):
    import numpy as np

    raw_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if raw_peak > 0:
        waveform = waveform * (target_peak / raw_peak)
    return waveform.astype(np.float32), raw_peak


def audio_stats(waveform, sample_rate: int) -> dict[str, Any]:
    import numpy as np

    abs_wave = np.abs(waveform)
    return {
        "duration": float(len(waveform) / sample_rate) if sample_rate else 0.0,
        "sample_rate": int(sample_rate),
        "channels": 1,
        "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) if waveform.size else 0.0,
        "peak": float(abs_wave.max()) if waveform.size else 0.0,
        "clipping_count": int(np.sum(abs_wave >= 0.999)) if waveform.size else 0,
        "silence_ratio": float(np.mean(abs_wave < 0.001)) if waveform.size else 1.0,
    }


def read_saved_stats(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    return audio_stats(mono, int(sr))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "speed": args.speed,
    }


def call_kwargs(parameters: dict[str, Any]) -> dict[str, Any]:
    kwargs = {}
    for key in SUPPORTED_PARAM_KEYS:
        value = parameters.get(key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def generate_one(
    api: Any,
    test: dict[str, Any],
    output_path: Path,
    reference_strategy: str,
    parameters: dict[str, Any],
    seed: int,
    model_load_seconds: float,
    device: str,
    configuration_id: str = "",
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    import TTS as tts_pkg

    refs = reference_paths(reference_strategy)
    sidecar_path = output_path.with_suffix(".json")
    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_version": getattr(tts_pkg, "__version__", ""),
        "test_id": test.get("id"),
        "category": test.get("category", ""),
        "text": test.get("text", ""),
        "language": LANGUAGE,
        "device": device,
        "reference_strategy": reference_strategy,
        "reference_paths": json.dumps([str(p) for p in refs], ensure_ascii=False),
        "configuration_id": configuration_id,
        "seed": seed,
        "model_load_time": model_load_seconds,
        "output_path": str(output_path),
        "status": "failed",
        "error": "",
    }
    for key, value in parameters.items():
        row[key] = value

    try:
        for ref in refs:
            if not ref.exists():
                raise FileNotFoundError(ref)
        torch.manual_seed(seed)
        np.random.seed(seed)
        sample_rate = int(getattr(getattr(api, "synthesizer", None), "output_sample_rate", 24000) or 24000)
        speaker_wav: str | list[str] = str(refs[0]) if len(refs) == 1 else [str(p) for p in refs]
        kwargs = call_kwargs(parameters)
        generation_start = time.perf_counter()
        generated = api.tts(
            text=test["text"],
            speaker_wav=speaker_wav,
            language=LANGUAGE,
            split_sentences=False,
            **kwargs,
        )
        generation_seconds = time.perf_counter() - generation_start
        waveform = to_mono_float32(generated)
        waveform, raw_peak = normalize_peak(waveform)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")
        saved_stats = read_saved_stats(output_path)
        rtf = generation_seconds / saved_stats["duration"] if saved_stats["duration"] > 0 else math.nan
        digest = sha256_file(output_path)
        row.update(
            {
                "status": "success",
                "generation_time": generation_seconds,
                "duration": saved_stats["duration"],
                "rtf": rtf,
                "sample_rate": saved_stats["sample_rate"],
                "channels": saved_stats["channels"],
                "rms": saved_stats["rms"],
                "raw_peak": raw_peak,
                "normalized_peak": float(audio_stats(waveform, sample_rate)["peak"]),
                "peak": saved_stats["peak"],
                "clipping_count": saved_stats["clipping_count"],
                "silence_ratio": saved_stats["silence_ratio"],
                "sha256": digest,
            }
        )
        print(f"Saved {output_path} | gen {generation_seconds:.3f}s | dur {saved_stats['duration']:.3f}s | RTF {rtf:.3f}")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        print(f"FAILED {output_path}: {row['error']}")
    finally:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def load_model(device: str):
    import torch
    from TTS.api import TTS

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    gpu = device == "cuda"
    print(f"Loading XTTS-v2 on {device} from cache {CACHE_DIR}")
    start = time.perf_counter()
    api = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=gpu)
    load_seconds = time.perf_counter() - start
    print(f"Model loaded in {load_seconds:.3f}s")
    return api, load_seconds


def run_reference_comparison(args: argparse.Namespace) -> None:
    tests = load_tests()
    greeting = tests[0]
    api, load_seconds = load_model(args.device)
    params = params_from_args(args)
    rows = []
    for strategy in ["short", "standard", "long", "multi"]:
        output_path = OUTPUT_DIR / "reference_tests" / f"ar_greeting_{strategy}.wav"
        print(f"Reference strategy: {strategy}")
        rows.append(generate_one(api, greeting, output_path, strategy, params, args.seed, load_seconds, args.device))
    write_csv(RESULTS_DIR / "xtts_arabic_reference_generation.csv", rows)


def run_parameter_tests(args: argparse.Namespace) -> None:
    tests = load_tests()
    greeting = tests[0]
    api, load_seconds = load_model(args.device)
    rows = []
    for config in PARAMETER_CONFIGS:
        config_id = config["configuration_id"]
        params = {} if config.get("installed_defaults") else dict(DEFAULT_PARAMETERS)
        params.update({k: v for k, v in config.items() if k in SUPPORTED_PARAM_KEYS})
        output_path = OUTPUT_DIR / "parameter_tests" / f"ar_greeting_config_{config_id}.wav"
        print(f"Parameter config {config_id}: {params if params else 'installed defaults'}")
        row = generate_one(api, greeting, output_path, args.reference_strategy, params, args.seed, load_seconds, args.device, config_id)
        row["installed_defaults"] = bool(config.get("installed_defaults", False))
        rows.append(row)
    write_csv(RESULTS_DIR / "xtts_arabic_parameter_tests.csv", rows)


def run_final(args: argparse.Namespace) -> None:
    tests = load_tests()
    selected = tests if args.test_id == "all" else [t for t in tests if t["id"] == args.test_id]
    if not selected:
        raise ValueError(f"No test found for {args.test_id}")
    api, load_seconds = load_model(args.device)
    params = params_from_args(args)
    rows = []
    for test in selected:
        output_path = OUTPUT_DIR / f"{test['id']}.wav"
        print(f"Final generation: {test['id']}")
        rows.append(generate_one(api, test, output_path, args.reference_strategy, params, args.seed, load_seconds, args.device, args.configuration_id))
    write_csv(RESULTS_DIR / "xtts_arabic_generation.csv", rows)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "final",
        "total": len(rows),
        "successful": sum(1 for row in rows if row.get("status") == "success"),
        "failed": sum(1 for row in rows if row.get("status") != "success"),
        "reference_strategy": args.reference_strategy,
        "reference_paths": [str(p) for p in reference_paths(args.reference_strategy)],
        "parameters": params,
        "seed": args.seed,
        "rows": rows,
    }
    (SNAPSHOT_DIR / "xtts_arabic_generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Arabic XTTS-v2 zero-shot samples.")
    parser.add_argument("--mode", choices=["reference-comparison", "parameter-tests", "final"], default="final")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reference-strategy", choices=["short", "standard", "long", "multi"], default="standard")
    parser.add_argument("--test-id", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configuration-id", default="selected")
    parser.add_argument("--temperature", type=float, default=DEFAULT_PARAMETERS["temperature"])
    parser.add_argument("--top-k", dest="top_k", type=int, default=DEFAULT_PARAMETERS["top_k"])
    parser.add_argument("--top-p", dest="top_p", type=float, default=DEFAULT_PARAMETERS["top_p"])
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=DEFAULT_PARAMETERS["repetition_penalty"])
    parser.add_argument("--length-penalty", dest="length_penalty", type=float, default=DEFAULT_PARAMETERS["length_penalty"])
    parser.add_argument("--speed", type=float, default=DEFAULT_PARAMETERS["speed"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "reference-comparison":
        run_reference_comparison(args)
    elif args.mode == "parameter-tests":
        run_parameter_tests(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
