from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR.parent))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "300")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODEL_NAME = "ResembleAI/chatterbox multilingual"
LANGUAGE_ID = "ar"
TESTS_PATH = PROJECT_ROOT / "data" / "test_sentences" / "arabic_voice_clone_tests.json"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_audio" / "arabic" / "professional_msa"
REFERENCES = {
    "short": REFERENCE_DIR / "arabic_reference_short.wav",
    "standard": REFERENCE_DIR / "arabic_reference_standard.wav",
    "long": REFERENCE_DIR / "arabic_reference_long.wav",
}
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chatterbox" / "arabic"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_PARAMETERS = {
    "temperature": 0.7,
    "cfg_weight": 0.5,
    "exaggeration": 0.5,
    "repetition_penalty": 2.0,
    "min_p": 0.05,
    "top_p": 1.0,
}
PARAMETER_MATRIX = [
    {"temperature": 0.6, "cfg_weight": 0.5},
    {"temperature": 0.7, "cfg_weight": 0.5},
    {"temperature": 0.8, "cfg_weight": 0.5},
    {"temperature": 0.7, "cfg_weight": 0.4},
    {"temperature": 0.7, "cfg_weight": 0.6},
]


def load_tests() -> list[dict[str, Any]]:
    data = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
    return data["tests"]


def to_numpy_mono(waveform: Any) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().float().squeeze().numpy()
    waveform = np.asarray(waveform, dtype=np.float32).squeeze()
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1).astype(np.float32)
    return waveform.astype(np.float32)


def audio_stats(waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if waveform.size == 0:
        return {
            "duration": 0.0,
            "sample_rate": int(sample_rate),
            "rms": 0.0,
            "peak": 0.0,
            "clipping_count": 0,
            "silence_ratio": 1.0,
        }
    abs_wave = np.abs(waveform)
    return {
        "duration": float(len(waveform) / sample_rate),
        "sample_rate": int(sample_rate),
        "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
        "peak": float(abs_wave.max()),
        "clipping_count": int(np.sum(abs_wave >= 0.999)),
        "silence_ratio": float(np.mean(abs_wave < 0.001)),
    }


def normalize_peak(waveform: np.ndarray, target_peak: float = 0.95) -> tuple[np.ndarray, float]:
    raw_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if raw_peak > 0:
        waveform = waveform * (target_peak / raw_peak)
    return waveform.astype(np.float32), raw_peak


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_one(
    model: ChatterboxMultilingualTTS,
    text: str,
    output_path: Path,
    reference_path: Path,
    parameters: dict[str, Any],
    seed: int,
    test_id: str,
    reference_name: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    start = time.perf_counter()
    waveform = model.generate(
        text=text,
        language_id=LANGUAGE_ID,
        audio_prompt_path=str(reference_path),
        exaggeration=float(parameters["exaggeration"]),
        cfg_weight=float(parameters["cfg_weight"]),
        temperature=float(parameters["temperature"]),
        repetition_penalty=float(parameters["repetition_penalty"]),
        min_p=float(parameters["min_p"]),
        top_p=float(parameters["top_p"]),
    )
    generation_seconds = time.perf_counter() - start

    waveform = to_numpy_mono(waveform)
    waveform, raw_peak = normalize_peak(waveform)
    sample_rate = int(model.sr)
    stats = audio_stats(waveform, sample_rate)
    rtf = generation_seconds / stats["duration"] if stats["duration"] > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")

    sidecar = {
        "model": MODEL_NAME,
        "language": LANGUAGE_ID,
        "test_id": test_id,
        "text": text,
        "reference_name": reference_name,
        "reference_path": str(reference_path),
        "seed": seed,
        "parameters": parameters,
        "generation_time": generation_seconds,
        "rtf": rtf,
        "raw_peak_before_normalization": raw_peak,
        "audio": stats,
        "output_path": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "test_id": test_id,
        "text": text,
        "seed": seed,
        "reference_name": reference_name,
        "reference_file": str(reference_path),
        "temperature": parameters["temperature"],
        "cfg_weight": parameters["cfg_weight"],
        "exaggeration": parameters["exaggeration"],
        "repetition_penalty": parameters["repetition_penalty"],
        "min_p": parameters["min_p"],
        "top_p": parameters["top_p"],
        "generation_time": generation_seconds,
        "duration": stats["duration"],
        "rtf": rtf,
        "sample_rate": stats["sample_rate"],
        "rms": stats["rms"],
        "peak": stats["peak"],
        "clipping_count": stats["clipping_count"],
        "silence_ratio": stats["silence_ratio"],
        "output_path": str(output_path),
    }


def merged_parameters(args: argparse.Namespace) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMETERS)
    params.update(
        {
            "temperature": args.temperature,
            "cfg_weight": args.cfg_weight,
            "exaggeration": args.exaggeration,
            "repetition_penalty": args.repetition_penalty,
            "min_p": args.min_p,
            "top_p": args.top_p,
        }
    )
    return params


def load_model(device: str) -> tuple[ChatterboxMultilingualTTS, float]:
    print(f"Loading Chatterbox multilingual on {device}...")
    start = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    load_seconds = time.perf_counter() - start
    print(f"Model loaded in {load_seconds:.2f}s")
    return model, load_seconds


def run_parameter_tests(args: argparse.Namespace) -> None:
    tests = load_tests()
    greeting = tests[0]
    reference_name = args.reference
    reference_path = REFERENCES[reference_name]
    model, load_seconds = load_model(args.device)
    rows = []

    for config in PARAMETER_MATRIX[: args.max_configs]:
        params = dict(DEFAULT_PARAMETERS)
        params.update(config)
        temp_tag = f"temp{int(params['temperature'] * 10):02d}"
        cfg_tag = f"cfg{int(params['cfg_weight'] * 10):02d}"
        output_path = OUTPUT_DIR / f"ar_clone_01_{temp_tag}_{cfg_tag}.wav"
        print(f"Generating {output_path.name}: temperature={params['temperature']}, cfg_weight={params['cfg_weight']}")
        row = generate_one(
            model=model,
            text=greeting["text"],
            output_path=output_path,
            reference_path=reference_path,
            parameters=params,
            seed=args.seed,
            test_id=greeting["id"],
            reference_name=reference_name,
        )
        row["model_load_seconds"] = load_seconds
        rows.append(row)
        print(f"Saved {output_path} | Wav {row['duration']:.2f}s | Gen {row['generation_time']:.2f}s | RTF {row['rtf']:.2f}")

    write_csv(RESULTS_DIR / "chatterbox_arabic_parameter_tests.csv", rows)


def run_final(args: argparse.Namespace) -> None:
    tests = load_tests()
    reference_name = args.reference
    reference_path = REFERENCES[reference_name]
    params = merged_parameters(args)
    model, load_seconds = load_model(args.device)
    rows = []

    for test in tests:
        output_path = OUTPUT_DIR / f"{test['id']}.wav"
        print(f"Generating {test['id']}: {test['text']}")
        row = generate_one(
            model=model,
            text=test["text"],
            output_path=output_path,
            reference_path=reference_path,
            parameters=params,
            seed=args.seed,
            test_id=test["id"],
            reference_name=reference_name,
        )
        row["model_load_seconds"] = load_seconds
        rows.append(row)
        print(f"Saved {output_path.name} | Wav {row['duration']:.2f}s | Gen {row['generation_time']:.2f}s | RTF {row['rtf']:.2f}")

    write_csv(RESULTS_DIR / "chatterbox_arabic_generation.csv", rows)


def run_reference_comparison(args: argparse.Namespace) -> None:
    tests = load_tests()
    greeting = tests[0]
    params = merged_parameters(args)
    model, load_seconds = load_model(args.device)
    rows = []

    for reference_name, reference_path in REFERENCES.items():
        output_path = OUTPUT_DIR / f"ar_clone_01_greeting_ref_{reference_name}.wav"
        print(f"Generating greeting with {reference_name} reference")
        row = generate_one(
            model=model,
            text=greeting["text"],
            output_path=output_path,
            reference_path=reference_path,
            parameters=params,
            seed=args.seed,
            test_id=greeting["id"],
            reference_name=reference_name,
        )
        row["model_load_seconds"] = load_seconds
        rows.append(row)
        print(f"Saved {output_path.name} | Wav {row['duration']:.2f}s | Gen {row['generation_time']:.2f}s | RTF {row['rtf']:.2f}")

    write_csv(RESULTS_DIR / "chatterbox_arabic_reference_generation.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Arabic Chatterbox voice-clone samples on CPU.")
    parser.add_argument("--mode", choices=["parameter-tests", "final", "reference-comparison"], default="final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", choices=sorted(REFERENCES), default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=DEFAULT_PARAMETERS["temperature"])
    parser.add_argument("--cfg-weight", dest="cfg_weight", type=float, default=DEFAULT_PARAMETERS["cfg_weight"])
    parser.add_argument("--exaggeration", type=float, default=DEFAULT_PARAMETERS["exaggeration"])
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=DEFAULT_PARAMETERS["repetition_penalty"])
    parser.add_argument("--min-p", dest="min_p", type=float, default=DEFAULT_PARAMETERS["min_p"])
    parser.add_argument("--top-p", dest="top_p", type=float, default=DEFAULT_PARAMETERS["top_p"])
    parser.add_argument("--max-configs", type=int, default=len(PARAMETER_MATRIX))
    args = parser.parse_args()

    if args.device != "cpu":
        raise ValueError("This local case study is configured for CPU generation. Pass --device cpu.")
    for name, path in REFERENCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} reference: {path}")
    if not TESTS_PATH.exists():
        raise FileNotFoundError(TESTS_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "parameter-tests":
        run_parameter_tests(args)
    elif args.mode == "reference-comparison":
        run_reference_comparison(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
