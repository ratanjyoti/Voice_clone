from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_NEUTTS = PROJECT_ROOT / "external" / "neutts"
SYSTEM_ESPEAK_DIR = Path(r"C:\Program Files\eSpeak NG")
SYSTEM_ESPEAK_LIBRARY = SYSTEM_ESPEAK_DIR / "libespeak-ng.dll"
SYSTEM_ESPEAK_DATA = SYSTEM_ESPEAK_DIR / "espeak-ng-data"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
if EXTERNAL_NEUTTS.exists():
    sys.path.insert(0, str(EXTERNAL_NEUTTS))
if SYSTEM_ESPEAK_LIBRARY.exists():
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(SYSTEM_ESPEAK_LIBRARY))
if SYSTEM_ESPEAK_DATA.exists():
    os.environ.setdefault("ESPEAK_DATA_PATH", str(SYSTEM_ESPEAK_DATA))

from preprocessing.english_tts_normalizer import normalize_for_tts


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "english" / "neutts_improved" / "final"
REFERENCE_AUDIO = PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_neutral.wav"
REFERENCE_TEXT_PATH = PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_neutral.txt"

MODEL_NAME = "neuphonic/neutts-air"
CODEC_NAME = "neuphonic/neucodec"
SAMPLE_RATE = 24000
SEED = 42
SAMPLE_TIMEOUT_SECONDS = int(os.environ.get("NEUTTS_SAMPLE_TIMEOUT_SECONDS", "300"))

BENCHMARK_SENTENCES = [
    {
        "test_id": "air_01_greeting",
        "category": "greeting",
        "expected_text": "Hello, welcome to Infinia. I am ready to help you today.",
    },
    {
        "test_id": "air_02_identity",
        "category": "identity",
        "expected_text": (
            "My name is Ratan Jyoti, and I enjoy building practical artificial "
            "intelligence systems."
        ),
    },
    {
        "test_id": "air_03_support",
        "category": "customer_support",
        "expected_text": (
            "Thank you for contacting customer support. I have reviewed your request "
            "and will guide you through the next steps."
        ),
    },
    {
        "test_id": "air_04_numbers",
        "category": "numbers_and_date",
        "expected_text": (
            "Your order number is five eight two four, and it will arrive on July "
            "twenty-first."
        ),
    },
    {
        "test_id": "air_05_expressive",
        "category": "expressive",
        "expected_text": (
            "That is wonderful news! Everything worked correctly, and I hope you have "
            "a great day."
        ),
    },
]


class TimeoutError(RuntimeError):
    pass


@contextmanager
def sample_timeout(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(_signum, _frame):
        raise TimeoutError(f"NeuTTS sample exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
    except Exception:
        pass


def audio_stats(path: Path) -> dict:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if mono.size == 0:
        raise RuntimeError(f"Generated audio is empty: {path}")

    abs_audio = np.abs(mono)
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "audio_duration": float(mono.size / sample_rate),
        "peak": float(abs_audio.max()),
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
        "clipping": bool(np.any(abs_audio >= 0.999)),
        "clipping_samples": int(np.sum(abs_audio >= 0.999)),
        "silence_ratio": float(np.mean(abs_audio < 0.001)),
    }


def validate_audio(path: Path, stats: dict) -> None:
    if not path.exists() or path.stat().st_size <= 44:
        raise RuntimeError(f"Output WAV is missing or header-only: {path}")
    if stats["audio_duration"] <= 0:
        raise RuntimeError("Output WAV duration is invalid")
    if stats["rms"] <= 0.0005:
        raise RuntimeError(f"Output WAV appears silent: RMS={stats['rms']}")
    if stats["sample_rate"] != SAMPLE_RATE:
        raise RuntimeError(f"Unexpected sample rate: {stats['sample_rate']}")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def import_official_neutts():
    script_dir = Path(__file__).resolve().parent
    original_sys_path = list(sys.path)
    filtered = []
    for entry in sys.path:
        try:
            if Path(entry).resolve() == script_dir:
                continue
        except Exception:
            pass
        filtered.append(entry)
    sys.path[:] = filtered
    cached = sys.modules.pop("neutts", None)
    try:
        from neutts import NeuTTS
    finally:
        sys.path[:] = original_sys_path
        if cached is not None:
            sys.modules["neutts"] = cached

    return NeuTTS

def failed_sidecar(sample: dict, error: str, reference_text: str = "") -> dict:
    normalized = normalize_for_tts(sample["expected_text"])
    return {
        "test_id": sample["test_id"],
        "category": sample["category"],
        "expected_text": normalized.expected_text,
        "tts_input_text": normalized.tts_input_text,
        "model": MODEL_NAME,
        "codec": CODEC_NAME,
        "reference_audio": str(REFERENCE_AUDIO),
        "reference_text": reference_text,
        "generation_time": 0.0,
        "audio_duration": 0.0,
        "RTF": 0.0,
        "sample_rate": SAMPLE_RATE,
        "peak": 0.0,
        "rms": 0.0,
        "clipping": False,
        "clipping_samples": 0,
        "silence_ratio": 1.0,
        "status": "failed",
        "error": error,
    }


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_AUDIO.exists():
        raise FileNotFoundError(f"Reference audio missing: {REFERENCE_AUDIO}")
    if not REFERENCE_TEXT_PATH.exists():
        raise FileNotFoundError(f"Reference transcript missing: {REFERENCE_TEXT_PATH}")

    reference_text = REFERENCE_TEXT_PATH.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise RuntimeError(f"Reference transcript is empty: {REFERENCE_TEXT_PATH}")

    print("Loading NeuTTS improved English pipeline once...")
    load_start = time.perf_counter()
    try:
        NeuTTS = import_official_neutts()

        tts = NeuTTS(
            backbone_repo=MODEL_NAME,
            backbone_device="cpu",
            codec_repo=CODEC_NAME,
            codec_device="cpu",
        )
    except Exception as exc:
        failure = {
            "status": "failed",
            "stage": "model_load",
            "model": MODEL_NAME,
            "codec": CODEC_NAME,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(OUTPUT_DIR / "generation_summary.json", failure)
        for sample in BENCHMARK_SENTENCES:
            failure_sidecar = failed_sidecar(sample, f"model_load failed: {exc}", reference_text)
            save_json(OUTPUT_DIR / f"{sample['test_id']}.json", failure_sidecar)
            save_json(OUTPUT_DIR / f"{sample['test_id']}_failure.json", failure_sidecar)
        raise

    model_load_time = time.perf_counter() - load_start
    print(f"Model loaded in {model_load_time:.3f}s")

    print("Encoding reference once...")
    encode_start = time.perf_counter()
    reference_codes = tts.encode_reference(str(REFERENCE_AUDIO))
    reference_encoding_time = time.perf_counter() - encode_start
    print(f"Reference encoded in {reference_encoding_time:.3f}s")

    results = []
    for index, sample in enumerate(BENCHMARK_SENTENCES, start=1):
        output_path = OUTPUT_DIR / f"{sample['test_id']}.wav"
        sidecar_path = OUTPUT_DIR / f"{sample['test_id']}.json"
        normalized = normalize_for_tts(sample["expected_text"])
        result = failed_sidecar(sample, "", reference_text)
        result.update(
            {
                "model_load_time": model_load_time,
                "reference_encoding_time": reference_encoding_time,
                "output_path": str(output_path),
            }
        )

        print(f"[{index}/{len(BENCHMARK_SENTENCES)}] {sample['test_id']}")
        print(f"TTS input: {normalized.tts_input_text}")

        try:
            start = time.perf_counter()
            with sample_timeout(SAMPLE_TIMEOUT_SECONDS):
                waveform = tts.infer(
                    normalized.tts_input_text,
                    reference_codes,
                    reference_text,
                )
            generation_time = time.perf_counter() - start
            waveform = np.asarray(waveform, dtype=np.float32).squeeze()
            if waveform.size == 0:
                raise RuntimeError("NeuTTS returned empty audio")

            sf.write(str(output_path), waveform, SAMPLE_RATE, subtype="PCM_16")
            stats = audio_stats(output_path)
            validate_audio(output_path, stats)

            result.update(
                {
                    "generation_time": generation_time,
                    "audio_duration": stats["audio_duration"],
                    "RTF": generation_time / stats["audio_duration"],
                    **stats,
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            failure_path = OUTPUT_DIR / f"{sample['test_id']}_failure.json"
            save_json(failure_path, result)
            print(f"FAILED: {exc}")

        save_json(sidecar_path, result)
        results.append(result)

    successes = sum(1 for row in results if row["status"] == "success")
    summary = {
        "status": "success" if successes == len(results) else "partial",
        "model": MODEL_NAME,
        "codec": CODEC_NAME,
        "seed": SEED,
        "sample_timeout_seconds": SAMPLE_TIMEOUT_SECONDS,
        "timeout_note": "SIGALRM timeout is active only on platforms that provide it.",
        "reference_audio": str(REFERENCE_AUDIO),
        "reference_text": reference_text,
        "model_load_time": model_load_time,
        "reference_encoding_time": reference_encoding_time,
        "total_samples": len(results),
        "successful_samples": successes,
        "failed_samples": len(results) - successes,
        "samples": results,
    }
    save_json(OUTPUT_DIR / "generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if successes != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()





