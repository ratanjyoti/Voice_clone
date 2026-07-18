from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_NEUTTS = PROJECT_ROOT / "external" / "neutts"
if str(EXTERNAL_NEUTTS) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_NEUTTS))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))

import numpy as np
import soundfile as sf
from neutts import NeuTTS


TESTS_PATH = PROJECT_ROOT / "data" / "test_sentences" / "english_voice_clone_tests.json"
REFERENCE_AUDIO = PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_neutral.wav"
REFERENCE_TEXT_PATH = PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_neutral.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "neutts" / "english" / "final"
MODEL_NAME = "neuphonic/neutts-air"
CODEC_NAME = "neuphonic/neucodec"
SAMPLE_RATE = 24000


def audio_stats(path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "audio_duration": duration,
        "peak": float(abs_audio.max()) if mono.size else 0.0,
        "clipping": bool(np.any(abs_audio >= 0.999)) if mono.size else False,
        "clipping_samples": int(np.sum(abs_audio >= 0.999)) if mono.size else 0,
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if mono.size else 1.0,
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0,
    }


def main() -> None:
    tests = json.loads(TESTS_PATH.read_text(encoding="utf-8"))["tests"]
    reference_text = REFERENCE_TEXT_PATH.read_text(encoding="utf-8", errors="replace").strip()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NeuTTS Air once on CPU...")
    load_start = time.perf_counter()
    tts = NeuTTS(
        backbone_repo=MODEL_NAME,
        backbone_device="cpu",
        codec_repo=CODEC_NAME,
        codec_device="cpu",
    )
    load_seconds = time.perf_counter() - load_start
    print(f"Model loaded in {load_seconds:.3f}s")

    print("Encoding English reference once...")
    ref_start = time.perf_counter()
    reference_codes = tts.encode_reference(str(REFERENCE_AUDIO))
    reference_encoding_seconds = time.perf_counter() - ref_start
    print(f"Reference encoded in {reference_encoding_seconds:.3f}s")

    rows: list[dict[str, object]] = []
    for test in tests:
        output_path = OUTPUT_DIR / f"{test['id']}.wav"
        print(f"Generating {test['id']}")
        start = time.perf_counter()
        waveform = tts.infer(test["text"], reference_codes, reference_text)
        generation_seconds = time.perf_counter() - start
        waveform = np.asarray(waveform, dtype=np.float32).squeeze()
        sf.write(str(output_path), waveform, SAMPLE_RATE, subtype="PCM_16")
        stats = audio_stats(output_path)
        rtf = generation_seconds / stats["audio_duration"] if stats["audio_duration"] else None
        sidecar = {
            "test_id": test["id"],
            "category": test["category"],
            "expected_text": test["text"],
            "language": "english",
            "language_code": "en",
            "model": "NeuTTS",
            "model_id": MODEL_NAME,
            "codec": CODEC_NAME,
            "pipeline_type": "zero-shot voice cloning",
            "voice_cloning": True,
            "device": "cpu",
            "reference_audio": str(REFERENCE_AUDIO),
            "reference_text": reference_text,
            "model_load_time": load_seconds,
            "reference_encoding_time": reference_encoding_seconds,
            "generation_time": generation_seconds,
            "rtf": rtf,
            "output_path": str(output_path),
            **stats,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(sidecar)
        print(f"  {generation_seconds:.3f}s, duration {stats['audio_duration']:.3f}s, RTF {rtf:.3f}")

    (OUTPUT_DIR / "generation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Final NeuTTS samples written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
