from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


TESTS_PATH = PROJECT_ROOT / "data" / "test_sentences" / "hindi_voice_clone_tests.json"
REFERENCE_AUDIO = PROJECT_ROOT / "data" / "reference_audio" / "hindi" / "ratan_hindi_neutral.wav"
REFERENCE_TEXT = PROJECT_ROOT / "data" / "reference_audio" / "hindi" / "ratan_hindi_neutral.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chatterbox" / "hindi" / "final"
MODEL_NAME = "ResembleAI/chatterbox multilingual"


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


def save_audio(output_path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    ta.save(str(output_path), waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)


def main() -> None:
    tests = json.loads(TESTS_PATH.read_text(encoding="utf-8"))["tests"]
    reference_text = REFERENCE_TEXT.read_text(encoding="utf-8", errors="replace").strip()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Chatterbox Multilingual once on {device}...")
    load_start = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    load_seconds = time.perf_counter() - load_start
    print(f"Model loaded in {load_seconds:.3f}s")

    rows: list[dict[str, object]] = []
    for test in tests:
        output_path = OUTPUT_DIR / f"{test['id']}.wav"
        sidecar_path = output_path.with_suffix(".json")
        if output_path.exists() and sidecar_path.exists():
            print(f"Skipping existing {test['id']}")
            rows.append(json.loads(sidecar_path.read_text(encoding="utf-8")))
            continue

        print(f"Generating {test['id']}")
        start = time.perf_counter()
        waveform = model.generate(
            test["text"],
            language_id="hi",
            audio_prompt_path=str(REFERENCE_AUDIO),
            exaggeration=0.45,
            cfg_weight=0.50,
        )
        generation_seconds = time.perf_counter() - start
        save_audio(output_path, waveform, model.sr)
        stats = audio_stats(output_path)
        rtf = generation_seconds / stats["audio_duration"] if stats["audio_duration"] else None
        sidecar = {
            "test_id": test["id"],
            "category": test["category"],
            "expected_text": test["text"],
            "language": "hindi",
            "language_code": "hi",
            "model": "Chatterbox",
            "model_id": MODEL_NAME,
            "pipeline_type": "zero-shot voice cloning",
            "voice_cloning": True,
            "device": device,
            "reference_audio": str(REFERENCE_AUDIO),
            "reference_text": reference_text,
            "exaggeration": 0.45,
            "cfg_weight": 0.50,
            "model_load_time": load_seconds,
            "generation_time": generation_seconds,
            "rtf": rtf,
            "output_path": str(output_path),
            **stats,
        }
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(sidecar)
        print(f"  {generation_seconds:.3f}s, duration {stats['audio_duration']:.3f}s, RTF {rtf:.3f}")

    (OUTPUT_DIR / "generation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Final Chatterbox Hindi samples written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
