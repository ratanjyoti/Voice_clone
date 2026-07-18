from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_REGISTRY = {
    "english": {
        "model_id": "facebook/mms-tts-eng",
        "language_code": "en",
        "tests": PROJECT_ROOT / "data" / "test_sentences" / "english_voice_clone_tests.json",
        "output": PROJECT_ROOT / "outputs" / "mms" / "english" / "final",
    },
    "hindi": {
        "model_id": "facebook/mms-tts-hin",
        "language_code": "hi",
        "tests": PROJECT_ROOT / "data" / "test_sentences" / "hindi_voice_clone_tests.json",
        "output": PROJECT_ROOT / "outputs" / "mms" / "hindi" / "final",
    },
}


def audio_stats(path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    peak = float(abs_audio.max()) if mono.size else 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "audio_duration": duration,
        "peak": peak,
        "clipping": bool(np.any(abs_audio >= 0.999)) if mono.size else False,
        "clipping_samples": int(np.sum(abs_audio >= 0.999)) if mono.size else 0,
        "silence_ratio": float(np.mean(abs_audio < 0.001)) if mono.size else 1.0,
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0,
    }


def safe_normalize(waveform: np.ndarray, ceiling: float = 0.95) -> tuple[np.ndarray, bool]:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak <= 0 or peak <= ceiling:
        return waveform.astype(np.float32), False
    return (waveform * (ceiling / peak)).astype(np.float32), True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final five-sample MMS benchmark.")
    parser.add_argument("--language", choices=MODEL_REGISTRY.keys(), required=True)
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.language]
    tests = json.loads(cfg["tests"].read_text(encoding="utf-8"))["tests"]
    output_dir = cfg["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {cfg['model_id']} once on {device}...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = VitsModel.from_pretrained(cfg["model_id"]).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    rows: list[dict[str, object]] = []
    for test in tests:
        output_path = output_dir / f"{test['id']}.wav"
        print(f"Generating {test['id']}")
        inputs = tokenizer(test["text"], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - start
        waveform = output.waveform.squeeze().detach().cpu().numpy().astype(np.float32)
        waveform, normalized = safe_normalize(waveform)
        sample_rate = int(model.config.sampling_rate)
        sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")
        stats = audio_stats(output_path)
        rtf = generation_seconds / stats["audio_duration"] if stats["audio_duration"] else None
        sidecar = {
            "test_id": test["id"],
            "category": test["category"],
            "expected_text": test["text"],
            "language": args.language,
            "language_code": cfg["language_code"],
            "model": "MMS-TTS",
            "model_id": cfg["model_id"],
            "pipeline_type": "fixed-speaker baseline",
            "voice_cloning": False,
            "device": str(device),
            "model_load_time": load_seconds,
            "generation_time": generation_seconds,
            "rtf": rtf,
            "normalized": normalized,
            "output_path": str(output_path),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            **stats,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(sidecar)
        print(f"  {generation_seconds:.3f}s, duration {stats['audio_duration']:.3f}s, RTF {rtf:.3f}")

    summary_path = output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Final MMS samples written to {output_dir}")


if __name__ == "__main__":
    main()
