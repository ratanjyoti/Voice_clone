from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NLTK_DATA_DIR = PROJECT_ROOT / "data" / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "english" / "melotts" / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [
    {
        "test_id": "en_01_greeting",
        "text": "Hello, welcome to our voice assistant. How can I help you today?",
    },
    {
        "test_id": "en_02_support",
        "text": "I can help you check your order status and answer your questions.",
    },
    {
        "test_id": "en_03_conversation",
        "text": "Sure, I can help with that. Please tell me what you need.",
    },
    {
        "test_id": "en_04_numbers",
        "text": "Your confirmation number is two zero six five, and the meeting starts at three thirty.",
    },
    {
        "test_id": "en_05_expressive",
        "text": "That is great news. I am happy everything worked out well.",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audio_stats(path: Path) -> dict:
    audio, sr = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    abs_audio = np.abs(mono)

    return {
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "audio_duration": float(len(mono) / sr),
        "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
        "peak": float(abs_audio.max()),
        "clipping": bool(np.any(abs_audio >= 0.999)),
        "clipping_samples": int(np.sum(abs_audio >= 0.999)),
        "silence_ratio": float(np.mean(abs_audio < 0.001)),
        "sha256": sha256(path),
    }


def save_sidecar(path: Path, payload: dict) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_nltk_data() -> None:
    import nltk

    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    package_paths = {
        "cmudict": "corpora/cmudict",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    }

    for package, resource_path in package_paths.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK package: {package}")
            nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=False)


def main() -> None:
    ensure_nltk_data()

    from melo.api import TTS

    print("Loading MeloTTS English model...")
    load_start = time.perf_counter()

    device = "cpu"
    model = TTS(language="EN", device=device)

    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.3f}s")

    raw_speaker_ids = model.hps.data.spk2id
    if isinstance(raw_speaker_ids, dict):
        speaker_ids = raw_speaker_ids
    elif hasattr(raw_speaker_ids, "items"):
        speaker_ids = dict(raw_speaker_ids.items())
    else:
        speaker_ids = {
            key: value
            for key, value in vars(raw_speaker_ids).items()
            if not key.startswith("_")
        }

    print("Available speakers:", speaker_ids)
    speaker = speaker_ids.get("EN-US", next(iter(speaker_ids.values())))

    rows = []

    for item in TESTS:
        test_id = item["test_id"]
        text = item["text"]
        output_path = OUTPUT_DIR / f"{test_id}.wav"

        print("=" * 80)
        print(test_id)
        print(text)

        start = time.perf_counter()
        model.tts_to_file(text, speaker, str(output_path), speed=1.0)
        generation_time = time.perf_counter() - start

        stats = audio_stats(output_path)
        rtf = generation_time / stats["audio_duration"]

        payload = {
            "test_id": test_id,
            "text": text,
            "model": "MeloTTS",
            "language": "english",
            "device": device,
            "speaker": str(speaker),
            "model_load_time": load_time,
            "generation_time": generation_time,
            "RTF": rtf,
            "output_path": str(output_path),
            **stats,
        }

        save_sidecar(output_path, payload)
        rows.append(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    summary_path = OUTPUT_DIR / "generation_summary.json"
    summary_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
