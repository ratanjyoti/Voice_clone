from __future__ import annotations

import json
import time
from pathlib import Path
from xml.parsers.expat import model

import soundfile as sf
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "arabic"
    / "professional_msa"
    / "arabic_reference_standard.wav"
)

TESTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "test_sentences"
    / "arabic_voice_clone_tests.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "chatterbox"
    / "arabic"
)

OUTPUT_PATH = OUTPUT_DIR / "ar_clone_01_greeting.wav"


def main() -> None:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {REFERENCE_PATH}"
        )

    if not TESTS_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found: {TESTS_PATH}"
        )

    tests_data = json.loads(
        TESTS_PATH.read_text(encoding="utf-8")
    )

    test = tests_data["tests"][0]
    text = test["text"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device="cpu"

    print("=" * 72)
    print("Chatterbox Arabic voice-cloning test")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Reference: {REFERENCE_PATH}")
    print(f"Test ID: {test['id']}")
    print(f"Text: {text}")
    print("Loading multilingual Chatterbox...")

    load_start = time.perf_counter()

    model = ChatterboxMultilingualTTS.from_pretrained(
        device=device
    )

    load_seconds = time.perf_counter() - load_start

    print(f"Model load time: {load_seconds:.2f} seconds")
    print("Generating Arabic audio...")

    generation_start = time.perf_counter()

    waveform = model.generate(
        text=text,
        language_id="ar",
        audio_prompt_path=str(REFERENCE_PATH),
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    )

    generation_seconds = (
        time.perf_counter() - generation_start
    )

    if isinstance(waveform, torch.Tensor):
        waveform = (
            waveform.detach()
            .cpu()
            .float()
            .squeeze()
            .numpy()
        )

    waveform = waveform.astype("float32")

    raw_peak = float(abs(waveform).max())

    if raw_peak > 0:
        target_peak = 0.95
        waveform = waveform * (target_peak / raw_peak)

    normalized_peak = float(abs(waveform).max())

    print(f"Raw waveform peak: {raw_peak:.4f}")
    print(f"Normalized peak: {normalized_peak:.4f}")

    sample_rate = int(model.sr)

    sf.write(
        str(OUTPUT_PATH),
        waveform,
        sample_rate,
        subtype="PCM_16",
    )

    duration_seconds = len(waveform) / sample_rate
    rtf = (
        generation_seconds / duration_seconds
        if duration_seconds > 0
        else 0.0
    )

    print("=" * 72)
    print("Generation completed")
    print("=" * 72)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(
        f"Generation time: "
        f"{generation_seconds:.2f} seconds"
    )
    print(f"RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()