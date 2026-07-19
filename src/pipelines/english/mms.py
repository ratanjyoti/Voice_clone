import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write
from transformers import AutoTokenizer, VitsModel

print("generate_mms.py started")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_REGISTRY = {
    "english": {
        "model_id": "facebook/mms-tts-eng",
        "language_code": "en",
        "default_output": (
            PROJECT_ROOT
            / "outputs"
            / "mms"
            / "english"
            / "en_mms_smoke_test.wav"
        ),
    },
    "arabic": {
        "model_id": "facebook/mms-tts-ara",
        "language_code": "ar",
        "default_output": (
            PROJECT_ROOT
            / "outputs"
            / "mms"
            / "arabic"
            / "ar_mms_smoke_test.wav"
        ),
    },
    "hindi": {
        "model_id": "facebook/mms-tts-hin",
        "language_code": "hi",
        "default_output": (
            PROJECT_ROOT
            / "outputs"
            / "mms"
            / "hindi"
            / "hi_mms_smoke_test.wav"
        ),
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speech using a Meta MMS-TTS checkpoint."
    )

    parser.add_argument(
        "--language",
        choices=MODEL_REGISTRY.keys(),
        default="english",
        help="Language pipeline to use.",
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Text that will be converted into speech.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output WAV path.",
    )

    return parser.parse_args()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def calculate_audio_duration(
    number_of_samples: int,
    sample_rate: int,
) -> float:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")

    return number_of_samples / sample_rate


def main() -> None:
    args = parse_arguments()

    model_config = MODEL_REGISTRY[args.language]
    model_id = model_config["model_id"]

    output_path = (
        args.output.resolve()
        if args.output is not None
        else model_config["default_output"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("MMS-TTS smoke test")
    print("=" * 70)
    print(f"Language: {args.language}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Text: {args.text}")
    print(f"Output: {output_path}")
    print()

    print("Loading tokenizer and model...")

    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    synchronize_device(device)
    load_time_seconds = time.perf_counter() - load_start

    print(f"Model loaded in {load_time_seconds:.3f} seconds.")

    inputs = tokenizer(
        args.text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("Generating audio...")

    synchronize_device(device)
    generation_start = time.perf_counter()

    with torch.inference_mode():
        output = model(**inputs)

    synchronize_device(device)
    generation_time_seconds = (
        time.perf_counter() - generation_start
    )

    waveform = output.waveform.squeeze().detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)

    sample_rate = int(model.config.sampling_rate)

    write(
        filename=str(output_path),
        rate=sample_rate,
        data=waveform,
    )

    audio_duration_seconds = calculate_audio_duration(
        number_of_samples=len(waveform),
        sample_rate=sample_rate,
    )

    rtf = (
        generation_time_seconds / audio_duration_seconds
        if audio_duration_seconds > 0
        else None
    )

    metadata = {
        "language": args.language,
        "language_code": model_config["language_code"],
        "model_id": model_id,
        "device": str(device),
        "input_text": args.text,
        "output_file": str(output_path),
        "sample_rate_hz": sample_rate,
        "audio_samples": len(waveform),
        "model_load_time_seconds": round(load_time_seconds, 4),
        "generation_time_seconds": round(
            generation_time_seconds,
            4,
        ),
        "audio_duration_seconds": round(
            audio_duration_seconds,
            4,
        ),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "supports_voice_cloning": False,
        "supports_streaming": False,
    }

    metadata_path = output_path.with_suffix(".json")

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Generation completed successfully.")
    print("-" * 70)
    print(f"WAV file: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(
        f"Generation time: "
        f"{generation_time_seconds:.3f} seconds"
    )
    print(
        f"Audio duration: "
        f"{audio_duration_seconds:.3f} seconds"
    )
    print(f"RTF: {rtf:.4f}")
    print("-" * 70)


if __name__ == "__main__":
    print("Script entry point reached.", flush=True)

    try:
        main()
    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise