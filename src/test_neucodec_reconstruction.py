import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from neucodec import NeuCodec


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_AUDIO = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "english"
    / "ratan_neutral.wav"
)

OUTPUT_AUDIO = (
    PROJECT_ROOT
    / "outputs"
    / "neutts"
    / "diagnostics"
    / "ratan_neutral_neucodec_reconstructed.wav"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "neucodec_reconstruction_test.json"
)


def read_audio_stats(path: Path) -> dict:
    audio, sample_rate = sf.read(
        str(path),
        always_2d=True,
    )

    if audio.size == 0:
        raise RuntimeError(
            f"Audio contains no samples: {path}"
        )

    duration = audio.shape[0] / float(sample_rate)
    rms = float(
        np.sqrt(np.mean(np.square(audio)))
    )
    peak = float(np.max(np.abs(audio)))

    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_seconds": duration,
        "rms": rms,
        "peak": peak,
        "all_finite": bool(
            np.isfinite(audio).all()
        ),
        "size_bytes": int(path.stat().st_size),
    }


def main() -> None:
    print("=" * 70)
    print("NeuCodec reconstruction diagnostic")
    print("=" * 70)

    if not INPUT_AUDIO.exists():
        raise FileNotFoundError(
            f"Input reference not found: {INPUT_AUDIO}"
        )

    OUTPUT_AUDIO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input audio: {INPUT_AUDIO}")
    print(f"Output audio: {OUTPUT_AUDIO}")
    print(f"Torch version: {torch.__version__}")
    print("Device: CPU")

    input_stats = read_audio_stats(INPUT_AUDIO)

    print("\nInput properties:")
    print(
        f"Sample rate: {input_stats['sample_rate']}"
    )
    print(f"Channels: {input_stats['channels']}")
    print(
        f"Duration: "
        f"{input_stats['duration_seconds']:.4f}s"
    )
    print(f"RMS: {input_stats['rms']:.8f}")
    print(f"Peak: {input_stats['peak']:.8f}")

    print("\nLoading NeuCodec...")

    load_start = time.perf_counter()

    codec = NeuCodec.from_pretrained(
        "neuphonic/neucodec"
    )

    codec = codec.to("cpu")
    codec.eval()

    load_time = time.perf_counter() - load_start

    print(
        f"NeuCodec loaded in {load_time:.4f}s"
    )

    print("\nLoading and preparing reference audio...")

    waveform, sample_rate = torchaudio.load(
        str(INPUT_AUDIO)
    )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    # NeuCodec expects 16 kHz input.
    if sample_rate != 16000:
        print(
            f"Resampling from {sample_rate} Hz "
            "to 16000 Hz..."
        )

        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=16000,
        )

    waveform = waveform.to(
    device="cpu",
    dtype=torch.float32,
)

# NeuCodec expects [batch, channel, samples].
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim != 3:
        raise RuntimeError(
            "Unexpected waveform shape after preparation: "
            f"{waveform.shape}"
        )

    print(f"Prepared shape: {waveform.shape}")
    print("\nEncoding and decoding reference audio...")

    reconstruction_start = time.perf_counter()

    with torch.inference_mode():
        # Different NeuCodec releases may expose
        # slightly different method signatures.
        try:
            codes = codec.encode_code(waveform)
        except (TypeError, AttributeError):
            codes = codec.encode_code(
                str(INPUT_AUDIO)
            )

        print(
            "Encoded reference successfully."
        )

        try:
            reconstructed = codec.decode_code(
                codes
            )
        except AttributeError as exc:
            raise RuntimeError(
                "This NeuCodec version does not expose "
                "decode_code(). Inspect its available "
                "methods before changing dependencies."
            ) from exc

    reconstruction_time = (
        time.perf_counter()
        - reconstruction_start
    )

    if isinstance(reconstructed, tuple):
        reconstructed = reconstructed[0]

    if not torch.is_tensor(reconstructed):
        reconstructed = torch.as_tensor(
            reconstructed
        )

    reconstructed = reconstructed.detach().cpu()

    while reconstructed.ndim > 2:
        reconstructed = reconstructed.squeeze(0)

    if reconstructed.ndim == 1:
        reconstructed = reconstructed.unsqueeze(0)

    if reconstructed.shape[0] > 1:
        reconstructed = reconstructed.mean(
            dim=0,
            keepdim=True,
        )

    if reconstructed.numel() == 0:
        raise RuntimeError(
            "NeuCodec returned empty audio."
        )

    if not torch.isfinite(reconstructed).all():
        raise RuntimeError(
            "Reconstructed audio contains "
            "NaN or infinite values."
        )

    torchaudio.save(
        str(OUTPUT_AUDIO),
        reconstructed,
        sample_rate=24000,
        encoding="PCM_S",
        bits_per_sample=16,
    )

    if not OUTPUT_AUDIO.exists():
        raise RuntimeError(
            f"Output was not created: {OUTPUT_AUDIO}"
        )

    output_stats = read_audio_stats(OUTPUT_AUDIO)

    report = {
        "status": "success",
        "test": "neucodec_reconstruction",
        "model": "neuphonic/neucodec",
        "device": "cpu",
        "input_audio": str(INPUT_AUDIO),
        "output_audio": str(OUTPUT_AUDIO),
        "model_load_time_seconds": load_time,
        "reconstruction_time_seconds": (
            reconstruction_time
        ),
        "input_stats": input_stats,
        "output_stats": output_stats,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("Reconstruction completed")
    print("=" * 70)

    print(f"Output: {OUTPUT_AUDIO}")
    print(
        f"Output duration: "
        f"{output_stats['duration_seconds']:.4f}s"
    )
    print(
        f"Output sample rate: "
        f"{output_stats['sample_rate']}"
    )
    print(
        f"Output channels: "
        f"{output_stats['channels']}"
    )
    print(
        f"Reconstruction time: "
        f"{reconstruction_time:.4f}s"
    )
    print(
        f"Output RMS: "
        f"{output_stats['rms']:.8f}"
    )
    print(
        f"Output peak: "
        f"{output_stats['peak']:.8f}"
    )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()