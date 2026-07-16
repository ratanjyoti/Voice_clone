import argparse
import json
import wave
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_AUDIO = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "ratan_reference_22050_mono.wav"
)

DEFAULT_REPORT = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "reference_audio_quality.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the voice-cloning reference WAV."
    )

    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    return parser.parse_args()


def load_pcm_wave(audio_path: Path) -> tuple[np.ndarray, dict]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()

        raw_audio = wav_file.readframes(frame_count)

    if sample_width == 1:
        samples = np.frombuffer(
            raw_audio,
            dtype=np.uint8,
        ).astype(np.float32)

        samples = (samples - 128.0) / 128.0

    elif sample_width == 2:
        samples = np.frombuffer(
            raw_audio,
            dtype=np.int16,
        ).astype(np.float32)

        samples /= 32768.0

    elif sample_width == 4:
        samples = np.frombuffer(
            raw_audio,
            dtype=np.int32,
        ).astype(np.float32)

        samples /= 2147483648.0

    else:
        raise ValueError(
            f"Unsupported sample width: {sample_width} bytes"
        )

    if channels > 1:
        samples = samples.reshape(-1, channels)
        samples = samples.mean(axis=1)

    metadata = {
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_seconds": (
            frame_count / sample_rate
            if sample_rate > 0
            else 0.0
        ),
    }

    return samples, metadata


def calculate_silence_ratio(
    samples: np.ndarray,
    threshold: float = 0.01,
) -> float:
    if len(samples) == 0:
        return 1.0

    return float(
        np.mean(
            np.abs(samples) < threshold
        )
    )


def main() -> None:
    args = parse_arguments()

    audio_path = args.audio.resolve()
    report_path = args.report.resolve()

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {audio_path}"
        )

    samples, metadata = load_pcm_wave(audio_path)

    if len(samples) == 0:
        raise ValueError(
            "Reference recording contains no audio samples."
        )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(samples)
            )
        )
    )

    peak = float(
        np.max(
            np.abs(samples)
        )
    )

    clipping_ratio = float(
        np.mean(
            np.abs(samples) >= 0.999
        )
    )

    silence_ratio = calculate_silence_ratio(samples)

    checks = {
        "duration_10_to_30_seconds": (
            10.0
            <= metadata["duration_seconds"]
            <= 30.0
        ),
        "mono": metadata["channels"] == 1,
        "sample_rate_22050": (
            metadata["sample_rate_hz"] == 22050
        ),
        "pcm_16_bit": (
            metadata["sample_width_bytes"] == 2
        ),
        "non_silent": rms >= 0.01,
        "acceptable_peak": (
            0.10 <= peak <= 1.0
        ),
        "low_clipping": clipping_ratio <= 0.001,
        "reasonable_silence": silence_ratio <= 0.70,
    }

    passed = all(checks.values())

    report = {
        "audio_file": str(audio_path),
        **metadata,
        "rms": round(rms, 8),
        "peak": round(peak, 8),
        "clipping_ratio": round(
            clipping_ratio,
            8,
        ),
        "silence_ratio": round(
            silence_ratio,
            8,
        ),
        "checks": checks,
        "passed": passed,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("=" * 72)
    print("Reference audio quality validation")
    print("=" * 72)
    print(f"File: {audio_path}")
    print(
        f"Duration: "
        f"{metadata['duration_seconds']:.4f} seconds"
    )
    print(
        f"Sample rate: "
        f"{metadata['sample_rate_hz']} Hz"
    )
    print(f"Channels: {metadata['channels']}")
    print(
        f"Sample width: "
        f"{metadata['sample_width_bytes'] * 8}-bit"
    )
    print(f"RMS: {rms:.8f}")
    print(f"Peak: {peak:.8f}")
    print(
        f"Clipping ratio: "
        f"{clipping_ratio:.8f}"
    )
    print(f"Silence ratio: {silence_ratio:.8f}")
    print()

    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"{status:<4} {name}")

    print()
    print(
        "Overall result: "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"Report: {report_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()