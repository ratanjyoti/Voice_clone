from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_ROOT / "outputs" / "xtts" / "arabic" / "ar_clone_01_greeting.wav"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    if not AUDIO_PATH.exists():
        print(f"exists: False {AUDIO_PATH}")
        return 1
    print(f"exists: True {AUDIO_PATH}")

    audio, sample_rate = sf.read(str(AUDIO_PATH), always_2d=True)
    channels = int(audio.shape[1])
    mono = audio.mean(axis=1).astype(np.float32)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    finite = bool(np.isfinite(mono).all())
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    clipped = int(np.sum(np.abs(mono) >= 0.999)) if mono.size else 0
    silence_ratio = float(np.mean(np.abs(mono) < 0.001)) if mono.size else 1.0
    digest = sha256_file(AUDIO_PATH)

    checks = {
        "duration_gt_0": duration > 0,
        "sample_rate_valid": sample_rate > 0,
        "channels_eq_1": channels == 1,
        "rms_gt_0_01": rms > 0.01,
        "peak_le_0_951": peak <= 0.951,
        "clipped_eq_0": clipped == 0,
        "finite_samples": finite,
    }
    for name, ok in checks.items():
        print(f"{name}: {ok}")
        if not ok:
            failures.append(name)

    print(f"sample_rate: {sample_rate}")
    print(f"channels: {channels}")
    print(f"duration: {duration:.6f}")
    print(f"rms: {rms:.9f}")
    print(f"peak: {peak:.9f}")
    print(f"clipped_samples: {clipped}")
    print(f"silence_ratio: {silence_ratio:.9f}")
    print(f"sha256: {digest}")

    payload = {
        "audio_path": str(AUDIO_PATH),
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "clipped_samples": clipped,
        "silence_ratio": silence_ratio,
        "finite_samples": finite,
        "sha256": digest,
        "checks": checks,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
