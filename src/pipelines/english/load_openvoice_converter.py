import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)

import torch
from openvoice.api import ToneColorConverter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = (
    PROJECT_ROOT
    / "checkpoints_v2"
    / "converter"
    / "config.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints_v2"
    / "converter"
    / "checkpoint.pth"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "openvoice_converter_load.json"
)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found: {CONFIG_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = "cpu"

    print("=" * 72)
    print("OpenVoice V2 converter load test")
    print("=" * 72)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()

    start_time = time.perf_counter()

    converter = ToneColorConverter(
        str(CONFIG_PATH),
        device=device,
    )

    converter.load_ckpt(
        str(CHECKPOINT_PATH)
    )

    elapsed = time.perf_counter() - start_time

    report = {
        "status": "success",
        "device": device,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config_path": str(CONFIG_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "load_time_seconds": round(elapsed, 4),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("OpenVoice converter loaded successfully.")
    print(f"Load time: {elapsed:.4f} seconds")
    print(f"Report: {REPORT_PATH}")

    # Keep a reference until the test completes.
    _ = converter


if __name__ == "__main__":
    main()
