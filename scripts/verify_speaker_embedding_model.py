from __future__ import annotations

import os
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PROJECT_ROOT / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
HF_CACHE = PROJECT_ROOT / "models" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))

from speechbrain.inference.speaker import EncoderClassifier


def main() -> None:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_CACHE),
        run_opts={"device": "cpu"},
    )

    print("Speaker embedding model: PASS")


if __name__ == "__main__":
    main()
