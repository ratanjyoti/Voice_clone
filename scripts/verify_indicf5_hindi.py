from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "ai4bharat/IndicF5"
SUPPORTED_LANGUAGES = [
    "Assamese",
    "Bengali",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Odia",
    "Punjabi",
    "Tamil",
    "Telugu",
]

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))


def module_location(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(Path(module.__file__).resolve())


def main() -> None:
    import torch
    import torchaudio
    import transformers
    from transformers import AutoModel

    f5_tts = importlib.import_module("f5_tts")

    print("IndicF5 import verification")
    print(f"Python: {sys.version.split()[0]}")
    print(f"f5_tts location: {module_location('f5_tts')}")
    print(f"f5_tts version: {getattr(f5_tts, '__version__', 'not exposed')}")
    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"torchaudio version: {torchaudio.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Expected checkpoint: {MODEL_ID}")
    print(f"Hindi support: {'Hindi' in SUPPORTED_LANGUAGES}")
    print(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"AutoModel.from_pretrained signature: {inspect.signature(AutoModel.from_pretrained)}")
    print("Official inference call pattern:")
    print(
        "model(text, ref_audio_path='<prompt.wav>', ref_text='<exact prompt transcript>')"
    )
    print("Reference transcript required by official example: True")
    print("Model download attempted in this script: False")
    print("IndicF5 import verification: PASS")


if __name__ == "__main__":
    main()
