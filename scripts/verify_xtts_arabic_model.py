from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "models" / "coqui"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

os.environ.setdefault("TTS_HOME", str(CACHE_DIR))
os.environ.setdefault("COQUI_TOS_AGREED", "1")
for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
    os.environ[proxy_name] = ""


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def main() -> int:
    try:
        import torch
        import TTS as tts_pkg
        from TTS.api import TTS

        before = rss_mb()
        start = time.perf_counter()
        api = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)
        load_seconds = time.perf_counter() - start
        after = rss_mb()

        synthesizer = getattr(api, "synthesizer", None)
        languages = getattr(api, "languages", None)
        sample_rate = getattr(synthesizer, "output_sample_rate", None)
        tts_config = getattr(synthesizer, "tts_config", None)
        if sample_rate is None and tts_config is not None:
            sample_rate = getattr(tts_config, "output_sample_rate", None)

        print(f"model_name: {MODEL_NAME}")
        print("device: cpu")
        print(f"torch_version: {torch.__version__}")
        print(f"coqui_tts_version: {tts_pkg.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"model_load_seconds: {load_seconds:.3f}")
        print(f"memory_before_mb: {before}")
        print(f"memory_after_mb: {after}")
        if before is not None and after is not None:
            print(f"memory_delta_mb: {after - before:.3f}")
        print(f"supported_languages: {languages}")
        print(f"arabic_supported: {'ar' in (languages or [])}")
        print(f"sample_rate: {sample_rate}")
        print(f"tts_home: {os.environ.get('TTS_HOME')}")
        print(f"expected_cache_dir: {CACHE_DIR / 'tts' / 'tts_models--multilingual--multi-dataset--xtts_v2'}")
        print("XTTS model load: PASS")
        return 0
    except Exception as exc:
        print("XTTS model load: FAIL")
        print(f"error_type: {type(exc).__name__}")
        print(f"error: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
