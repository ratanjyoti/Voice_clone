import os
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"

print("=" * 70)
print("Downloading NeuTTS Air")
print("=" * 70)
print("Repository: neuphonic/neutts-air")
print(f"Cache directory: {CACHE_DIR}")

model_path = snapshot_download(
    repo_id="neuphonic/neutts-air",
    cache_dir=str(CACHE_DIR),
    resume_download=True,
)

print("\nDownload completed successfully.")
print(f"Model snapshot: {model_path}")
