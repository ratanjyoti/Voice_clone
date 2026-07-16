from pathlib import Path

from openvoice.api import ToneColorConverter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "checkpoints_v2" / "converter" / "config.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints_v2" / "converter" / "checkpoint.pth"

print("Before ToneColorConverter init", flush=True)
print(f"Config: {CONFIG_PATH}", flush=True)
print(f"Checkpoint: {CHECKPOINT_PATH}", flush=True)

converter = ToneColorConverter(
    str(CONFIG_PATH),
    device="cpu",
)

print("ToneColorConverter initialized on CPU", flush=True)
converter.load_ckpt(str(CHECKPOINT_PATH))
print("OpenVoice V2 converter checkpoint loaded on CPU", flush=True)

