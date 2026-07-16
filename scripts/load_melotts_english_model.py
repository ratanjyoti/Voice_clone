import time
import torch
from melo.api import TTS

print("Loading MeloTTS English model on CPU", flush=True)
print(f"Torch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
start = time.perf_counter()
model = TTS(language="EN", device="cpu")
elapsed = time.perf_counter() - start
print(f"MeloTTS English model loaded successfully in {elapsed:.4f} seconds", flush=True)
print(f"Speakers: {sorted(dict(model.hps.data.spk2id))}", flush=True)
