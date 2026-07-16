import sys

print("Python:", sys.executable, flush=True)
print("Before MeloTTS import", flush=True)

from melo.api import TTS

print("MeloTTS import successful", flush=True)