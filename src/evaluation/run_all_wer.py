import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "test_sentences"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Configuration of the pipelines you've run
PIPELINES = {
    "mms": {
        "model_id": "facebook-mms",
        "langs": {"english": "en", "arabic": "ar", "hindi": "hi"}
    },
    "xtts_v2": {
        "model_id": "coqui-xtts-v2",
        "langs": {"english": "en", "arabic": "ar", "hindi": "hi"}
    }
}

def run_wer():
    for pipe_name, config in PIPELINES.items():
        for lang_name, lang_code in config["langs"].items():
            print(f"Processing {lang_name} for {pipe_name}...")
            
            # Load the test sentences
            with open(DATA_DIR / f"{lang_name}.json", "r", encoding="utf-8") as f:
                sentences = json.load(f)
            
            for sent in sentences:
                # Construct path to the wav file
                # Matches the naming convention: {sentence_id}_run_01.wav
                # We take run_01 as the representative sample
                wav_path = OUTPUTS_DIR / pipe_name / lang_name / f"{sent['id']}_run_01.wav"
                
                if wav_path.exists():
                    cmd = [
                        "python", "src/calculate_wer.py",
                        "--audio", str(wav_path),
                        "--reference", sent['text'],
                        "--language", lang_code,
                        "--model-id", config["model_id"],
                        "--sentence-id", sent['id']
                    ]
                    subprocess.run(cmd)
                else:
                    print(f"Warning: File not found {wav_path}")

if __name__ == "__main__":
    run_wer()
