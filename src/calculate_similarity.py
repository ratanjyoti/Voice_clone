import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np

# Load the pre-trained speaker embedding model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    # Ensure audio is mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    # The model expects 16kHz
    # (Simplified: assuming your audio is already correct or model handles it)
    embeddings = model.encode_batch(signal)
    return embeddings.squeeze().detach().numpy()

def calculate_similarity(ref_path, gen_path):
    emb_ref = get_embedding(ref_path)
    emb_gen = get_embedding(gen_path)
    
    # Cosine similarity expects 2D arrays
    sim = cosine_similarity(emb_ref.reshape(1, -1), emb_gen.reshape(1, -1))
    return sim[0][0]

if __name__ == "__main__":
    ref_voice = "data/reference_audio/my_voice.wav"
    # Test one cloned file
    test_gen = "outputs/xtts_v2/english/en_01_run_01.wav"
    
    score = calculate_similarity(ref_voice, test_gen)
    print(f"Speaker Similarity Score: {score:.4f}")
    # Target: > 0.75
