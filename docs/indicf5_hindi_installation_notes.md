# IndicF5 Hindi installation notes

Sources checked on 2026-07-18:
- GitHub: https://github.com/AI4Bharat/IndicF5
- Hugging Face model card: https://huggingface.co/ai4bharat/IndicF5
- Raw requirements: https://raw.githubusercontent.com/AI4Bharat/IndicF5/main/requirements.txt
- Raw setup.py: https://raw.githubusercontent.com/AI4Bharat/IndicF5/main/setup.py

## Official installation command
The official README installs with Python 3.10:

```powershell
conda create -n indicf5 python=3.10 -y
conda activate indicf5
pip install git+https://github.com/ai4bharat/IndicF5.git
```

This repository uses the already-created `.venv-indicf5` with Python 3.10.20 instead of recreating the environment.

## Supported Python version
The README explicitly creates a Python 3.10 environment. `setup.py` says `python_requires >=3.7`, but the project README is the stricter operational instruction.

## PyTorch requirement
`requirements.txt` lists `torch>=2.0.0` and `torchaudio>=2.0.0`; it does not pin a specific version. On this CPU-only Windows machine, the recovery run uses CPU PyTorch wheels explicitly instead of CUDA packages.

## Windows support
The README does not state official Windows support. Windows CPU inference is therefore treated as best-effort.

## GPU requirement
The README does not say GPU is mandatory. Given model size and F5-style inference, GPU is likely recommended for practical speed. CPU is best-effort.

## Model checkpoint
The README and Hugging Face card use `ai4bharat/IndicF5` with `AutoModel.from_pretrained(..., trust_remote_code=True)`.

## Cache behavior
Model files are expected to download through Hugging Face Transformers cache. Runtime scripts set `HF_HOME` to the project-local `models/huggingface` directory.

## Hindi support
The README lists Hindi among 11 supported Indian languages.

## Reference audio and transcript
The usage requires three inputs: synthesis text, reference prompt audio, and the text spoken in the reference prompt audio. The reference transcript is mandatory for the public API example.

## Inference API
Official example:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
audio = model(text, ref_audio_path="...wav", ref_text="...")
```

## License and restrictions
The Hugging Face model card lists license `mit` and includes terms prohibiting unauthorized voice cloning. Only voices with explicit permission should be cloned. Commercial restrictions beyond MIT are not stated in the README/model card, but consent and misuse restrictions apply.
## Authenticated CPU test result
After Hugging Face access was granted, model access verification passed for user `ratanjyoti` and model `ai4bharat/IndicF5`.

The first authenticated CPU run exposed a Transformers/meta-device issue with the remote Vocos loader under `transformers 5.14.1`. The environment was pinned to `transformers==4.49.0`, `huggingface-hub==0.36.2`, and `tokenizers==0.21.4` to match the official `<4.50` compatibility range. Because the remote model stores checkpoint keys under compiled module names, `src/generate_indicf5_hindi_test.py` uses a CPU eager `torch.compile` stand-in that preserves `_orig_mod` keys while avoiding actual CPU compilation.

The one authorized Hindi greeting generated successfully on CPU:
- Output: `outputs/indicf5/hindi/hi_clone_01_greeting.wav`
- Generation time: 297.668s
- Audio duration: 7.125s
- RTF: 41.778
- Peak: 0.950012
- Clipping samples: 0
- Faster Whisper Small WER: 50.0%

Decision: do not expand IndicF5 to the full five-sample Hindi benchmark in this CPU run. It is now runnable, but the initial greeting WER is worse than the existing Chatterbox greeting result and CPU latency is far above target.
