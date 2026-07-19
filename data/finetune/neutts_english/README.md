# NeuTTS English Fine-Tuning Dataset

This folder is a scaffold only. Do not start NeuTTS fine-tuning until a valid single-speaker dataset exists and passes validation.

Minimum prototype dataset:
- At least 30 minutes of clean speech.
- One speaker only.
- WAV mono, 24 kHz.
- Clips should be 3-12 seconds.
- Exact transcript required for every clip.
- No background noise, music, or overlapping speech.

Recommended real-improvement dataset:
- 2-10 hours of clean, consistent speech from the same speaker.
- The official NeuTTS guidance indicates roughly 1000-2000 steps may be sufficient around 10 hours, with learning rates around 1e-5 to 4e-5.

Expected layout:

```text
data/finetune/neutts_english/
  audio/
    clip_0001.wav
  metadata_template.csv
```

Run validation before training:

```powershell
.\.venvs\.venv\Scripts\python.exe -u scripts\prepare_neutts_finetune_dataset.py
```

Outputs:
- `results/diagnostics/neutts_finetune_dataset_validation.csv`
- `evidence/result_snapshots/neutts_finetune_dataset_summary.json`
