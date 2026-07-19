# Running NeuTTS Fine-Tuning In Colab

This is a scaffold, not evidence that fine-tuning completed.

1. Clone the official NeuTTS repository.

```bash
git clone https://github.com/neuphonic/neutts-air.git
cd neutts-air
```

2. Install the official dependencies from that repository.

```bash
pip install -r requirements.txt
pip install -e .
```

3. Copy this repository's validated dataset into the Colab runtime.

Required files:
- `data/finetune/neutts_english/audio/*.wav`
- `data/finetune/neutts_english/manifest_clean.csv`
- `configs/neutts_finetune_ratan_english.yaml`

4. Encode the WAV files with NeuCodec as required by the official NeuTTS fine-tuning guide.

Use the command/API from the official repo version you cloned. Save encoded codes into the path configured in `encoded_audio_dir`.

5. Compare `configs/neutts_finetune_ratan_english.yaml` with official `examples/finetune_config.yaml`.

Edit all fields marked `EDIT`, especially checkpoint, manifest, encoded-code paths, precision, batch size, and output directory.

6. Run the official fine-tuning script.

```bash
python examples/finetune.py --config /content/neutts_finetune_ratan_english.yaml
```

7. Export the fine-tuned checkpoint.

Copy the final checkpoint and its config back into this repository under a new checkpoint folder. Do not overwrite the baseline evidence.

8. Rerun the exact same five-sample improved generation/evaluation workflow.

```powershell
.\.venvs\.venv-neutts\Scripts\python.exe -u src\pipelines\english\neutts_improved.py
.\.venvs\.venv\Scripts\python.exe -u src\evaluation\neutts_english_improved_eval.py
```

NeuTTS can only become the English winner if the rerun passes WER <= 10%, no clipping, all five WAVs generated, clean sidecars/logs, and manual listening confirms it remains best human-like.
