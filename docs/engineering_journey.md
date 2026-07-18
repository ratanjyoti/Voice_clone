# TTS Engineering Journey

This document records verified implementation steps, commands, outputs, failures, fixes, and decisions.

## XTTS-v2 Arabic Setup

### Environment creation

- Environment: .venv-xtts-arabic
- Python: 3.11.15
- Architecture: 64-bit
- Device target: CPU
- PyTorch: 2.8.0+cpu
- Torchaudio: 2.8.0+cpu
- CUDA available: False
- Status: PASS

The XTTS environment was isolated from the existing Chatterbox and Arabic dataset environments to prevent dependency conflicts.

### Coqui TTS installation verification

- Coqui TTS version: 0.27.5
- Transformers version: 4.57.6
- PyTorch version: 2.8.0+cpu
- CUDA available: False
- TTS import: PASS
- TTS module: .venv-xtts-arabic\Lib\site-packages\TTS\__init__.py
- Verification log: evidence\terminal_logs\xtts_coqui_import_verification.txt

A compatibility failure occurred with Transformers 5.14.1 because Coqui attempted to import isin_mps_friendly. Transformers was isolatedly downgraded inside .venv-xtts-arabic to 4.57.6. The working .venv and .venv-arabic-data environments were not modified.

### XTTS-v2 registry and Arabic support

- Model: tts_models/multilingual/multi-dataset/xtts_v2
- Registered models found: 94
- XTTS-v2 registry entry: PASS
- Arabic language code r: PRESENT
- Registry log: evidence\terminal_logs\xtts_model_registry_verification.txt
- Supported-language log: evidence\terminal_logs\xtts_supported_languages.txt
- Status: PASS

## 2026-07-17 - XTTS-v2 Phase 1-3: Install, Registry, Language, Model Load, License

- Objective: Install and verify Coqui XTTS-v2 Arabic zero-shot evaluation environment without modifying Chatterbox.
- Commands: `pip install coqui-tts`, Coqui import verification, registry listing, supported-language/model load, `scripts\verify_xtts_arabic_model.py`.
- Result: Coqui TTS 0.27.5 imports successfully with torch 2.8.0+cpu; CUDA unavailable. Registry includes `tts_models/multilingual/multi-dataset/xtts_v2`; installed model metadata reports XTTS-v2.0.3, commit `480a6cdf7`, license `CPML`, and `tos_required: true`.
- Model cache: `models\coqui\tts\tts_models--multilingual--multi-dataset--xtts_v2`.
- Model load: PASS on CPU; load time 22.957 seconds; memory before 487.46 MB, after 2561.19 MB, delta 2073.73 MB; sample rate 24000; languages include `ar`.
- Failure: Initial download failed because inherited proxy variables pointed to `127.0.0.1:9`; a partial 8 KB checkpoint was created and locked by the timed-out downloader.
- Fix: Stopped the stuck XTTS Python process, removed the partial cache, cleared proxy environment variables, accepted Coqui TOS through `COQUI_TOS_AGREED=1`, and downloaded into the project-local cache.
- Evidence: `evidence\terminal_logs\xtts_coqui_install.txt`, `xtts_coqui_import_verification.txt`, `xtts_model_registry_verification.txt`, `xtts_supported_languages.txt`, `xtts_model_load.txt`, `xtts_model_registry_item.txt`.
- Decision: Continue zero-shot XTTS evaluation only; licensing must be reviewed before deployment or commercial use.

## 2026-07-17 - XTTS-v2 Phase 4-5: Initial Arabic Greeting

- Objective: Generate and evaluate one Arabic XTTS-v2 zero-shot greeting before any batch generation.
- Commands: `src\generate_xtts_arabic_test.py`, `scripts\validate_xtts_audio.py`, `scripts\validate_xtts_arabic_sample.py`.
- Result: Initial WAV generated at `outputs\xtts\arabic\ar_clone_01_greeting.wav`; PCM16 mono, 24000 Hz, duration 5.846 s, generation time 18.591 s, RTF 3.180, RMS 0.204839, peak 0.950012, clipping 0, silence ratio 0.172418.
- ASR/WER: Faster Whisper Small, language `ar`, probability 1.0000, WER 11.11%; target WER <= 15% passed.
- Failure: The generation command returned nonzero through PowerShell because a torchaudio deprecation warning was emitted on stderr, but the sidecar and audio validation show successful generation.
- Fix: Continued with explicit audio validation using `soundfile`; no TorchCodec decoding was used in validation.
- Evidence: `evidence\terminal_logs\xtts_arabic_initial_generation.txt`, `xtts_arabic_initial_audio_validation.txt`, `xtts_arabic_initial_wer.txt`, `results\xtts_arabic_initial_test.csv`.
- Decision: Continue to reference comparison; do not treat Whisper WER as native-pronunciation proof.

## 2026-07-17 - XTTS-v2 Phase 6: Reference Strategy Comparison

- Objective: Compare short, standard, long, and multi-reference XTTS cloning on the same Arabic greeting with seed 42.
- Command: `src\generate_xtts_arabic.py --mode reference-comparison --device cpu --seed 42`, followed by `scripts\evaluate_xtts_arabic.py --mode reference-comparison`.
- Result: Short WER 11.11% / RTF 3.094; standard WER 33.33% / RTF 3.176; long WER 11.11% / RTF 3.058; multi-reference WER 0.00% / RTF 3.218. All had clipping 0 and Arabic probability 1.0000.
- Failure: None in generation or evaluation. Log contains Windows/cmd mojibake for Arabic text, while CSV/JSON are UTF-8.
- Evidence: `evidence\terminal_logs\xtts_arabic_reference_comparison.txt`, `results\xtts_arabic_reference_comparison.csv`.
- Decision: Automatic reference winner is `multi` by lowest WER, zero clipping, and no generation failure. This is not a human-quality winner.

## 2026-07-17 - XTTS-v2 Phase 7: Parameter Matrix

- Objective: Run a small CPU-friendly greeting-only parameter matrix using the automatic `multi` reference strategy.
- Command: `src\generate_xtts_arabic.py --mode parameter-tests --device cpu --reference-strategy multi --seed 42`, followed by `scripts\evaluate_xtts_arabic.py --mode parameter-tests`.
- Result: Config A WER 22.22% / RTF 3.228; Config B WER 0.00% / RTF 3.237; Config C WER 0.00% / RTF 3.198; Config D installed defaults WER 22.22% / RTF 3.130. All successful configurations had clipping 0.
- Failure: None in generation or evaluation. Logs include a torchaudio deprecation warning from Coqui internals.
- Evidence: `evidence\terminal_logs\xtts_arabic_parameter_tests.txt`, `results\xtts_arabic_parameter_tests.csv`, `results\xtts_arabic_parameter_evaluation.csv`.
- Decision: Automatic parameter winner is Config C: temperature 0.65, top_k 40, top_p 0.80, repetition_penalty 5.0, length_penalty 1.0, speed 1.0. This is not a human naturalness/similarity decision.

## 2026-07-17 - XTTS-v2 Phase 8-13: Final Generation, Evaluation, Comparison, Review Prep

- Objective: Generate all five Arabic XTTS-v2 samples, evaluate them, compare against Chatterbox, and prepare native-review and completion evidence.
- Commands: `src\generate_xtts_arabic.py --mode final --device cpu --reference-strategy multi --seed 42 --configuration-id C --temperature 0.65 --top-k 40 --top-p 0.80 --repetition-penalty 5.0 --length-penalty 1.0 --speed 1.0`; `scripts\evaluate_xtts_arabic.py --mode final`; `scripts\compare_arabic_models.py`.
- Result: XTTS-v2 final five-sample run succeeded: average WER 5.36%, median WER 7.69%, max WER 10.00%, WER pass count 5/5, clipping pass count 5/5, failed generations 0, average RTF 3.225.
- Chatterbox comparison: Chatterbox average WER 20.13%, pass count 2/5, clipping pass 5/5, average RTF 12.105. XTTS-v2 is the automatic winner under the requested rules.
- Failure: The XTTS summary initially read torch from the ASR environment; fixed the evaluator to parse the XTTS import-verification log, which records torch 2.8.0+cpu.
- Evidence: `evidence\terminal_logs\xtts_arabic_final_generation.txt`, `xtts_arabic_final_evaluation.txt`, `arabic_model_comparison.txt`; `results\xtts_arabic_evaluation.csv`, `results\arabic_model_comparison.csv`; `evidence\result_snapshots\xtts\arabic\xtts_arabic_summary.json`, `evidence\result_snapshots\arabic_model_comparison_summary.json`.
- Decision: XTTS-v2 is the automatic metric winner, but final production suitability remains blocked on native Arabic review, licensing review, and GPU benchmarking. Website integration and fine-tuning were not started.
