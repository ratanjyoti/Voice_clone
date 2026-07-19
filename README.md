# Infinia Voice Case Study

Recommendation: use MMS-TTS as the fast English baseline, use XTTS-v2 as the best automatic Arabic WER candidate with clear CPU latency and speaker-similarity caveats, and treat Hindi cloning as not production-ready from this benchmark because both MMS-TTS Hindi and Chatterbox Hindi missed the 10% WER target. Human MOS is prepared but still pending real listener scores, so final production approval should wait for listener review.

## 1. Project objective
Evaluate English, Hindi, and Arabic TTS/voice-cloning pipelines for WER, batch latency, RTF, clipping/silence, speaker cosine similarity, and human listening review readiness.

## 2. Final recommendation table
See `results/final_three_language_benchmark.csv`.

Short version:

| Language | Recommended pipeline | Reason |
| --- | --- | --- |
| English | MMS-TTS baseline | Passed average WER target and CPU RTF target; not a cloning pipeline. |
| Hindi | No production winner | Chatterbox is the better tested cloning candidate but missed WER/latency and had clipping. |
| Arabic | XTTS-v2 quality candidate | Passed Arabic WER target but failed CPU latency/RTF and speaker cosine target. |

## 3. Hardware
All final automatic metrics were run on Windows CPU in this repository workspace. CUDA was not available for the final benchmark runs.

## 4. Model versions
- MMS-TTS: `facebook/mms-tts-eng`, `facebook/mms-tts-hin`, `facebook/mms-tts-ara`
- NeuTTS: `neuphonic/neutts-air` with `neuphonic/neucodec`
- Chatterbox: `ResembleAI/chatterbox multilingual`
- XTTS-v2: Coqui XTTS-v2 Arabic pipeline
- Speaker similarity: SpeechBrain ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb`
- ASR/WER: Faster Whisper Small on CPU/int8

## 5. Folder structure
- `src/`: generation implementations used during experiments
- `scripts/`: verification, final generation, evaluation, and packaging scripts
- `data/test_sentences/`: fixed benchmark sentence JSON files
- `outputs/`: generated WAV files and sidecar metadata
- `results/`: final CSV metrics
- `evidence/terminal_logs/`: command logs
- `evidence/result_snapshots/`: JSON summaries
- `report/`: final write-up

## 6. Environment setup
Create and install the speaker-similarity environment:

```powershell
py -3.11 -m venv .venv-speaker-similarity
.\.venv-speaker-similarity\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-speaker-similarity\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-speaker-similarity\Scripts\python.exe -m pip install speechbrain soundfile pandas numpy scipy
```

The main `.venv` contains MMS, Chatterbox, Faster Whisper, JiWER, NumPy, and SoundFile. NeuTTS uses `.venv-neutts` plus the local `external/neutts` source path.

## 7. English commands
```powershell
.\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language english
$env:PHONEMIZER_ESPEAK_LIBRARY='C:\Program Files\eSpeak NG\libespeak-ng.dll'
$env:ESPEAK_DATA_PATH='C:\Program Files\eSpeak NG\espeak-ng-data'
.\.venv-neutts\Scripts\python.exe -u scripts\generate_final_neutts_english.py
```

Note: the final NeuTTS fixed-sentence retry hung during model/codec load. The final NeuTTS folder therefore reuses the previously completed five-sample NeuTTS Air evaluation and is labelled as a fallback in sidecars.

## 8. Hindi commands
```powershell
.\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language hindi
.\.venv\Scripts\python.exe -u scripts\generate_final_chatterbox_hindi.py
```

## 9. Arabic commands
```powershell
.\.venv-xtts-arabic\Scripts\python.exe -u src\generate_xtts_arabic.py
.\.venv-xtts-arabic\Scripts\python.exe -u scripts\evaluate_xtts_arabic.py
.\.venv\Scripts\python.exe -u src\generate_chatterbox_arabic.py
.\.venv\Scripts\python.exe -u scripts\evaluate_chatterbox_arabic.py
```

## 10. Evaluation commands
```powershell
.\.venv-speaker-similarity\Scripts\python.exe -u scripts\verify_speaker_embedding_model.py
.\.venv-speaker-similarity\Scripts\python.exe -u scripts\evaluate_speaker_similarity.py
.\.venv\Scripts\python.exe -u scripts\evaluate_final_samples.py --target all
.\.venv\Scripts\python.exe scripts\summarize_human_evaluation.py
```

## 11. Results
- Speaker similarity: `results/all_language_speaker_similarity.csv`
- English/Hindi WER: `results/mms_english_evaluation.csv`, `results/neutts_english_evaluation.csv`, `results/mms_hindi_evaluation.csv`, `results/chatterbox_hindi_evaluation.csv`
- Arabic WER: `results/xtts_arabic_evaluation.csv`, `results/chatterbox_arabic_evaluation.csv`
- Latency: `results/standardized_latency_benchmark.csv`
- Master table: `results/final_three_language_benchmark.csv`

## 12. Audio samples
Final or review-ready samples are under:
- `outputs/mms/english/final`
- `outputs/neutts/english/final`
- `outputs/mms/hindi/final`
- `outputs/chatterbox/hindi/final`
- `outputs/xtts/arabic`
- `outputs/chatterbox/arabic`
- `outputs/human_review/*`

## 13. Failure modes
- Chatterbox Arabic failed the final WER target.
- XTTS Arabic passed WER but failed CPU latency/RTF and speaker-cosine targets.
- MMS was fast but did not provide cloning.
- NeuTTS fixed final generation hung during model/codec loading; prior five-sample NeuTTS Air results were used as fallback evidence.
- Chatterbox Hindi was very slow on CPU and clipped on some samples.

## 14. Licensing
Model licenses and reference-audio redistribution rights must be verified before production or external redistribution. Consent notes are under `data/reference_audio/consent.txt` where available.

## 15. Remaining work
Collect real MOS scores from qualified listeners, replace the NeuTTS fallback with a successful fixed-sentence run, evaluate an Indic ASR alternative for Hindi, and retest latency on GPU or deployment hardware.
## Results by Language

- English: docs\english_results.md
- Hindi: docs\hindi_results.md
- Arabic: docs\arabic_results.md
- Final recommendation: docs\final_recommendation.md

Rebuild these summaries with:

```powershell
python scripts\build_language_reports.py
```
## Language Workspaces

Use these folders when changing or reviewing one language at a time:

- English workspace: languages\english\README.md
- Hindi workspace: languages\hindi\README.md
- Arabic workspace: languages\arabic\README.md
- Shared workspace: languages\shared\README.md

Each workspace contains `files.csv` plus grouped Markdown indexes for code, data, outputs, results, docs, and evidence. Rebuild the indexes with:

```powershell
python scripts\build_language_workspaces.py
```

