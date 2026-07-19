# Infinia Voice Case Study — Open-Source Multilingual TTS Benchmark

**Languages:** English, Arabic, Hindi  
**Focus:** Naturalness, latency, intelligibility, reproducibility, stability, and honest failure analysis  

This repository is my submission for the **Infinia AI Engineer (Voice)** take-home case study. The goal was to build and evaluate open-source Text-to-Speech pipelines for English, Arabic, and Hindi using **real generated audio, benchmark numbers, logs, and failure evidence**.

---

## Live Demo

**Streamlit Dashboard:**  
[https://voiceclone-multilingual.streamlit.app/](https://voiceclone-multilingual.streamlit.app/)

**GitHub Repository:**  
[https://github.com/ratanjyoti/Voice_clone](https://github.com/ratanjyoti/Voice_clone)

The dashboard is the easiest way to review the work. It includes final results, playable audio clips, benchmark tables, evidence logs, generated audio review, and failure analysis.

Recommended dashboard pages to check first:

1. **Overview**
2. **Final Results**
3. **Audio Clips**
4. **Evidence & Logs**
5. **Generated Audio**
6. **Reports**

---

## One-Paragraph Summary

I evaluated multiple open-source TTS and voice-cloning models across **English, Arabic, and Hindi**. The final recommendation is a **language-router architecture**, because no single model performed best across all three languages. **MeloTTS** was selected as the English winner because it gave the best balance of intelligibility, clean audio, and naturalness. **XTTS-v2** gave the best Arabic WER, while **Meta MMS-TTS** remains the faster CPU fallback. Hindi remained unresolved: MMS was stable but had poor pronunciation, Chatterbox performed better but was too slow, and IndicF5 required multiple fixes but showed prompt/beginning-language leakage.

---

## Final Recommendation

| Language | Recommended Pipeline | Reason |
|---|---|---|
| **English** | **MeloTTS** | Best balance of naturalness, clean audio, and intelligibility. |
| **Arabic** | **XTTS-v2 for quality, MMS-TTS for CPU speed** | XTTS-v2 had the best Arabic WER; MMS-TTS was faster and more stable on CPU. |
| **Hindi** | **No production-ready winner** | MMS-TTS was stable but poorly pronounced; Chatterbox was better but too slow; IndicF5 had prompt/beginning-language leakage. |

---

## Key Benchmark Results

| Language | Model | WER | RTF | Status | Notes |
|---|---|---:|---:|---|---|
| **English** | **MeloTTS** | **9.67%** | **1.428** | **Winner** | Clean audio, no clipping, best English quality tradeoff |
| English | Meta MMS-TTS | 8.0% | <0.3 | Speed baseline | Very fast, but robotic |
| English | NeuTTS | 12.67% | Not finalized | Rejected | Failed WER target and final standardized run was unstable |
| **Arabic** | XTTS-v2 | 5.36% | Poor on CPU | Quality winner | Strong WER, but CPU latency was high |
| Arabic | Meta MMS-TTS | ~12% | ~0.28 | CPU fallback | Fast and stable, but robotic |
| Arabic | Chatterbox | ~20% | Slow | Rejected | Poor intelligibility |
| **Hindi** | MMS-TTS | 52.45% | 0.38–0.50 | Stable baseline only | Fast, but poor pronunciation |
| Hindi | Chatterbox | 36.78% | 8–39 | Best tested Hindi WER | Too slow and some samples clipped |
| Hindi | IndicF5 | 50% greeting | 41.778 | Rejected | Prompt/beginning-language issue |

> Note: Human MOS collection is prepared through the review workflow, but final MOS should not be treated as complete until real listener ratings are collected.

---

## Evaluation Metrics

The case study was evaluated against the requested metrics.

| Metric | How it was measured | Target |
|---|---|---|
| **Naturalness / MOS** | Human listening review workflow prepared; manual scores pending | ≥4.0 / 5 |
| **Speaker Similarity** | SpeechBrain ECAPA-TDNN cosine similarity where voice cloning was applicable | ≥0.75 |
| **Latency / Runtime** | Batch full-clip generation time for test sentences | <2s full clip preferred |
| **RTF** | `generation_time / audio_duration` | ≤0.5 |
| **WER** | Generated audio → faster-whisper ASR → compare to input text | ≤10% preferred |
| **Technical Validation** | Peak, RMS, clipping, duration, silence ratio, NaN/Inf checks | No clipping / valid waveform |

---

## Models Tested

| Model | Languages Tested | Purpose | Final Outcome |
|---|---|---|---|
| **Meta MMS-TTS** | English, Arabic, Hindi | Fast multilingual baseline | Very stable and fast, but robotic; weak Hindi pronunciation |
| **MeloTTS** | English | English naturalness and intelligibility candidate | **Final English winner** |
| **NeuTTS** | English | English naturalness / cloning candidate | Failed WER target and final standardized generation became unstable |
| **XTTS-v2** | Arabic | Multilingual voice cloning and Arabic quality candidate | Best Arabic WER, but slow on CPU |
| **Chatterbox** | Hindi, Arabic | Voice cloning / naturalness candidate | Better Hindi than MMS-TTS, but too slow and some clipping |
| **IndicF5** | Hindi | Hindi-specialized voice-conditioned model | Made runnable after fixes, but rejected due to prompt/beginning-language issue |
| **OpenVoice** | Explored | Voice conversion / cloning direction | Explored, but not selected as a final benchmark winner |

---

## Repository Structure

```text
src/
  Model generation pipelines and implementation code.

scripts/
  Generation, evaluation, validation, speaker-similarity, and report-building scripts.

outputs/
  Generated WAV files and JSON sidecars for tested models.

results/
  Raw and summary CSV files containing WER, RTF, clipping, generation time, and comparison results.

evidence/
  Terminal logs, JSON result snapshots, audio manifests, and debugging evidence.

docs/
  Final report, language-specific reports, engineering journey, and recommendation notes.

languages/
  Per-language workspaces for English, Hindi, Arabic, and shared files.

dashboard/
  Streamlit dashboard used to display results, audio clips, evidence logs, and reports.

data/
  Test sentences, reference audio, transcripts, and consent notes where applicable.
```

---

## Important Files to Review

### Final Reports

| File | Purpose |
|---|---|
| `docs/final_report.md` | Main technical case-study write-up |
| `docs/engineering_journey.md` | Chronological engineering log with failures and fixes |
| `docs/english_results.md` | English-specific results and reasoning |
| `docs/arabic_results.md` | Arabic-specific results and reasoning |
| `docs/hindi_results.md` | Hindi-specific results and failure analysis |
| `docs/final_recommendation.md` | Final language-router recommendation |

### Main Result Tables

| File | Purpose |
|---|---|
| `results/final_three_language_benchmark.csv` | Master result table across English, Arabic, and Hindi |
| `results/summary/english_model_comparison.csv` | English model comparison |
| `results/summary/melotts_english_summary.csv` | MeloTTS English summary |
| `results/raw/melotts_english_evaluation.csv` | Raw MeloTTS English WER/RTF rows |
| `results/mms_english_evaluation.csv` | MMS English benchmark results |
| `results/neutts_english_evaluation.csv` | NeuTTS English benchmark/fallback evidence |
| `results/xtts_arabic_evaluation.csv` | XTTS-v2 Arabic WER and runtime results |
| `results/chatterbox_arabic_evaluation.csv` | Chatterbox Arabic results |
| `results/mms_hindi_evaluation.csv` | MMS Hindi results |
| `results/chatterbox_hindi_evaluation.csv` | Chatterbox Hindi results |
| `results/indicf5_hindi_initial_test.csv` | IndicF5 one-sample Hindi diagnostic result |
| `results/hindi_greeting_model_comparison.csv` | Hindi greeting comparison across MMS, Chatterbox, IndicF5 |

### Evidence

| Folder | Purpose |
|---|---|
| `evidence/terminal_logs/` | Full stdout/stderr logs from installs, generation, evaluation, and failures |
| `evidence/result_snapshots/` | JSON snapshots for successful and failed runs |
| `evidence/audio_manifest/` | Audio inventory and generated-clip metadata |

### Audio Output Folders

| Folder | Description |
|---|---|
| `outputs/english/melotts/final/` | Final English MeloTTS clips |
| `outputs/english/mms/final/` or `outputs/mms/english/final/` | English MMS baseline clips |
| `outputs/arabic/xtts/` | Arabic XTTS-v2 clips |
| `outputs/chatterbox/arabic/` | Arabic Chatterbox clips |
| `outputs/mms/hindi/final/` | Hindi MMS baseline clips |
| `outputs/chatterbox/hindi/final/` | Hindi Chatterbox clips |
| `outputs/indicf5/hindi/` | IndicF5 Hindi diagnostic output and sidecar |

---

## How to View the Results

### Option 1: Use the Streamlit Dashboard

Open:

[https://voiceclone-multilingual.streamlit.app/](https://voiceclone-multilingual.streamlit.app/)

Suggested review order:

1. **Overview** — final model choices and key tradeoffs
2. **Final Results** — benchmark CSV tables
3. **Audio Clips** — play selected audio samples
4. **Evidence & Logs** — inspect raw logs and JSON snapshots
5. **Reports** — read the final written analysis

### Option 2: Inspect the Repository Locally

```powershell
git clone https://github.com/ratanjyoti/Voice_clone
cd Voice_clone
```

Open the report:

```powershell
notepad docs\final_report.md
```

Open result CSVs:

```powershell
dir results
```

Play audio clips:

```powershell
Start-Process outputs\english\melotts\final
Start-Process outputs\arabic\xtts
Start-Process outputs\chatterbox\hindi\final
```

---

## Reproducing the Evaluation

The full benchmark uses multiple model-specific environments because the open-source TTS ecosystem has conflicting Python and PyTorch requirements.

### Evaluation Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

### Speaker Similarity Environment

```powershell
py -3.11 -m venv .venv-speaker-similarity
.\.venv-speaker-similarity\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-speaker-similarity\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-speaker-similarity\Scripts\python.exe -m pip install speechbrain soundfile pandas numpy scipy
```

### Main Evaluation Commands

```powershell
.\.venv\Scripts\python.exe -u scripts\evaluate_final_samples.py --target all
.\.venv-speaker-similarity\Scripts\python.exe -u scripts\evaluate_speaker_similarity.py
python scripts\build_language_reports.py
python scripts\build_language_workspaces.py
```

---

## Language-Specific Commands

### English

```powershell
.\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language english
.\.venvs\.venv-melotts\Scripts\python.exe -u src\pipelines\english\melotts.py
.\.venv\Scripts\python.exe -u src\evaluation\melotts_english_eval.py
```

### Hindi

```powershell
.\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language hindi
.\.venv\Scripts\python.exe -u scripts\generate_final_chatterbox_hindi.py
.\.venvs\.venv-indicf5\Scripts\python.exe -u src\pipelines\hindi\indicf5.py
```

### Arabic

```powershell
.\.venvs\.venv-xtts-arabic\Scripts\python.exe -u src\pipelines\arabic\xtts.py
.\.venvs\.venv-xtts-arabic\Scripts\python.exe -u scripts\evaluate_xtts_arabic.py
.\.venv\Scripts\python.exe -u src\pipelines\arabic\chatterbox.py
.\.venv\Scripts\python.exe -u scripts\evaluate_chatterbox_arabic.py
```

Some commands may require model weights, Hugging Face access, or local environment setup. The repository includes generated outputs and logs so the benchmark can still be reviewed without rerunning every model.

---

## What Worked

### English

- MeloTTS produced the best overall English result.
- It passed the WER target with `9.67%`.
- It produced clean non-clipped audio.
- It sounded more natural than MMS-TTS.
- MMS-TTS remained the fastest fallback.

### Arabic

- XTTS-v2 achieved the strongest Arabic WER at `5.36%`.
- It was the best quality candidate.
- CPU runtime was too slow for production latency.
- MMS-TTS remained a stable CPU fallback.

### Hindi

- MMS-TTS was technically stable and fast.
- Chatterbox achieved the best tested Hindi WER among the attempted models.
- IndicF5 was successfully made runnable after resolving multiple environment/model-loading issues.

---

## Where It Breaks

### English Weaknesses

- MeloTTS was slower than MMS-TTS.
- Number-heavy English text caused a high WER sample.
- NeuTTS had naturalness potential but final standardized generation was unstable.

### Arabic Weaknesses

- XTTS-v2 passed WER but was not CPU practical.
- Chatterbox Arabic failed intelligibility.
- MMS-TTS Arabic was fast but robotic.

### Hindi Weaknesses

- MMS-TTS Hindi was fast but had poor pronunciation.
- Chatterbox Hindi was better than MMS-TTS but extremely slow and clipped on some samples.
- IndicF5 generated audio but showed prompt/beginning-language leakage and poor WER.
- No Hindi model met the final production-quality target in this benchmark.

---

## Engineering Challenges Documented

This project includes the “messy parts” rather than hiding them.

### Dependency Conflicts

Several models required conflicting Python, PyTorch, Transformers, and audio dependencies. The solution was a multi-venv approach instead of forcing all models into one environment.

### IndicF5 Gated Access

IndicF5 initially failed with Hugging Face `401 Unauthorized`. After access was approved, generation proceeded.

### PyTorch Meta-Device Error

IndicF5 then failed with a CPU/meta-device error involving Vocos and `torch.compile`. The fix was a CPU-safe eager wrapper that preserved `_orig_mod` checkpoint key structure while avoiding actual CPU compilation.

### Prompt Leakage

IndicF5 generated a technically valid Hindi WAV, but manual listening showed unrelated or non-Hindi speech at the beginning. The corrected design includes reference validation, early-segment ASR, prompt-leakage detection, and evidence-based trimming only when justified.

---

## Ground Rules Followed

- Core speech generation used open-source models.
- No closed API was used for final TTS generation.
- Evaluation used reproducible scripts and saved CSV/JSON/log evidence.
- Voice-cloning references were based on self-owned or consented audio.
- Failed runs were documented instead of hidden.
- AI coding assistance was used, but results were produced through real model runs, audio files, benchmark scripts, and logs.

---

## Known Limitations

- Human MOS scores are prepared but still require real listener review.
- Some model environments are heavy and may not reproduce easily on Streamlit Cloud.
- Streamlit deployment is intended mainly for result viewing, not rerunning every model.
- Hindi quality remains unresolved.
- GPU benchmarking is still needed for fair latency comparison of XTTS-v2, Chatterbox, and IndicF5.
- Model licenses and redistribution rights should be rechecked before production use.

---

## Future Improvements

1. **Dockerize each model pipeline**  
   Separate containers for MMS, MeloTTS, XTTS-v2, Chatterbox, and IndicF5 would avoid dependency conflicts.

2. **GPU benchmarking**  
   Retest XTTS-v2, Chatterbox, and IndicF5 on T4/L4/A10 GPUs.

3. **Streaming synthesis**  
   Implement chunked generation to reduce time-to-first-audio below 500 ms.

4. **Better Hindi reference data**  
   Record a clean 8–12 second Hindi-only reference clip with an exact transcript.

5. **Hindi fine-tuning**  
   Fine-tune a VITS/FastPitch-style model on clean Hindi speech data.

6. **ASR cross-checking**  
   Use IndicWhisper / IndicConformer for Hindi and Arabic-specific ASR for Arabic WER validation.

---

## Submission Checklist

This repository includes:

- [x] English, Arabic, and Hindi pipelines
- [x] Open-source TTS generation models
- [x] Generated audio clips
- [x] WER and RTF numbers
- [x] Clipping/audio validation
- [x] Speaker similarity workflow
- [x] Evidence logs
- [x] JSON snapshots
- [x] Failure documentation
- [x] Streamlit dashboard
- [x] Final technical report
- [x] Per-language result reports

Still pending:

- [ ] Real human MOS scores from multiple listeners
- [ ] GPU latency benchmark
- [ ] Production-ready Hindi winner

---

## Final Conclusion

This project demonstrates a realistic open-source multilingual TTS benchmark under practical engineering constraints. English reached the strongest result with **MeloTTS**. Arabic achieved the best intelligibility with **XTTS-v2**, but CPU latency remained a limitation. Hindi remained unresolved despite testing MMS-TTS, Chatterbox, and IndicF5.

The final recommendation is a **language-router system** with strict validation gates for WER, clipping, prompt leakage, runtime, and manual listening quality. The repository includes both successful outputs and failed runs, showing the actual engineering process instead of only polished results.