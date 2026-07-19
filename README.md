# Infinia Voice Case Study — Open-Source Multilingual TTS Benchmark

**Languages:** English, Arabic, Hindi  
**Focus:** Naturalness, latency, intelligibility, reproducibility, and engineering stability  

This repository contains my submission for the Infinia AI Engineer (Voice) take-home case study.  
The goal was to build and evaluate open-source Text-to-Speech pipelines for English, Arabic, and Hindi using real generated audio, benchmark numbers, logs, and failure evidence.

---

## Live Demo

**Streamlit Dashboard:**  
[Open the demo dashboard](PASTE_STREAMLIT_LINK_HERE)

**GitHub Repository:**  
https://github.com/ratanjyoti/Voice_clone

The dashboard includes final results, playable audio clips, benchmark tables, evidence logs, generated audio review, and failure analysis. The Streamlit app has dedicated pages such as **Overview**, **Final Results**, **Audio Clips**, **Evidence & Logs**, **Generated Audio**, and **Reports**. :contentReference[oaicite:0]{index=0}

---

## One-Paragraph Summary

I evaluated multiple open-source TTS and voice-cloning models across English, Arabic, and Hindi. The final recommendation is a **language-router architecture** because no single model performed best across all three languages. **MeloTTS** was selected as the English winner because it gave the best balance of intelligibility and naturalness. **XTTS-v2** gave the best Arabic WER, while **Meta MMS-TTS** remains the faster CPU fallback. Hindi remained unresolved: MMS was stable but had poor pronunciation, Chatterbox performed better but was too slow, and IndicF5 required multiple fixes but showed prompt/beginning-language leakage.

---

## Final Recommendation

| Language | Recommended Pipeline | Reason |
|---|---|---|
| **English** | **MeloTTS** | Best balance of naturalness, clean audio, and WER. |
| **Arabic** | **XTTS-v2 for quality, MMS for CPU speed** | XTTS-v2 had the best WER; MMS was faster and more stable on CPU. |
| **Hindi** | **No production-ready winner** | MMS was stable but poorly pronounced; Chatterbox was better but too slow; IndicF5 had prompt leakage. |

---

## Key Results

| Language | Model | WER | RTF | Status | Notes |
|---|---|---:|---:|---|---|
| **English** | **MeloTTS** | **9.67%** | **1.428** | Winner | Clean audio, no clipping, best English quality tradeoff |
| English | Meta MMS-TTS | 8.0% | <0.3 | Speed baseline | Very fast but robotic |
| English | NeuTTS | 12.67% | Not finalized | Rejected | Failed WER target and final run was unstable |
| **Arabic** | XTTS-v2 | 5.36% | Poor on CPU | Quality winner | Strong WER, but CPU latency was high |
| Arabic | Meta MMS-TTS | ~12% | ~0.28 | CPU fallback | Fast but robotic |
| Arabic | Chatterbox | ~20% | Slow | Rejected | Poor intelligibility |
| **Hindi** | MMS-TTS | 52.45% | 0.38–0.50 | Stable baseline only | Fast but poor pronunciation |
| Hindi | Chatterbox | 36.78% | 8–39 | Best tested Hindi WER | Too slow and some samples clipped |
| Hindi | IndicF5 | 50% greeting | 41.778 | Rejected | Prompt/beginning-language issue |

---

## What This Repository Contains

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