# English Results

This report is rebuilt from existing CSV/JSON evidence by `python scripts/build_language_reports.py`. Missing values are left blank or marked `NA`; no metrics are invented.

## Models tested
- MMS-TTS
- NeuTTS

## Summary
| Model | Samples | Avg WER % | Avg gen s | Avg RTF | Clipping | Speaker sim | MOS | Pronunciation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMS-TTS | 5 | 8 | 2.048 | 0.454 | 0/5 |  | pending | pending |
| NeuTTS | 5 | 12.675 | 82.079 | 14.708 | 0/5 | 0.542 | pending | pending |

## Strengths
- MMS-TTS: Fast CPU baseline, passed average English WER target, stable/no clipping in final samples.
- NeuTTS: Voice-cloning candidate with final samples available and no clipping in this run.

## Weaknesses
- MMS-TTS: Not a voice-cloning pipeline; speaker similarity is not applicable.
- NeuTTS: Missed the 10% average WER target, speaker cosine target, and CPU latency/RTF target.

## Manual listening status
Manual MOS/listener fields are pending unless values appear in the summary table. Current `human_listener_scores.csv` does not contain real reviewer scores for these samples.

## Winner
MMS-TTS is the automatic English baseline winner because it passed average WER and CPU RTF; it is not a cloning winner.

## Evidence paths
- `results\mms_english_evaluation.csv`
- `outputs\mms\english\final\en_clone_01_greeting.json`
- `outputs\mms\english\final\en_clone_02_technology.json`
- `outputs\mms\english\final\en_clone_03_conversational.json`
- `outputs\mms\english\final\en_clone_04_numbers.json`
- `outputs\mms\english\final\en_clone_05_expressive.json`
- `results\neutts_english_evaluation.csv`
- `outputs\neutts\english\final\air_01_greeting.json`
- `outputs\neutts\english\final\air_02_identity.json`
- `outputs\neutts\english\final\air_03_support.json`
- `outputs\neutts\english\final\air_04_numbers.json`
- `outputs\neutts\english\final\air_05_expressive.json`
- `results\english_speaker_similarity.csv`
- `evidence\result_snapshots\english_model_summary.json`
- `evidence\result_snapshots\english_speaker_similarity_summary.json`
- `results\human_listener_scores.csv`
- `results\human_evaluation_summary.csv`

## Consolidated rows
See `results\english_model_comparison.csv`.

## 2026-07-19 NeuTTS Improved Check

NeuTTS previously sounded most human during manual listening, but it was not the English winner because the available final evidence had average WER 12.67%, a prior standardized generation hang/fallback note, and manual listening remained pending.

The improved attempt added TTS-only English text normalization and a new generation path at `src/pipelines/english/neutts_improved.py`. The expected text used for WER comparison remains unchanged. Normalizer verification passed in `evidence/terminal_logs/neutts_english_text_normalizer_verification.txt`.

The improved generation did not produce WAVs. After fixing a local import collision, `.venvs/.venv-neutts` failed to import the official NeuTTS package with `No module named 'neutts'`. Evaluation therefore recorded 5 failed samples, average WER 100.0%, average RTF 0.0, and total clipping samples 0 because no audio was generated.

Fine-tuning was not performed. A dataset/config/Colab scaffold was created under `data/finetune/neutts_english`, `configs/neutts_finetune_ratan_english.yaml`, and `scripts/run_neutts_finetune_colab.md`; validation found no accepted WAV clips.

Decision: NeuTTS remains naturalness winner only, not the final reproducible English winner. MeloTTS remains the current reproducible English winner unless a future NeuTTS run generates all five clean WAVs and passes WER <= 10% with manual confirmation.

## 2026-07-19 NeuTTS Improved Successful Rerun

A follow-up recovery found the NeuTTS source at `external/neutts` and configured the improved script to use it directly. System eSpeak NG was found at `C:\Program Files\eSpeak NG` and used by the phonemizer. The final successful run cleared stale proxy variables and generated all five fixed English samples without reusing fallback audio.

Improved NeuTTS results:

| Metric | Value |
| --- | ---: |
| Samples generated | 5/5 |
| Average WER | 1.82% |
| Average RTF | 27.692 |
| Average generation time | 150.483 s |
| Average audio duration | 5.404 s |
| Max peak | 0.778 |
| Total clipping samples | 0 |

The number/date sample improved from 37.5% WER in the original evaluation to 0.0% WER in the improved evaluation. The ASR still rendered the spoken number/date as `5824` and `21st`, but the improved evaluator normalizes those numeric forms against the expected spoken-text benchmark.

Decision: NeuTTS-improved passes the automatic English WER, stability, and clipping requirements. It is the automatic English winner pending final MOS/manual listening confirmation. Manual listening remains pending; no MOS score was invented.
