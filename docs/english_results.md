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
