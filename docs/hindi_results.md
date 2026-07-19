# Hindi Results

This report is rebuilt from existing CSV/JSON evidence by `python scripts/build_language_reports.py`. Missing values are left blank or marked `NA`; no metrics are invented.

## Models tested
- Chatterbox
- IndicF5
- MMS-TTS

## Summary
| Model | Samples | Avg WER % | Avg gen s | Avg RTF | Clipping | Speaker sim | MOS | Pronunciation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chatterbox | 5 | 36.783 | 108.916 | 19.015 | 2/5 | 0.738 | pending | pending |
| IndicF5 | 1 | 50 | 297.668 | 41.778 | 0/1 |  | pending | pending |
| MMS-TTS | 5 | 52.45 | 2.184 | 0.422 | 0/5 |  | pending | pending |

## Strengths
- Chatterbox: Best Hindi WER among the five-sample Hindi models tested and provides voice cloning.
- IndicF5: Hindi-specialized model became runnable after authenticated access and CPU compatibility fixes; one sample had no clipping.
- MMS-TTS: Fastest Hindi baseline with no clipping in final samples.

## Weaknesses
- Chatterbox: Missed Hindi WER target, very slow on CPU, and some final samples clipped.
- IndicF5: Only one greeting sample was evaluated; WER was worse than Chatterbox and CPU RTF was very high.
- MMS-TTS: Missed Hindi WER target and is not a cloning pipeline.

## Manual listening status
Manual MOS/listener fields are pending unless values appear in the summary table. Current `human_listener_scores.csv` does not contain real reviewer scores for these samples.

## Winner
No production winner. Chatterbox is the best tested Hindi cloning candidate, but it fails WER/latency and has clipping; MMS is faster but not cloning; IndicF5 should not expand beyond the initial sample yet.

## Evidence paths
- `results\mms_hindi_evaluation.csv`
- `outputs\mms\hindi\final\hi_clone_01_greeting.json`
- `outputs\mms\hindi\final\hi_clone_02_technology.json`
- `outputs\mms\hindi\final\hi_clone_03_conversational.json`
- `outputs\mms\hindi\final\hi_clone_04_numbers.json`
- `outputs\mms\hindi\final\hi_clone_05_expressive.json`
- `results\chatterbox_hindi_evaluation.csv`
- `outputs\chatterbox\hindi\final\hi_clone_01_greeting.json`
- `outputs\chatterbox\hindi\final\hi_clone_02_technology.json`
- `outputs\chatterbox\hindi\final\hi_clone_03_conversational.json`
- `outputs\chatterbox\hindi\final\hi_clone_04_numbers.json`
- `outputs\chatterbox\hindi\final\hi_clone_05_expressive.json`
- `results\indicf5_hindi_initial_test.csv`
- `results\hindi_speaker_similarity.csv`
- `results\hindi_greeting_model_comparison.csv`
- `evidence\result_snapshots\hindi_model_summary.json`
- `evidence\result_snapshots\hindi_speaker_similarity_summary.json`
- `evidence\result_snapshots\indicf5_hindi_initial_generation_success.json`
- `results\human_listener_scores.csv`
- `results\human_evaluation_summary.csv`

## Consolidated rows
See `results\hindi_model_comparison.csv`.
