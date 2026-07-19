# Arabic Results

This report is rebuilt from existing CSV/JSON evidence by `python scripts/build_language_reports.py`. Missing values are left blank or marked `NA`; no metrics are invented.

## Models tested
- Chatterbox
- XTTS-v2

## Summary
| Model | Samples | Avg WER % | Avg gen s | Avg RTF | Clipping | Speaker sim | MOS | Pronunciation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chatterbox | 5 | 20.126 | 89.41 | 12.104 | 0/5 | 0.673 | pending | pending |
| XTTS-v2 | 5 | 5.357 | 22.637 | 3.225 | 0/5 | 0.618 | pending | pending |

## Strengths
- Chatterbox: Arabic voice-cloning candidate with higher speaker cosine than XTTS-v2 in the available summary, though still below the 0.75 target.
- XTTS-v2: Best Arabic automatic WER result and no clipping in final samples.

## Weaknesses
- Chatterbox: Failed Arabic WER target and was slower than XTTS-v2 in the final Arabic benchmark.
- XTTS-v2: Failed CPU latency/RTF target and speaker cosine target; licensing needs review.

## Manual listening status
Manual MOS/listener fields are pending unless values appear in the summary table. Current `human_listener_scores.csv` does not contain real reviewer scores for these samples.

## Winner
XTTS-v2 is the automatic Arabic WER winner, with clear caveats for CPU latency, RTF, speaker similarity, and licensing.

## Evidence paths
- `results\xtts_arabic_evaluation.csv`
- `outputs\xtts\arabic\ar_clone_01_greeting.json`
- `outputs\xtts\arabic\ar_clone_02_technology.json`
- `outputs\xtts\arabic\ar_clone_03_conversational.json`
- `outputs\xtts\arabic\ar_clone_04_numbers.json`
- `outputs\xtts\arabic\ar_clone_05_expressive.json`
- `results\chatterbox_arabic_evaluation.csv`
- `outputs\chatterbox\arabic\ar_clone_01_greeting.json`
- `outputs\chatterbox\arabic\ar_clone_02_technology.json`
- `outputs\chatterbox\arabic\ar_clone_03_conversational.json`
- `outputs\chatterbox\arabic\ar_clone_04_numbers.json`
- `outputs\chatterbox\arabic\ar_clone_05_expressive.json`
- `results\arabic_speaker_similarity.csv`
- `evidence\result_snapshots\arabic_model_comparison_summary.json`
- `evidence\result_snapshots\arabic_speaker_similarity_summary.json`
- `evidence\result_snapshots\xtts\arabic\xtts_arabic_summary.json`
- `evidence\result_snapshots\chatterbox\arabic\chatterbox_arabic_summary.json`
- `results\human_listener_scores.csv`
- `results\human_evaluation_summary.csv`

## Consolidated rows
See `results\arabic_model_comparison.csv`.
