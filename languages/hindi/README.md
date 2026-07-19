# Hindi Workspace

This folder is an index for language-specific work. The actual source/evidence files stay in their original locations so existing scripts and reports keep working.

## Start here
- `docs/hindi_results.md`
- `results/hindi_model_comparison.csv`
- `results/mms_hindi_evaluation.csv`
- `results/chatterbox_hindi_evaluation.csv`
- `results/indicf5_hindi_initial_test.csv`
- `scripts/generate_final_mms.py`
- `scripts/generate_final_chatterbox_hindi.py`
- `scripts/evaluate_indicf5_hindi_initial.py`
- `src/generate_chatterbox_hindi.py`
- `src/generate_indicf5_hindi_test.py`
- `data/test_sentences/hindi_voice_clone_tests.json`

## Common commands
```powershell
python scripts\build_language_reports.py
```
```powershell
.\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language hindi
```
```powershell
.\.venv\Scripts\python.exe -u scripts\generate_final_chatterbox_hindi.py
```
```powershell
.\.venv\Scripts\python.exe -u scripts\evaluate_final_samples.py --target hindi
```
```powershell
.\.venv-indicf5\Scripts\python.exe -u src\generate_indicf5_hindi_test.py
```

## File groups
- data: 13 files
- doc: 2 files
- evidence: 38 files
- output: 63 files
- result: 7 files
- script: 3 files
- source: 2 files

## Most relevant files
| Kind | Path | Notes |
| --- | --- | --- |
| data | `data/reference_audio/hindi/ratan_hindi_conversational.original.m4a` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_conversational.txt` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_conversational.wav` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_expressive.original.m4a` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_expressive.txt` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_expressive.wav` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_neutral.original.m4a` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_neutral.txt` | reference voice/audio for this language |
| data | `data/reference_audio/hindi/ratan_hindi_neutral.wav` | reference voice/audio for this language |
| data | `data/reference_audio/indicf5/ratan_reference_indicf5.wav` | reference voice/audio for this language |
| data | `data/reference_audio/indicf5/ratan_reference_transcript.txt` | reference voice/audio for this language |
| data | `data/test_sentences/hindi.json` | benchmark text set for this language |
| data | `data/test_sentences/hindi_voice_clone_tests.json` | benchmark text set for this language |
| doc | `docs/hindi_results.md` | human-readable documentation for this language |
| doc | `docs/indicf5_hindi_installation_notes.md` | human-readable documentation for this language |
| evidence | `evidence/result_snapshots/chatterbox/hindi/conversational_hindi_cloning_results.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/result_snapshots/chatterbox/hindi/expressive_hindi_cloning_results.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/result_snapshots/chatterbox/hindi/neutral_hindi_cloning_results.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/result_snapshots/hindi_model_summary.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/result_snapshots/hindi_speaker_similarity_summary.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/result_snapshots/indicf5_hindi_initial_generation_failure.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/result_snapshots/indicf5_hindi_initial_generation_success.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/result_snapshots/indicf5_hindi_reference_validation.json` | execution proof/log/snapshot for this language |
| evidence | `evidence/terminal_logs/chatterbox_hindi_conversational.txt` | execution proof/log/snapshot for this language |
| evidence | `evidence/terminal_logs/chatterbox_hindi_expressive.txt` | execution proof/log/snapshot for this language |
| evidence | `evidence/terminal_logs/chatterbox_hindi_final_generation.txt` | execution proof/log/snapshot for this language |
| evidence | `evidence/terminal_logs/chatterbox_hindi_final_generation_resume.txt` | execution proof/log/snapshot for this language |
| evidence | `evidence/terminal_logs/chatterbox_hindi_neutral.txt` | execution proof/log/snapshot for this language |
| evidence | `evidence/terminal_logs/english_hindi_final_wer_evaluation.txt` | execution proof/log/snapshot for this language |
| evidence | `evidence/terminal_logs/english_hindi_final_wer_evaluation_retry_no_vad.txt` | execution proof/log/snapshot for this language |
| ... | See `files.csv` | 98 more files |

## Full manifest
See `files.csv`.
