# Arabic Workspace

This folder is an index for language-specific work. The actual source/evidence files stay in their original locations so existing scripts and reports keep working.

## Start here
- `docs/arabic_results.md`
- `results/arabic_model_comparison.csv`
- `results/xtts_arabic_evaluation.csv`
- `results/chatterbox_arabic_evaluation.csv`
- `scripts/evaluate_xtts_arabic.py`
- `scripts/evaluate_chatterbox_arabic.py`
- `src/generate_xtts_arabic.py`
- `src/generate_chatterbox_arabic.py`
- `data/test_sentences/arabic_voice_clone_tests.json`

## Common commands
```powershell
python scripts\build_language_reports.py
```
```powershell
.\.venv-xtts-arabic\Scripts\python.exe -u src\generate_xtts_arabic.py
```
```powershell
.\.venv-xtts-arabic\Scripts\python.exe -u scripts\evaluate_xtts_arabic.py
```
```powershell
.\.venv\Scripts\python.exe -u src\generate_chatterbox_arabic.py
```
```powershell
.\.venv\Scripts\python.exe -u scripts\evaluate_chatterbox_arabic.py
```

## File groups
- data: 452 files
- doc: 3 files
- evidence: 37 files
- output: 79 files
- result: 17 files
- script: 10 files
- source: 5 files

## Most relevant files
| Kind | Path | Notes |
| --- | --- | --- |
| data | `data/reference_audio/arabic/professional_msa/arabic_reference_long.wav` | reference voice/audio for this language |
| data | `data/reference_audio/arabic/professional_msa/arabic_reference_short.wav` | reference voice/audio for this language |
| data | `data/reference_audio/arabic/professional_msa/arabic_reference_standard.wav` | reference voice/audio for this language |
| data | `data/reference_audio/arabic/professional_msa/arabic_reference_standard_candidate_rejected.wav` | reference voice/audio for this language |
| data | `data/reference_audio/arabic/professional_msa/arabic_reference_standard_rejected.wav` | reference voice/audio for this language |
| data | `data/test_sentences/arabic.json` | benchmark text set for this language |
| data | `data/test_sentences/arabic_voice_clone_tests.json` | benchmark text set for this language |
| data | `data/training/arabic/professional_msa/extraction_summary.json` | language-related file |
| data | `data/training/arabic/professional_msa/metadata_all.csv` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000000.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000001.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000002.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000003.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000004.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000005.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000006.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000007.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000008.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000009.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000010.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000011.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000012.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000013.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000014.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000015.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000016.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000017.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000018.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000019.wav` | language-related file |
| data | `data/training/arabic/professional_msa/processed/clip_000020.wav` | language-related file |
| ... | See `files.csv` | 573 more files |

## Full manifest
See `files.csv`.
