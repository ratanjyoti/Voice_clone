# English Workspace

This folder is an index for language-specific work. The actual source/evidence files stay in their original locations so existing scripts and reports keep working.

## Start here
- `docs/english_results.md`
- `results/english_model_comparison.csv`
- `results/mms_english_evaluation.csv`
- `results/neutts_english_evaluation.csv`
- `scripts/generate_final_mms.py`
- `scripts/generate_final_neutts_english.py`
- `src/generate_mms.py`
- `src/generate_neutts_english.py`
- `data/test_sentences/english_voice_clone_tests.json`

## Common commands
```powershell
python scripts\build_language_reports.py
```
```powershell
.\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language english
```
```powershell
.\.venv-neutts\Scripts\python.exe -u scripts\generate_final_neutts_english.py
```
```powershell
.\.venv\Scripts\python.exe -u scripts\evaluate_final_samples.py --target english
```

## File groups
- data: 29 files
- doc: 1 files
- evidence: 103 files
- output: 76 files
- result: 5 files
- script: 8 files
- source: 4 files

## Most relevant files
| Kind | Path | Notes |
| --- | --- | --- |
| data | `data/reference_audio/english/ratan_20s.original.m4a` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_20s.pt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_20s.txt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_20s.wav` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_20s_norm.pt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_20s_norm.txt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_20s_norm.wav` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_conversational.original.m4a` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_conversational.txt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_conversational.wav` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_expressive.original.m4a` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_expressive.txt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_expressive.wav` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_neutral.original.m4a` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_neutral.pt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_neutral.txt` | reference voice/audio for this language |
| data | `data/reference_audio/english/ratan_neutral.wav` | reference voice/audio for this language |
| data | `data/test_sentences/english.json` | benchmark text set for this language |
| data | `data/test_sentences/english_voice_clone_tests.json` | benchmark text set for this language |
| data | `data/user_voice_clones/ratan/english/reference.txt` | language-related file |
| data | `data/user_voice_clones/ratan/english/reference.wav` | language-related file |
| data | `data/user_voice_clones/ratan/english/reference_input.m4a` | language-related file |
| data | `data/user_voice_clones/ratan/english/target_text.txt` | language-related file |
| data | `data/user_voice_clones/ratan/english/voice_profile.json` | language-related file |
| data | `data/voice_profiles/ratan_20s_natural.json` | language-related file |
| data | `data/voice_profiles/ratan_20s_natural_normalized.json` | language-related file |
| data | `data/voice_profiles/ratan_conversational.json` | language-related file |
| data | `data/voice_profiles/ratan_expressive.json` | language-related file |
| data | `data/voice_profiles/ratan_neutral.json` | language-related file |
| doc | `docs/english_results.md` | human-readable documentation for this language |
| ... | See `files.csv` | 196 more files |

## Full manifest
See `files.csv`.
