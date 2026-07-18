param(
  [switch]$SkipGeneration,
  [switch]$SkipNeuTTS
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "This benchmark may load large local/Hugging Face models and can take a long time on CPU."
Write-Host "It does not silently install dependencies or download virtual environments."

if (-not $SkipGeneration) {
  & .\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language english
  & .\.venv\Scripts\python.exe -u scripts\generate_final_mms.py --language hindi
  & .\.venv\Scripts\python.exe -u scripts\generate_final_chatterbox_hindi.py

  if (-not $SkipNeuTTS) {
    $env:PATH = "C:\Program Files\eSpeak NG;" + $env:PATH
    $env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
    $env:ESPEAK_DATA_PATH = "C:\Program Files\eSpeak NG\espeak-ng-data"
    & .\.venv-neutts\Scripts\python.exe -u scripts\generate_final_neutts_english.py
  }
}

& .\.venv-speaker-similarity\Scripts\python.exe -u scripts\verify_speaker_embedding_model.py
& .\.venv-speaker-similarity\Scripts\python.exe -u scripts\evaluate_speaker_similarity.py
& .\.venv\Scripts\python.exe -u scripts\evaluate_final_samples.py --target all
& .\.venv\Scripts\python.exe scripts\summarize_human_evaluation.py

Write-Host "Final benchmark outputs are in results/ and evidence/result_snapshots/."