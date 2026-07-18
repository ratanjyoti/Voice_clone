# XTTS-v2 Arabic Experiment Notes

Date: 2026-07-17

## Environment

- Generation environment: `.venv-xtts-arabic`
- Python: 3.11.15
- Coqui TTS package: 0.27.5
- XTTS torch: 2.8.0+cpu
- CUDA available: False
- Evaluation environment: `.venv-arabic-data`
- ASR: Faster Whisper Small, `language="ar"`, `beam_size=5`, `vad_filter=True`

## Sample Sets

- Chatterbox: `outputs/chatterbox/arabic`
- XTTS-v2: `outputs/xtts/arabic`
- Test sentences: `data/test_sentences/arabic_voice_clone_tests.json`

Review matching test IDs across both models:

- `ar_clone_01_greeting`
- `ar_clone_02_technology`
- `ar_clone_03_conversational`
- `ar_clone_04_numbers`
- `ar_clone_05_expressive`

## Scores To Record

For every model/sample pair, record:

- similarity score: 1-5
- naturalness score: 1-5
- pronunciation score: 1-5
- metallic: None, Mild, or Strong
- missing words: Yes or No
- repeated words: Yes or No
- accepted: Yes or No
- reviewer notes

The project-local cache is:

`models/coqui/tts/tts_models--multilingual--multi-dataset--xtts_v2`

A first download attempt failed because proxy variables pointed to `127.0.0.1:9`; the partial 8 KB checkpoint was removed after stopping the stuck downloader. The successful download used cleared proxy variables and `COQUI_TOS_AGREED=1`.

## Reference Strategy

The greeting was tested with short, standard, long, and multi-reference strategies. Automatic reference winner: `multi` using short + standard + long references.

Reference comparison WER:

- short: 11.11%
- standard: 33.33%
- long: 11.11%
- multi: 0.00%

## Parameters

Automatic parameter winner: Config C.

- temperature: 0.65
- top_k: 40
- top_p: 0.80
- repetition_penalty: 5.0
- length_penalty: 1.0
- speed: 1.0
- seed: 42

## Final Generation Metrics

Final XTTS-v2 five-sample evaluation:

- average WER: 5.36%
- median WER: 7.69%
- maximum WER: 10.00%
- WER pass count <= 15%: 5/5
- failed generations: 0
- clipping pass count: 5/5
- average generation time: 22.64 seconds
- average duration: 7.02 seconds
- average RTF: 3.22

XTTS-v2 is faster than Chatterbox on this CPU test but remains slower than real-time (`RTF > 1`). GPU benchmarking is recommended before deployment.

## Comparison With Chatterbox

Chatterbox final Arabic metrics:

- average WER: 20.13%
- WER pass count <= 15%: 2/5
- clipping pass count: 5/5
- average RTF: 12.10

XTTS-v2 automatic metrics are stronger for this dataset: average WER 5.36%, pass count 5/5, clipping pass 5/5, average RTF 3.22. The automatic comparison selects XTTS-v2, but this does not replace native review for speaker similarity, naturalness, and pronunciation.

## Failures And Fixes

- Coqui install was already satisfied; PowerShell marked the pip run nonzero due temporary cleanup warnings on stderr.
- Initial model download failed under blocked proxy variables.
- Partial checkpoint cache was removed and download retried with proxy variables cleared.
- Coqui emitted torchaudio deprecation warnings; generated WAVs were validated with `soundfile`.
- Some terminal logs render Arabic as mojibake under Windows/cmd, but CSV and JSON files are UTF-8.

## Production Recommendation

Do not deploy solely on automatic WER. Run native Arabic listening review, license review, consent review, and GPU benchmarks. Website integration and fine-tuning were intentionally not started.
