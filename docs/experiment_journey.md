# Chatterbox Arabic CPU Engineering Journey

**Date:** July 17, 2026
**Project:** Infinia Voice Case Study
**Language:** Modern Standard Arabic
**Model:** Chatterbox Multilingual TTS
**Execution environment:** Windows, CPU only
**Project root:** `<project-root>`

---

## 1. Objective

The objective of this experiment was to build and evaluate an Arabic voice-cloning pipeline using Chatterbox Multilingual TTS.

The experiment covered:

* licensed Arabic dataset preparation
* audio extraction and conversion
* audio-quality filtering
* deterministic train, validation, and test splits
* reference-speaker selection
* Arabic reference transcription validation
* Chatterbox multilingual setup
* CPU model-loading fixes
* Arabic voice-clone generation
* clipping detection and normalization
* parameter testing
* automatic Arabic WER evaluation
* reference-length comparison
* final five-sample benchmark
* documentation of limitations and remaining human-review work

XTTS-v2 was intentionally not started until the Chatterbox Arabic pipeline was completed and verified.

---

## 2. Environment Preparation

A separate environment was created for Arabic dataset processing and automatic speech evaluation:

```text
.venv-arabic-data
```

Python version:

```text
Python 3.11.15
```

The Python interpreter came from the local `uv` installation because the Windows `py -3.11` launcher did not recognize the installed Astral Python runtime.

Verified interpreter path:

```text
py -3.11
```

Installed dataset and evaluation packages included:

```text
datasets
soundfile
pandas
numpy
librosa
pyarrow
faster-whisper
jiwer
```

The existing Chatterbox environment was reused:

```text
.venv
```

Verified Chatterbox environment:

```text
torch: 2.6.0+cpu
CUDA available: False
Chatterbox import: PASS
```

Available Chatterbox modules included:

```text
models
mtl_tts
tts
tts_turbo
vc
```

The multilingual class was successfully verified:

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
```

The installed generation interface supported:

```text
text
language_id
audio_prompt_path
exaggeration
cfg_weight
temperature
repetition_penalty
min_p
top_p
```

---

## 3. Arabic Dataset Selection

The selected dataset was:

```text
NightPrince/Arabic-professional-voice
```

The dataset contained:

```text
439 rows
```

Available columns:

```text
audio
transcription
```

The dataset used one Arabic professional male voice and fully written Arabic transcripts.

The dataset was downloaded automatically through Hugging Face and cached inside the project model directory.

---

## 4. Initial Dataset Download Failure

The first extraction attempt failed with:

```text
ConnectionError: Couldn't reach 'NightPrince/Arabic-professional-voice'
on the Hub (OfflineModeIsEnabled)
```

### Cause

The PowerShell session still contained Hugging Face offline-mode environment variables:

```text
HF_HUB_OFFLINE
TRANSFORMERS_OFFLINE
HF_DATASETS_OFFLINE
```

### Fix

The offline-mode variables were removed:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue
```

The Hugging Face cache was then configured inside the project:

```text
models\huggingface
```

After disabling offline mode, the dataset downloaded successfully.

Downloaded dataset information:

```text
Rows: 439
Columns: ['audio', 'transcription']
Transcript column: transcription
```

---

## 5. TorchCodec Audio-Decoding Failure

After downloading the dataset, the extraction process failed while iterating over the audio rows.

Error:

```text
ImportError: To support decoding audio data, please install 'torchcodec'.
```

TorchCodec was installed, but its import failed with:

```text
RuntimeError: Could not load libtorchcodec.
```

The reported causes included:

* missing shared FFmpeg DLLs on Windows
* TorchCodec and PyTorch compatibility
* dependency-loading failures
* the installed FFmpeg build not being the required full-shared Windows version

The environment had installed:

```text
torch 2.13.0+cpu
```

TorchCodec attempted to load several DLLs such as:

```text
libtorchcodec_core8.dll
libtorchcodec_core7.dll
libtorchcodec_core6.dll
libtorchcodec_core5.dll
libtorchcodec_core4.dll
```

All attempts failed.

### Decision

TorchCodec was removed from the critical path instead of changing the working Windows FFmpeg and PyTorch setup.

### Fix

The extraction script was changed from:

```python
Audio(decode=True)
```

to:

```python
Audio(decode=False)
```

This made Hugging Face return the original audio bytes instead of invoking TorchCodec.

The audio bytes were decoded with `soundfile`:

```python
sf.read(
    io.BytesIO(audio_bytes),
    dtype="float32",
    always_2d=True,
)
```

A one-row decoding test passed:

```text
Rows: 439
Audio keys: dict_keys(['bytes', 'path'])
Shape: (128000, 1)
Sample rate: 16000
Raw-byte decoding: PASS
```

This confirmed that TorchCodec was not required for the project.

---

## 6. Audio Extraction and Conversion

The dataset extraction script was:

```text
scripts\extract_arabic_professional_voice.py
```

The original audio was decoded at:

```text
16000 Hz
```

All extracted audio was converted to:

```text
24000 Hz
Mono
PCM16 WAV
```

Processed audio location:

```text
data\training\arabic\professional_msa\processed
```

Metadata output:

```text
data\training\arabic\professional_msa\metadata_all.csv
```

Summary output:

```text
data\training\arabic\professional_msa\extraction_summary.json
```

Final extraction result:

```text
Successfully extracted: 439
Failed extractions: 0
```

A total of 439 WAV files were written successfully.

---

## 7. Audio-Quality Filtering

Each extracted clip was checked for:

* transcript availability
* duration
* RMS level
* peak level
* hard clipping
* silence ratio
* channel count
* sample rate
* invalid or non-finite samples

The filtering rules rejected clips when:

* duration was below 2.5 seconds
* duration was above 15 seconds
* RMS was too low
* hard clipping was detected
* silence ratio was 30% or higher
* transcript was missing

Final quality result:

```text
Total clips: 439
Usable clips: 393
Rejected clips: 46
Usable rate: approximately 89.5%
```

Rejection breakdown:

```text
31  silence ratio too high
8   duration above 15 seconds
4   duration below 2.5 seconds
2   duration below 2.5 seconds and silence ratio too high
1   duration above 15 seconds and silence ratio too high
```

The rejected clips were not necessarily corrupted. Most contained too much silence for clean voice-cloning reference or training use.

The filtering was kept conservative because 393 usable clips were still available.

---

## 8. Deterministic Dataset Splits

A reproducible split script was created:

```text
scripts\create_arabic_dataset_splits.py
```

Random seed:

```text
42
```

Final split:

```text
Train: 314
Validation: 39
Test: 40
Total usable: 393
```

Files created:

```text
data\training\arabic\professional_msa\splits\train.csv
data\training\arabic\professional_msa\splits\validation.csv
data\training\arabic\professional_msa\splits\test.csv
data\training\arabic\professional_msa\splits\split_summary.json
```

Only training-split clips were considered for reference-audio selection. Validation and test clips remained held out.

---

## 9. Arabic Reference Candidate Selection

Reference candidates were selected using:

* duration between approximately 6 and 12 seconds
* silence ratio below 20%
* RMS above 0.10
* training split only
* complete Arabic transcript
* one consistent speaker

The initial selected references were:

```text
Short:
clip_000158.wav
Duration: 6.64 seconds

Standard:
clip_000209.wav
Duration: 9.04 seconds

Long:
clip_000342.wav
Duration: 11.92 seconds
```

They were copied into:

```text
data\reference_audio\arabic\professional_msa
```

Initial filenames:

```text
arabic_reference_short.wav
arabic_reference_standard.wav
arabic_reference_long.wav
```

---

## 10. Technical Reference Validation

The three references were checked for:

* duration
* sample rate
* channel count
* RMS
* peak level
* clipping
* SHA256 hash

Results:

```text
arabic_reference_short.wav
Duration: 6.64 s
Sample rate: 24000 Hz
Channels: 1
RMS: 0.2873
Peak: 0.8980
Clipped samples: 0
SHA256:
8d595f96e496e9f597833ebbd8b1f4853080e292b36b17b7feb489aa05889d8f
```

```text
arabic_reference_standard.wav
Duration: 9.04 s
Sample rate: 24000 Hz
Channels: 1
RMS: 0.3004
Peak: 0.9057
Clipped samples: 0
SHA256:
7de5f32f40052b9f64ca16eaf36a7495213c24a92d127d1f9917dbeb02fe8023
```

```text
arabic_reference_long.wav
Duration: 11.92 s
Sample rate: 24000 Hz
Channels: 1
RMS: 0.2892
Peak: 0.8966
Clipped samples: 0
SHA256:
94854a75878e1fcc454cd433cadf4603ca6a6c2ad187f6d7bfa91c7ede0b17d0
```

All three passed technical audio validation.

---

## 11. Arabic Reference ASR Validation

Because the local evaluator did not speak Arabic, Faster Whisper was used as an automatic intelligibility check.

The first test used Whisper Tiny.

Tiny produced many incorrect Arabic transcriptions, even when the source audio appeared technically strong.

Examples included substitutions such as:

```text
Expected:
أَنَا أُسَاعِدُكَ الْآنَ

Tiny prediction:
انا اصعيدك الان
```

This showed that Whisper Tiny was too weak for reliable Arabic validation.

### Improvement

Whisper Small was used instead:

```python
WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)
```

Whisper Small produced significantly better Arabic transcriptions.

Arabic normalization removed:

* diacritics
* punctuation
* tatweel
* common Alef variants
* common letter-form differences

The first reference WER results were:

```text
Short reference WER: 0.00%
Standard reference WER: 17.65%
Long reference WER: 9.09%
Average WER: 8.91%
```

The standard reference had the weakest automatic score and was therefore replaced.

---

## 12. Standard Reference Replacement Tests

### Candidate 1

Candidate:

```text
clip_000296.wav
```

Automatic WER:

```text
16.67%
```

This was only a small improvement over the original standard reference and was rejected.

### Candidate 2

Candidate:

```text
clip_000216.wav
```

Transcript:

```text
التَّفْسِيرُ الْمَوْضُوعِيُّ لِلْقُرْآنِ يَهْتَمُّ
بِجَمْعِ الْآيَاتِ الَّتِي تَتَحَدَّثُ عَنْ
قَضِيَّةٍ وَاحِدَةٍ وَدِرَاسَتِهَا مَعًا.
```

Whisper Small prediction matched the normalized transcript.

Result:

```text
WER: 0.00%
Arabic probability: 1.0000
```

This became the final standard reference.

The rejected candidates were preserved as evidence:

```text
arabic_reference_standard_candidate_rejected.wav
arabic_reference_standard_rejected.wav
```

Final reference files:

```text
arabic_reference_short.wav
arabic_reference_standard.wav
arabic_reference_long.wav
```

---

## 13. Final Reference Validation

The final three-reference validation result was:

```text
Short reference WER: 0.00%
Standard reference WER: 0.00%
Long reference WER: 9.09%
Average normalized WER: 3.03%
```

All references were identified as Arabic with:

```text
Arabic probability: 1.0000
```

The final reference set was therefore accepted for Chatterbox voice-cloning experiments.

Important limitation:

Whisper-based WER is an intelligibility proxy. It does not prove perfect native pronunciation, prosody, or speaker similarity.

---

## 14. Arabic Test Sentences

The Arabic test file was created at:

```text
data\test_sentences\arabic_voice_clone_tests.json
```

It contained five Modern Standard Arabic tests:

```text
ar_clone_01_greeting
ar_clone_02_technology
ar_clone_03_conversational
ar_clone_04_numbers
ar_clone_05_expressive
```

Categories:

```text
Neutral greeting
Technology
Conversational
Numbers
Expressive
```

The JSON structure was validated successfully using Python’s JSON parser.

---

## 15. Chatterbox CPU Loading Failure

The first Chatterbox Arabic generation script was:

```text
src\generate_chatterbox_arabic_test.py
```

The initial device configuration used:

```python
device = torch.device("cpu")
```

The model download completed, but checkpoint loading failed with:

```text
RuntimeError:
Attempting to deserialize object on a CUDA device but
torch.cuda.is_available() is False.
```

### Cause

The installed Chatterbox multilingual code did not correctly apply CPU checkpoint remapping when it received a `torch.device("cpu")` object.

### Fix

The device was changed to a literal string:

```python
device = "cpu"
```

An intermediate editing mistake changed the line to:

```python
device("cpu")
```

This caused:

```text
NameError: name 'device' is not defined
```

The mistake was corrected to:

```python
device = "cpu"
```

After this correction, the model loaded successfully on CPU.

Successful model loading output included:

```text
loaded PerthNet (Implicit) at step 250,000
Model load time: 23.32 seconds
```

The `pkg_resources`, LoRA, CUDA context, and generation-flag messages were warnings and did not prevent generation.

---

## 16. First Successful Arabic Voice Clone

The first generated sample was:

```text
outputs\chatterbox\arabic\ar_clone_01_greeting.wav
```

Generation details:

```text
Model load time: 23.32 seconds
Generation time: 62.14 seconds
Audio duration: 6.52 seconds
RTF: 9.530
Sample rate: 24000 Hz
```

This confirmed that Chatterbox Multilingual Arabic voice cloning worked on the CPU-only Windows machine.

However, the runtime was much slower than real time.

---

## 17. Clipping Failure

The first generated WAV passed basic generation but failed the peak and clipping check.

Initial result:

```text
Duration: 6.52 seconds
Sample rate: 24000 Hz
Channels: 1
RMS: 0.3314
Peak: 1.0000
Clipped samples: 601
Silence ratio: 0.1377
```

The raw generated waveform peak was later found to be:

```text
1.1494
```

Saving this waveform directly as PCM16 caused hard clipping.

### Fix

Safe peak normalization was added before saving:

```python
raw_peak = float(abs(waveform).max())

if raw_peak > 0:
    target_peak = 0.95
    waveform = waveform * (target_peak / raw_peak)
```

The normalized waveform result was:

```text
Raw waveform peak: 1.1494
Normalized peak: 0.9500
```

After regeneration:

```text
Duration: 6.52 seconds
Sample rate: 24000 Hz
Channels: 1
RMS: 0.2203
Peak: 0.9500
Clipped samples: 0
Silence ratio: 0.2061
```

Peak normalization successfully removed all clipping.

All final generated samples used this normalization rule.

---

## 18. First Generated-Sample Arabic Validation

Whisper Small detected the first Chatterbox output as Arabic:

```text
Arabic probability: 1.0000
```

Expected sentence:

```text
مَرْحَبًا بِكَ. أَنَا مُسَاعِدُكَ الصَّوْتِيُّ،
وَيَسْعَدُنِي أَنْ أُسَاعِدَكَ الْيَوْمَ.
```

Initial predicted sentence:

```text
مرحبا بك أنا مساعدك الصوتي ويساعدني أن أساعدك اليومة
```

The prediction suggested substitutions around:

```text
وَيَسْعَدُنِي
```

and:

```text
الْيَوْمَ
```

This showed that the output was intelligible Arabic but could contain pronunciation or lexical substitutions.

A dedicated evaluation script was created:

```text
scripts\validate_chatterbox_arabic_sample.py
```

An initial attempt to run it failed because the file had not actually been created in the `scripts` directory.

Error:

```text
can't open file:
scripts\validate_chatterbox_arabic_sample.py
[Errno 2] No such file or directory
```

The file was then created correctly and verified with:

```text
Test-Path: True
```

A separate Windows console issue also occurred when printing Arabic text:

```text
UnicodeEncodeError:
'charmap' codec can't encode characters
```

### Fix

UTF-8 console variables were set:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

After this, Arabic transcripts and WER results printed correctly.

---

## 19. Parameter Testing

A reusable Arabic generator was created:

```text
src\generate_chatterbox_arabic.py
```

Parameter tests used:

```text
Seed: 42
Reference: standard
Exaggeration: 0.5
Repetition penalty: 2.0
Minimum probability: 0.05
Top-p: 1.0
```

The experiment varied:

```text
temperature
cfg_weight
```

Five meaningful greeting configurations were generated because exhaustive CPU testing would have required excessive time.

Parameter outputs were saved under:

```text
outputs\chatterbox\arabic
```

Parameter-generation metrics were written to CSV files under:

```text
results
```

Each run recorded:

* seed
* reference file
* temperature
* cfg weight
* exaggeration
* repetition penalty
* minimum probability
* top-p
* generation time
* duration
* RTF
* sample rate
* RMS
* peak
* clipping count
* silence ratio
* output path

Each candidate was evaluated using Faster Whisper Small and normalized Arabic WER.

---

## 20. Selected Chatterbox Parameters

The automatic parameter winner used:

```text
temperature: 0.6
cfg_weight: 0.5
exaggeration: 0.5
repetition_penalty: 2.0
min_p: 0.05
top_p: 1.0
seed: 42
reference: standard
```

The selected candidate was:

```text
outputs\chatterbox\arabic\ar_clone_01_temp06_cfg05.wav
```

Automatic WER:

```text
0.00%
```

Several candidates tied at 0.00% WER.

Therefore, this was only the automatic winner. It was not considered proof of the best:

* speaker similarity
* naturalness
* emotional quality
* native Arabic pronunciation
* prosody

---

## 21. Final Five-Sample Generation

The final Arabic generator loaded the model once and generated all five tests.

Generated outputs:

```text
outputs\chatterbox\arabic\ar_clone_01_greeting.wav
outputs\chatterbox\arabic\ar_clone_02_technology.wav
outputs\chatterbox\arabic\ar_clone_03_conversational.wav
outputs\chatterbox\arabic\ar_clone_04_numbers.wav
outputs\chatterbox\arabic\ar_clone_05_expressive.wav
```

Each output had a JSON sidecar containing:

* model name
* language
* language code
* reference path
* seed
* generation parameters
* generation time
* audio duration
* RTF
* sample rate
* RMS
* peak
* clipping count
* silence ratio
* output path

All WAV files were saved as:

```text
PCM16
Mono
24000 Hz
Normalized to peak 0.95
```

---

## 22. Final Automatic Evaluation

Evaluation script:

```text
scripts\evaluate_chatterbox_arabic.py
```

Final evaluation CSV:

```text
results\chatterbox_arabic_evaluation.csv
```

Summary JSON:

```text
evidence\result_snapshots\chatterbox\arabic\
chatterbox_arabic_summary.json
```

The evaluation measured:

* Arabic language probability
* ASR transcript
* normalized reference text
* normalized hypothesis
* WER
* duration
* generation time
* RTF
* RMS
* peak
* clipping samples
* silence ratio

Human-review fields were added but intentionally left empty:

```text
similarity_score_1_to_5
naturalness_score_1_to_5
pronunciation_score_1_to_5
metallic
missing_words
repeated_words
accepted
reviewer_notes
```

No human scores were invented.

---

## 23. Final Five-Sample Results

### Greeting

```text
Sample: ar_clone_01_greeting
WER: 0.00%
RTF: 14.29
Clipping samples: 0
```

### Technology

```text
Sample: ar_clone_02_technology
WER: 30.00%
RTF: 8.99
Clipping samples: 0
```

### Conversational

```text
Sample: ar_clone_03_conversational
WER: 9.09%
RTF: 15.92
Clipping samples: 0
```

### Numbers

```text
Sample: ar_clone_04_numbers
WER: 38.46%
RTF: 10.84
Clipping samples: 0
```

### Expressive

```text
Sample: ar_clone_05_expressive
WER: 23.08%
RTF: 10.48
Clipping samples: 0
```

Overall result:

```text
Average WER: 20.13%
Samples passing Arabic WER <= 15%: 2 of 5
Samples with zero clipping: 5 of 5
Average RTF: approximately 12.10
RTF range: approximately 8.99 to 15.92
```

---

## 24. Reference-Length Comparison

The greeting sentence was generated using:

```text
Short reference
Standard reference
Long reference
```

All three tests used:

```text
Same text
Same seed
Same selected parameters
```

Automatic result:

```text
Short reference: 0.00% WER
Standard reference: 0.00% WER
Long reference: higher WER
```

The short and standard references tied on automatic WER.

No final voice-reference winner should be selected using WER alone.

Native Arabic listening review must compare:

* speaker similarity
* vocal identity stability
* naturalness
* pronunciation
* pacing
* prosody
* artifacts
* consistency across sentence types

---

## 25. Successful Outcomes

The following parts were completed successfully:

```text
Arabic dataset downloaded
439 clips extracted
393 clips passed quality filtering
Train, validation, and test splits created
Reference candidates selected
Reference WER reduced to 3.03% average
Chatterbox multilingual loaded on CPU
CUDA checkpoint-loading issue fixed
Arabic voice cloning generated successfully
Raw clipping detected
Peak normalization implemented
Final clipping reduced to zero
Parameter configurations tested
Automatic WER evaluation completed
Reference comparison completed
Five final Arabic samples generated
CSV and JSON evidence files created
Human-review fields added
XTTS work correctly deferred
```

---

## 26. Main Failures and Their Fixes

| Failure                                      | Cause                                              | Fix                                     |
| -------------------------------------------- | -------------------------------------------------- | --------------------------------------- |
| Hugging Face dataset unreachable             | Offline-mode variables enabled                     | Removed offline environment variables   |
| Dataset audio iteration failed               | Hugging Face required TorchCodec                   | Used `Audio(decode=False)`              |
| TorchCodec import failed                     | Windows FFmpeg shared DLL and compatibility issues | Decoded raw bytes with `soundfile`      |
| Whisper Tiny gave poor Arabic transcripts    | Model too small for reliable MSA validation        | Switched to Whisper Small               |
| Initial standard reference had 17.65% WER    | Less reliable automatic transcription              | Replaced with `clip_000216.wav`         |
| Chatterbox attempted CUDA checkpoint loading | `torch.device("cpu")` was not handled correctly    | Used `device="cpu"` string              |
| Device line produced `NameError`             | Accidental `device("cpu")` edit                    | Corrected to `device = "cpu"`           |
| First output clipped                         | Raw waveform peak was 1.1494                       | Normalized peak to 0.95                 |
| Arabic text failed to print                  | Windows CP1252 console encoding                    | Enabled UTF-8 Python and console output |
| WER script not found                         | File had not been created in `scripts`             | Created and verified file path          |
| CPU inference was slow                       | Large multilingual model on CPU                    | Documented limitation; GPU recommended  |

---

## 27. Chatterbox Arabic Strengths

The experiment showed that Chatterbox Multilingual:

* can run Arabic voice cloning on CPU
* can condition on a reference speaker
* can generate intelligible Arabic
* can produce some samples with 0% automatic WER
* performs relatively well on greeting and conversational content
* produces technically valid 24 kHz mono WAV files
* works without clipping after normalization
* supports reproducible generation with fixed seeds
* allows several generation controls
* can reuse the same professional Arabic reference speaker

---

## 28. Chatterbox Arabic Weaknesses

The experiment also showed important weaknesses:

* average final WER was 20.13%
* only two of five samples passed the 15% Arabic WER target
* numeric content produced the highest WER
* technology and expressive sentences also contained substitutions
* longer or more complex sentences were less reliable
* CPU RTF ranged from approximately 9 to 16
* CPU generation was much slower than real time
* automatic WER does not prove native pronunciation
* speaker similarity remains unscored
* naturalness remains unscored
* expressive quality remains unscored
* native Arabic review is still required
* final production suitability cannot yet be declared

---

## 29. Engineering Conclusion

Chatterbox Multilingual successfully demonstrated Arabic zero-shot voice cloning on the tested CPU-only Windows system.

The source Arabic references were strong:

```text
Average reference WER: 3.03%
```

The final generated WAV files were technically clean:

```text
5 of 5 samples with zero clipping
```

However, pronunciation consistency was not sufficient across all test categories:

```text
Average generated-sample WER: 20.13%
Only 2 of 5 samples passed WER <= 15%
```

CPU inference was also unsuitable for real-time deployment:

```text
Average RTF: approximately 12.10
```

Therefore, Chatterbox should be treated as a working Arabic voice-cloning candidate, but not yet as the final Arabic production winner.

---
