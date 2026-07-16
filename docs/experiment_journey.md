### SILMA experiment deferred after installation profiling

SILMA TTS was selected as a promising Arabic-specific cloning model,
but local installation proved disproportionately expensive for the
available CPU-focused Windows laptop and network connection.

Observed issues:

- The official dependency path pulled `onnxruntime-gpu`.
- The GPU wheel was approximately 213.6 MB.
- Several downloads failed with `IncompleteRead`.
- The GPU package installed but its native DLL did not load.
- Installing `onnxruntime==1.20.1` over the shared namespace restored
  ONNX Runtime with CPU providers.
- The remaining SILMA dependency installation continued for several
  minutes and required additional large ML packages.

Decision:

SILMA was deferred rather than falsely reported as benchmarked. Its
environment and installation logs were retained as evidence. The next
voice-cloning experiment will use the already working MMS outputs as
source speech and a lighter tone-conversion model.

Status: researched and attempted, not successfully benchmarked.

## Reference voice preparation

A personally recorded and consented voice sample was converted into a
controlled format for speaker-cloning evaluation.

Reference file:

`data/reference_audio/ratan_reference_22050_mono.wav`

Properties:

- Duration: 20.202676 seconds
- Codec: PCM signed 16-bit little-endian
- Sample rate: 22,050 Hz
- Channels: mono
- SHA256:
  `58FCC93C94D09471712F5D34A7A8E596C2F6B653A68A72283B95A75F81DE8C4C`

The file was then checked for RMS level, clipping, silence ratio,
duration and format before being supplied to a cloning model.

## OpenVoice environment setup

The validated reference recording passed all signal-quality checks.

A separate environment named `.venv-openvoice` was created so that
OpenVoice dependencies could not damage the working MMS and
faster-whisper environments.

The official OpenVoice repository was cloned and its exact Git commit
was recorded.

The first OpenVoice checkpoint verifies only:

1. Compatible Python environment
2. Repository installation
3. Core `ToneColorConverter` import

No model checkpoint or cloned output is reported until the core import
works successfully.


### OpenVoice installation attempt 1: outdated AV dependency

The official editable installation failed on Windows while resolving:

- `faster-whisper==0.9.0`
- `av==10.*`

PyPI did not provide a compatible prebuilt AV wheel for the active
environment, so pip attempted to compile AV from source. Compilation
failed in `av/logging.pyx` with a Cython `noexcept` type error.

A second environment issue was also identified: the shell's `python`
command resolved to Anaconda Python 3.11.7 instead of the intended
OpenVoice virtual environment.

Recovery plan:

1. Recreate `.venv-openvoice` explicitly with CPython 3.9.
2. Use the environment interpreter by full path.
3. Install binary `av==15.1.0`.
4. Install `faster-whisper==1.2.1`.
5. Install the OpenVoice source package without allowing its outdated
   dependency pins to replace these working packages.

This is a local compatibility patch. The exact upstream OpenVoice Git
commit remains:

`74a1d147b17a8c3092dd5430504bd83ef6c7eb23`

### OpenVoice setup continuation: CPU PyTorch installed

The existing `.venv-openvoice` environment was preserved and all commands
used the explicit interpreter path:

`.\.venv-openvoice\Scripts\python.exe`

CPU-only PyTorch dependencies were installed with pinned versions:

- `torch==2.2.2+cpu`
- `torchaudio==2.2.2+cpu`

The first torch import check exposed a NumPy ABI warning because the
environment had `numpy==2.0.2`. This was resolved by installing
`numpy==1.26.4`, which is compatible with the selected PyTorch wheel.

Verification passed:

```text
torch: 2.2.2+cpu
torchaudio: 2.2.2+cpu
CUDA available: False
```

Retrying the OpenVoice core import then advanced past `torch` and stopped
at the next missing module:

```text
ModuleNotFoundError: No module named 'soundfile'
```

No OpenVoice checkpoints were downloaded. No full dependency install was
performed.

### OpenVoice core import resolved incrementally

OpenVoice setup continued in the existing `.venv-openvoice` environment
using only the explicit interpreter path:

`.\.venv-openvoice\Scripts\python.exe`

PyTorch verification passed with CPU-only builds:

```text
Torch: 2.5.1+cpu
TorchAudio: 2.5.1+cpu
CUDA available: False
```

The OpenVoice `ToneColorConverter` import was then resolved
incrementally, without running `pip install -r external\OpenVoice\requirements.txt`
and without downloading model checkpoints.

Missing/import issues encountered and fixed:

- `soundfile` missing: installed `soundfile==0.13.1`
- `librosa` missing: installed `librosa==0.9.1`
- `pkg_resources` missing for `librosa`: pinned `setuptools==80.9.0`
- `unidecode` missing: installed `unidecode==1.3.7`
- `eng_to_ipa` missing: installed `eng_to_ipa==0.0.2`
- `pypinyin` missing: installed `pypinyin==0.50.0`
- `jieba` missing: installed `jieba==0.42.1`
- `cn2an` missing: installed `cn2an==0.5.22`

Final core import result:

```text
Before OpenVoice import
OpenVoice core import successful
```

No OpenVoice checkpoints were downloaded in this step.

### OpenVoice V2 converter checkpoint loaded on CPU

The official OpenVoice docs at commit `74a1d147b17a8c3092dd5430504bd83ef6c7eb23`
point to the V2 checkpoint archive:

`https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip`

That URL was tested and returned HTTP 404 with `NoSuchBucket`, so the
archive was not available from the documented S3 location.

As a repair path, the official Hugging Face repository owned by `myshell-ai`
was used instead:

`https://huggingface.co/myshell-ai/OpenVoiceV2`

Only the converter checkpoint files were downloaded into project-level
`checkpoints_v2`:

- `checkpoints_v2/converter/config.json`
- `checkpoints_v2/converter/checkpoint.pth`

Recorded hashes:

```text
config.json
SHA256: 9DFFF60350B8C63F2C664EFD92A61B2516EFB22671466960F0E5DFEBD881FA47

checkpoint.pth
SHA256: 9652C27E92B6B2A91632590AC9962EF7AE2B712E5C5B7F4C34EC55EE2B37AB9E
```

A CPU-only loader script was created at:

`scripts/load_openvoice_converter_cpu.py`

Checkpoint load result:

```text
Before ToneColorConverter init
ToneColorConverter initialized on CPU
Loaded checkpoint '...checkpoints_v2\converter\checkpoint.pth'
missing/unexpected keys: [] []
OpenVoice V2 converter checkpoint loaded on CPU
Exit code: 0
```

No reference speaker embedding was extracted and no cloned audio was generated.

### OpenVoice converter load test via `src/load_openvoice_converter.py`

The requested official archive URL was tested:

`https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip`

`Invoke-WebRequest` returned `403 Forbidden`; the `curl.exe` fallback downloaded
only a 243-byte S3 XML error response with `AccessDenied`. ZIP inspection
therefore reported:

```text
BadZipFile: File is not a zip file
```

Because the archive was invalid, it was not extracted. The already verified
official `myshell-ai/OpenVoiceV2` converter files remained in place:

```text
checkpoints_v2/converter/config.json      838 bytes
checkpoints_v2/converter/checkpoint.pth   131320490 bytes
```

A checkpoint manifest was saved to:

`evidence/result_snapshots/openvoice_checkpoint_manifest.csv`

`src/load_openvoice_converter.py` was run with `.venv-openvoice` on CPU and
completed successfully:

```text
OpenVoice converter loaded successfully.
missing/unexpected keys: [] []
```

The load report was saved to:

`evidence/result_snapshots/openvoice_converter_load.json`

No speaker embedding extraction or audio conversion was performed.

## OpenVoice target-speaker embedding

The validated personal reference recording was passed directly to
`ToneColorConverter.extract_se()`.

The higher-level `se_extractor.get_se()` path was intentionally avoided
because its audio segmentation implementation has CPU/Windows
compatibility issues and may attempt to initialize Whisper using CUDA.

Input:

- `data/reference_audio/ratan_reference_22050_mono.wav`
- Duration: 20.202676 seconds
- Sample rate: 22,050 Hz
- Channels: mono
- Reference SHA256:
  `58FCC93C94D09471712F5D34A7A8E596C2F6B653A68A72283B95A75F81DE8C4C`

Output:

- `outputs/openvoice/embeddings/ratan_target_se.pth`
- Extraction device: CPU
- Converter load time: [ACTUAL VALUE]
- Embedding extraction time: [ACTUAL VALUE]
- Embedding shape: [ACTUAL VALUE]
- Embedding L2 norm: [ACTUAL VALUE]
- Embedding SHA256: [ACTUAL VALUE]

The embedding contained finite, non-zero values, and the tensor returned
by OpenVoice matched the tensor saved to disk.

No cloned speech was produced during this checkpoint.
### OpenVoice target speaker embedding extracted

The validated reference recording was used directly with
`ToneColorConverter.extract_se()`; `se_extractor.get_se` was not used.

Reference file:

`data/reference_audio/ratan_reference_22050_mono.wav`

Reference SHA256:

`58FCC93C94D09471712F5D34A7A8E596C2F6B653A68A72283B95A75F81DE8C4C`

Output embedding:

`outputs/openvoice/embeddings/ratan_target_se.pth`

Extraction result:

```text
Converter loaded in 3.1812 seconds.
Target embedding extracted successfully.
Extraction time: 0.0588 seconds
Embedding shape: [1, 256, 1]
Embedding dtype: torch.float32
Embedding norm: 10.67109108
Returned/saved match: True
SHA256: 13489076B73E4DD699FEB362DF8DFDD68719F0707FE9B699725198463E4B643B
```

Verification passed:

- tensor loads successfully
- all values are finite
- tensor is not all zero
- L2 norm is greater than zero
- saved tensor matches the returned tensor
- embedding file size is greater than zero

Report saved to:

`evidence/result_snapshots/openvoice_target_embedding.json`

Separate hash proof saved to:

`evidence/terminal_logs/openvoice_target_embedding_hash.txt`

No source speaker embedding was extracted and no cloned audio was generated.

### OpenVoice single English clone generated

Exactly one English MMS source WAV was converted with OpenVoice V2 on CPU.
No Arabic or Hindi files were processed and WER was not run in this checkpoint.

Source WAV:

`outputs/mms/english/en_01_run_01.wav`

Source text:

`Hello, welcome to Infinia. How can I help you today?`

Source speaker embedding:

`outputs/openvoice/embeddings/mms_english_en_01_source_se.pth`

Target speaker embedding:

`outputs/openvoice/embeddings/ratan_target_se.pth`

Cloned output:

`outputs/openvoice/english/en_01_openvoice_cloned.wav`

Timing:

```text
Converter load time: 7.2419 seconds
Source embedding extraction time: 2.1147 seconds
Conversion time: 4.1774 seconds
Total processing time: 13.6058 seconds
```

Audio and RTF:

```text
Source duration: 3.984000 seconds
Output duration: 3.982222 seconds
Conversion RTF: 1.049023
Total pipeline RTF excluding MMS generation: 3.416644
```

Signal validation passed:

```text
Output RMS: 0.11094506
Output peak: 0.56610107
Clipping ratio: 0.00000000
All samples finite: True
```

SHA256 hashes:

```text
Source WAV: B7CC5A76A2EDB933ED3A9B8901ED90493C94DAA1A2297008344E6D639193DCA0
Source embedding: 63043A183C9BEF5D1A6098F0FA53D06FB8522463C586DC52FF2C187A0A4BE7EE
Target embedding: 13489076B73E4DD699FEB362DF8DFDD68719F0707FE9B699725198463E4B643B
Cloned output: 66F3CE6D20EF0EE16F761EB76BF22360938E21387494A00F675E56DC8455ED59
```

Report saved to:

`evidence/result_snapshots/openvoice_english_single_conversion.json`

Terminal log saved to:

`evidence/terminal_logs/openvoice_english_single_conversion.txt`

### First OpenVoice English conversion result

The first MMS English → OpenVoice conversion completed successfully and
produced a valid, complete WAV file.

Manual listening result:

- Complete sentence: yes
- Intelligibility: understandable
- Speaker similarity to reference: approximately 3%
- Naturalness: poor
- Main artifacts: metallic and synthetic voice quality
- Human-like result: no

Conclusion:

The pipeline is operational, but the first conversion does not meet the
required naturalness or speaker-similarity targets. It will remain as a
measured experimental baseline rather than being presented as the final
voice-cloning solution.
### Speaker similarity evaluation environment prepared

A separate speaker-evaluation environment was created without modifying
`.venv`, `.venv-openvoice`, or `.venv-silma`.

Environment:

```text
.venv-speaker-eval
Python: 3.11.15
```

`uv venv --python 3.11` was attempted first, but uv failed with a Windows
cache permission error. The environment was then created with the exact
Astral CPython 3.11.15 interpreter and pip was bootstrapped with `ensurepip`.

CPU PyTorch installation initially failed through pip due to an interrupted
large wheel download. The CPU wheels were then downloaded with resume-capable
`curl.exe` and installed locally.

Import checkpoint passed:

```text
Torch: 2.5.1+cpu
TorchAudio: 2.5.1+cpu
SpeechBrain: 1.1.0
CUDA: False
```

Logs saved:

```text
evidence/terminal_logs/speaker_eval_environment.txt
evidence/terminal_logs/speaker_eval_pytorch_install.txt
evidence/terminal_logs/speaker_eval_speechbrain_install.txt
evidence/terminal_logs/speaker_eval_import_test.txt
```

No ECAPA-TDNN checkpoint was downloaded in this step.

## OpenVoice English experiment conclusion

The MMS English and OpenVoice V2 pipeline completed successfully and
generated a valid, intelligible cloned WAV.

Measured results:

- Source sentence: `en_01`
- OpenVoice conversion time: 3.3633 seconds
- Output duration: 3.9822 seconds
- Conversion RTF: 0.8446
- Total OpenVoice processing time: 7.7583 seconds
- Total processing RTF excluding MMS generation: 1.9482
- Round-trip WER: 0.10
- Round-trip CER: 0.0408

OpenVoice internal cosine:

- Source to target: 0.089402
- Cloned to target: 0.812604

Independent SpeechBrain ECAPA cosine:

- Reference to original MMS: 0.105977
- Reference to OpenVoice output: 0.264830
- Improvement: 0.158853
- Required 0.75 threshold reached: no

Human listening:

- Complete sentence: yes
- Intelligibility: acceptable
- Speaker similarity: approximately 3%
- Naturalness: poor
- Main artifact: metallic and synthetic output

Decision:

The OpenVoice pipeline is retained as a completed experimental baseline
but rejected as the final English pipeline. Its internal speaker score
was not accepted as conclusive because an independent ECAPA model and
human listening did not confirm strong similarity.
## MeloTTS local package repair checkpoint

MeloTTS was repaired only inside `.venv-melotts`; no other virtual
environments were modified and no model checkpoints were downloaded.

Source verification passed:

```text
external\MeloTTS\melo\__init__.py exists: True
external\MeloTTS\melo\api.py exists: True
```

Initial failure being addressed:

```text
ModuleNotFoundError: No module named 'melo'
```

The normal editable install of `external\MeloTTS` could not be completed
cleanly. First, pip hit a Windows temporary build-tracker permission error;
a project-local temp directory produced the same permission error. The same
install was then run with elevated permissions and reached dependency
collection, but it did not complete within the run window while downloading
and resolving MeloTTS dependencies. This output was preserved in:

```text
evidence/terminal_logs/melotts_package_install.txt
evidence/terminal_logs/melotts_package_install_retry_workspace_temp.txt
evidence/terminal_logs/melotts_package_install_retry_escalated.txt
```

Following the checkpoint rule, the local repository was installed with:

```text
.\.venv-melotts\Scripts\python.exe -m pip install --no-deps -e external\MeloTTS
```

Result:

```text
Successfully installed melotts-0.1.2
```

This fixed the original `melo` package import path. The first import retry
then failed at:

```text
ModuleNotFoundError: No module named 'librosa'
```

Only the pinned MeloTTS import dependency `librosa==0.9.1` was installed.
That exposed the next missing legacy module:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

`setuptools` was pinned to `80.9.0` inside `.venv-melotts`, restoring
`pkg_resources`. The next missing module was:

```text
ModuleNotFoundError: No module named 'tqdm'
```

Only `tqdm` was installed. The final import retry advanced further into
MeloTTS text normalization and stopped at the next real traceback:

```text
ModuleNotFoundError: No module named 'cn2an'
```

Current status:

```text
melotts: 0.1.2 installed editable from external\MeloTTS
melo package import path: fixed
from melo.api import TTS: not yet successful
next missing dependency: cn2an
```

Evidence logs:

```text
evidence/terminal_logs/melotts_source_file_check.txt
evidence/terminal_logs/melotts_package_install_nodeps.txt
evidence/terminal_logs/melotts_librosa_install.txt
evidence/terminal_logs/melotts_setuptools_install.txt
evidence/terminal_logs/melotts_tqdm_install.txt
evidence/terminal_logs/melotts_import_test_after_tqdm.txt
evidence/terminal_logs/melotts_pip_list_after_checkpoint.txt
```

## MeloTTS English CPU smoke checkpoint

A fully working MeloTTS English pipeline was built inside `.venv-melotts` only.
No packages were installed globally and the existing CPU PyTorch installation
was preserved:

```text
torch==2.5.1+cpu
torchaudio==2.5.1+cpu
CUDA available: False
```

The official local repository at `external\MeloTTS` remains installed editable:

```text
melotts==0.1.2
Editable project location: external\MeloTTS
```

Source inspection showed that `melo.text.cleaner` imported all language
modules unconditionally, and `melo.text.english` imported Japanese only for the
small `distribute_phone` helper. To avoid installing the full multilingual,
UI, and training stack for an English-only smoke test, a minimal local patch
was made:

- `external/MeloTTS/melo/text/cleaner.py`: lazy-load only the requested language module.
- `external/MeloTTS/melo/text/english.py`: define `distribute_phone` locally instead of importing Japanese.
- `external/MeloTTS/melo/text/__init__.py`: lazy-load only the requested BERT feature module.

The English dependency set installed into `.venv-melotts` included:

```text
cached_path==1.8.10
transformers==4.27.4
g2p-en==2.1.0
inflect==7.0.0
txtsplit==1.0.0
pydub==0.25.1
Unidecode==1.3.7
eng_to_ipa==0.0.2
librosa==0.9.1
setuptools==80.9.0
tqdm==4.68.4
cn2an==0.5.22
```

Transitive English/runtime dependencies included `nltk==3.9.2`,
`huggingface_hub==0.36.2`, `soundfile==0.13.1`, `numpy==2.0.2`,
`scipy==1.13.1`, `scikit-learn==1.6.1`, and `numba==0.60.0`.
The omitted dependencies are multilingual, UI, or training dependencies that
are not needed by the patched English-only API path.

Import test passed:

```text
from melo.api import TTS
MeloTTS import successful
```

Model load test passed on CPU with the official Hugging Face model:

```text
Model: myshell-ai/MeloTTS-English
Speakers: EN-AU, EN-BR, EN-Default, EN-US, EN_INDIA
Cold model load time: 126.9081 seconds
Cached model load time during generation: 5.0498 seconds
```

Generated smoke-test WAV:

```text
Text: Hello, welcome to Infinia. How can I help you today?
Output: outputs/melotts/english/en_01_melotts_smoke.wav
SHA256: 381851DDBBB5CD7DF0F6CD973EE6B774AA2F71E38C3989E3302DAFE827212DB8
Exit code: 0
```

Signal validation:

```text
Sample rate: 44100
Channels: 1
Duration: 3.149863945578231 seconds
Frames: 138909
Size: 277862 bytes
RMS: 0.06855980405547263
Peak: 0.511962890625
All finite: true
```

Performance:

```text
Model load time: 5.049823099999998 seconds
Synthesis time: 235.6533478 seconds
RTF: 74.81381795261646
Total RTF including model load: 76.41709310584628
```

Evidence files:

```text
src/generate_melotts_english.py
scripts/load_melotts_english_model.py
evidence/result_snapshots/melotts_english_smoke.json
evidence/terminal_logs/melotts_english_deps_install_stdout.txt
evidence/terminal_logs/melotts_english_deps_install_stderr.txt
evidence/terminal_logs/melotts_import_test_stdout.txt
evidence/terminal_logs/melotts_import_test_stderr.txt
evidence/terminal_logs/melotts_model_load_stdout.txt
evidence/terminal_logs/melotts_model_load_stderr.txt
evidence/terminal_logs/melotts_generation_stdout.txt
evidence/terminal_logs/melotts_generation_stderr.txt
evidence/terminal_logs/melotts_validation_stdout.txt
evidence/terminal_logs/melotts_validation_stderr.txt
evidence/terminal_logs/melotts_english_smoke_hash.txt
evidence/terminal_logs/melotts_final_pip_freeze.txt
```

Remaining warnings/limitations:

- `librosa==0.9.1` uses deprecated `pkg_resources`; `setuptools==80.9.0` keeps it available.
- Python 3.9.25 is past support for some Google client libraries pulled in by `cached_path`.
- Hugging Face/Xet emitted a performance warning because `hf_xet` is not installed; regular HTTP download worked.
- `torch.load(weights_only=False)` warnings are emitted by MeloTTS/Transformers.
- CPU synthesis is very slow for this model on this laptop: RTF about 74.8 for the smoke sample.
