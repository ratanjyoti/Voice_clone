# Experiment Journey

Arabic achieved a median full-clip latency of 1.878 seconds across mixed-length inputs. The dedicated short latency sentence completed in 1.18–1.47 seconds, meeting the under-two-second batch target in all three runs. Longer Arabic numbers and paragraph samples sometimes exceeded two seconds.

## Environment

- Platform: Windows
- Device used for measured runs: CPU
- TTS framework: Hugging Face Transformers VITS
- ASR evaluation: faster-whisper
- Core generation: fully open-source Meta MMS-TTS checkpoints
- AI assistant usage: coding assistance, debugging guidance and documentation support
- Closed speech-generation APIs: none

## Models currently evaluated

| Language | Checkpoint |
|---|---|
| English | facebook/mms-tts-eng |
| Arabic | facebook/mms-tts-ara |
| Hindi | facebook/mms-tts-hin |

## Work completed

1. Created separate English, Arabic and Hindi sentence datasets.
2. Implemented data validation.
3. Implemented individual MMS speech generation.
4. Implemented repeated benchmark runs with warm-up.
5. Measured model load time, full-clip latency, audio duration and RTF.
6. Generated WAV files for all three languages.
7. Implemented faster-whisper round-trip transcription.
8. Calculated WER and CER.
9. Aggregated benchmark and intelligibility results.
10. Preserved raw outputs, terminal logs and audio hashes.

## Problems encountered

### Heavy model limitations

Heavier models such as Chatterbox were attempted but did not run reliably on the available hardware. They were not included as successful benchmark candidates.

### faster-whisper native crash

The initial faster-whisper configuration failed during model loading. The evaluation environment was stabilized by using:

- faster-whisper 1.2.1
- CTranslate2 4.6.0
- setuptools 80.9.0
- Whisper tiny on CPU with int8 compute

### ONNX Runtime DLL failure

ONNX Runtime was installed but its native DLL failed to initialize on Windows. Because faster-whisper transcription itself was functional, VAD was disabled:

```python
vad_filter=False

## SILMA Arabic voice-cloning experiment

A separate Python environment named `.venv-silma` was created to
protect the working MMS and faster-whisper environment.

Candidate:

- Model: SILMA TTS v1
- Architecture: F5-TTS
- Parameters: 150M
- Languages: Arabic MSA and English
- Voice cloning: Reference-audio conditioned
- Code licence: MIT
- Model-weight licence: Apache 2.0
- Published GPU result: approximately 0.12 RTF on RTX 4090
- Local target: Measure actual CPU latency and RTF without assuming
  published GPU performance

The first checkpoint only verifies:

1. Python version
2. FFmpeg availability
3. Isolated environment creation
4. Package installation
5. Successful Python import

No audio benchmark values are recorded until a real WAV file is
generated locally.