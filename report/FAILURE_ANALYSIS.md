## MMS-TTS Arabic smoke test

- Model: `facebook/mms-tts-ara`
- Hardware: CPU
- Generation time: 1.8398 seconds
- Audio duration: 6.672 seconds
- RTF: 0.2758
- Listening result: Robotic
- Positive: Fast CPU inference and good RTF
- Limitation: Speech does not sound sufficiently human-like for the final Arabic pipeline
- Decision: Retain as a lightweight Arabic baseline and compare it with a stronger multilingual model

## MMS-TTS Hindi evaluation

- Model: `facebook/mms-tts-hin`
- Hardware: CPU
- Voice cloning: Not supported
- Streaming: Not supported
- Listening result: Robotic speech with incorrect Hindi pronunciation
- Main failures:
  - Unnatural rhythm and prosody
  - Incorrect pronunciation of Hindi words
  - Weak handling of names
  - Poor handling of code-mixed Hindi-English text
- Positive:
  - Lightweight enough to run on CPU
  - Useful as a speed baseline
- Decision:
  - Reject as the final Hindi pipeline
  - Retain only as a lightweight baseline for comparison