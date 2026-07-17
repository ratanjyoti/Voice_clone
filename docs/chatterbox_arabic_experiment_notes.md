# Chatterbox Arabic CPU Experiment Notes

Date: 2026-07-17

## Scope

This note covers the Arabic Chatterbox multilingual voice-cloning pass only. XTTS work has not been started.

## Implementation Status

- Chatterbox multilingual runs on CPU for Arabic generation in the local Windows environment.
- The model was loaded with `device="cpu"` as a string.
- WAV decoding and validation used `soundfile`; TorchCodec was not used.
- Generated WAV files were saved as PCM16 mono under `outputs/chatterbox/arabic`.
- Peak normalization to 0.95 was applied to generated samples, and the final five samples reported zero clipping samples.
- The Arabic reference set passed the earlier technical validation, with final reference average normalized WER of 3.03%.

## Parameters Selected

The automatic parameter screen tested five greeting candidates with seed 42 and the standard reference. The automatic WER winner used:

- temperature: 0.6
- cfg_weight: 0.5
- exaggeration: 0.5
- repetition_penalty: 2.0
- min_p: 0.05
- top_p: 1.0

Several candidates tied at 0.00% Whisper-based WER, so this is only the automatic selection. It is not proof of best voice similarity or naturalness.

## CPU Performance

CPU generation works but is slow. The final five-sample run averaged about 12.10 RTF, with individual final-sample RTF values from about 8.99 to 15.92. This is not suitable for real-time production on the tested CPU setup. GPU benchmarking is recommended before deployment.

## Automatic Evaluation

Faster Whisper Small on CPU was used as an automatic intelligibility proxy. The final five-sample average WER was about 20.13%, with two of five samples passing the Arabic WER <= 15% threshold.

Whisper-based WER is useful for screening missing, substituted, or repeated words, but it is not proof of native Arabic pronunciation quality. Generated Arabic is intelligible in several cases, but the automatic transcripts show pronunciation or word substitutions in longer, numeric, and expressive samples.

## Reference Comparison

The greeting was generated once with each short, standard, and long reference using the selected parameters and seed 42. Short and standard references tied at 0.00% automatic WER, while the long reference had higher automatic WER. The lowest WER reference should not be treated as the final winner until native listening review compares voice similarity and naturalness.

## Remaining Review

Native Arabic listening review is still required. The evaluation CSV includes empty human-review fields for similarity, naturalness, pronunciation, metallic artifacts, missing words, repeated words, acceptance, and reviewer notes. No human scores were invented.
