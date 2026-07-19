# Infinia Voice Case Study

## 1. Executive summary
The benchmark does not produce a clean production winner for all three languages. English MMS-TTS is the strongest automatic baseline because it passed the 10% WER target and CPU RTF target, but it is not a cloning system. Arabic XTTS-v2 is the best automatic Arabic WER candidate, but it failed CPU latency/RTF and speaker-cosine targets. Hindi has no production-ready winner in this run: Chatterbox Hindi was the better tested cloning candidate but still missed the WER target, was very slow, and clipped on some samples.

## 2. Hardware and methodology
All final automatic runs were performed on Windows CPU. Each language used five fixed test categories: greeting, technology, conversational, numbers, and expressive. Automatic evaluation used Faster Whisper Small for ASR/WER, SpeechBrain ECAPA-TDNN for speaker cosine similarity, WAV sidecars for generation timing/RTF, and audio statistics for clipping and silence.

Human review files were prepared anonymously under `outputs/human_review`, with the hidden mapping in `evidence/human_review_model_mapping.csv`. Real listener MOS is still pending; the summary file records this instead of inventing scores.

## 3. Models tested
English compared MMS-TTS English, NeuTTS Air, and MeloTTS English. MeloTTS was added as the final English candidate after generating and evaluating five fixed English samples.

Hindi compared MMS-TTS Hindi and Chatterbox Hindi. AI4Bharat was not implemented due time and hardware constraints and is listed as future work.

Arabic compared MMS-TTS, Chatterbox Arabic, and XTTS-v2 Arabic using the existing Arabic evaluation pipeline.

## 4. English results
English MMS-TTS averaged 8.0% WER and passed the 10% target. It also passed clipping checks and ran near the RTF target on CPU.

MeloTTS English averaged 9.67% WER across five fixed samples. Average RTF was 1.428, average generation time was 6.548 seconds, average audio duration was 4.557 seconds, and total clipping samples were 0. Manual naturalness and pronunciation scores are still pending. MeloTTS is the final English winner only if manual listening confirms it sounds more natural than MMS-TTS.

NeuTTS averaged 12.67% WER on the available five-sample run and failed the 10% WER target. Its SpeechBrain speaker cosine average was about 0.513, below the 0.75 cloning target, and CPU RTF was far too slow for production.

## 5. Hindi results
Hindi MMS-TTS averaged 52.45% WER. It was fast, but it is not a cloning model and did not pass the language quality target.

Chatterbox Hindi averaged 36.78% WER. It improved over MMS in this ASR metric, but still failed the 10% target, had very slow CPU generation, and clipped on two of five final samples.

## 6. Arabic results
Chatterbox Arabic failed the WER target. Its speaker cosine was comparatively strong, but automatic pronunciation accuracy was not good enough.

XTTS Arabic passed WER but failed CPU latency and RTF targets. Its SpeechBrain speaker cosine average was below 0.75, so the automatic cloning score did not confirm same-speaker reproduction even though WER was strong.

MMS Arabic was fast but less natural and did not provide cloning.

## 7. Cross-language comparison
The final master table is `results/final_three_language_benchmark.csv`. The clearest pattern is that fast CPU models were not necessarily natural or clone-capable, while cloning models were slow and inconsistent on CPU. Automatic WER and embedding evaluation changed the ranking: XTTS looked best for Arabic pronunciation, Chatterbox looked better for Arabic speaker similarity, and neither alone satisfied all production criteria.

## 8. Failures and lessons
- Chatterbox Arabic failed WER target.
- XTTS Arabic passed WER but failed CPU latency and RTF targets.
- MMS was fast but less natural and did not provide cloning.
- Human and embedding evaluations are necessary because WER alone can reward intelligibility while missing voice identity.
- Hindi ASR may need a stronger Indic recognizer to reduce evaluation uncertainty.

## 9. Final recommendation
English: MeloTTS is the leading final English candidate if manual listening confirms naturalness; otherwise use MMS-TTS as the automatic baseline. Keep NeuTTS as a research cloning candidate only.

Hindi: no production recommendation from this benchmark. Chatterbox is the best tested cloning candidate, but it needs quality and speed improvements.

Arabic: use XTTS-v2 as the best quality candidate for further work, while clearly reporting that CPU latency/RTF and speaker similarity failed target.

## 10. Remaining limitations
Real MOS scores are not yet collected. NeuTTS fixed-sentence generation needs a clean rerun. Latency should be rerun on target deployment hardware. Model and reference-audio licenses must be verified before external distribution.
## 11. 2026-07-19 NeuTTS Improved Follow-Up

NeuTTS remained the most human-sounding English candidate in manual listening, but it could not be promoted without a passing reproducible run. A new TTS-only normalizer, improved generation script, improved evaluator, comparison CSV, and fine-tuning scaffold were added.

The improved run did not generate any WAVs because `.venvs/.venv-neutts` does not currently contain an importable official `neutts` package. The evaluator recorded 5 failed samples, average WER 100.0%, average RTF 0.0, and 0 clipping samples. The number sentence did not improve because no new audio existed to evaluate.

Decision: NeuTTS is still a research/naturalness candidate only. MeloTTS remains the reproducible English winner pending final manual MOS confirmation. Fine-tuning was not performed; only a scaffold was created and dataset validation found no accepted clips.

## 12. 2026-07-19 NeuTTS Improved Recovery Result

NeuTTS was recovered by using the local source checkout at `external/neutts`, system eSpeak NG, and a clean proxy-free generation command. The improved run generated all five English WAVs and sidecars successfully.

Automatic results: average WER 1.82%, average RTF 27.692, total clipping samples 0, failed samples 0. The number/date sentence improved to 0.0% WER. NeuTTS-improved therefore passes the automatic English WER/stability/clipping gate, but final manual MOS/listening confirmation is still pending.

Recommendation update: NeuTTS-improved is the automatic English winner pending final MOS confirmation. MeloTTS remains the fallback reproducible English winner if manual listening does not confirm NeuTTS-improved naturalness.
