from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR.parent))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "300")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_audio" / "hindi"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chatterbox" / "hindi"
RESULTS_DIR = PROJECT_ROOT / "evidence" / "result_snapshots" / "chatterbox" / "hindi"
CSV_PATH = PROJECT_ROOT / "results" / "chatterbox_hindi_evaluation.csv"


PROFILES = {
    "neutral": {
        "audio": REFERENCE_DIR / "ratan_hindi_neutral.wav",
        "transcript": REFERENCE_DIR / "ratan_hindi_neutral.txt",
        "tests": [
            {
                "id": "neutral_01_identity",
                "text": "नमस्ते, मेरा नाम रतन ज्योति है। यह मेरी हिंदी आवाज़ के क्लोन का परीक्षण है।",
            },
            {
                "id": "neutral_02_technology",
                "text": "कृत्रिम बुद्धिमत्ता हमारे काम करने के तरीके को तेज़ी से बदल रही है।",
            },
        ],
        "exaggeration": 0.45,
        "cfg_weight": 0.50,
    },
    "conversational": {
        "audio": REFERENCE_DIR / "ratan_hindi_conversational.wav",
        "transcript": REFERENCE_DIR / "ratan_hindi_conversational.txt",
        "tests": [
            {
                "id": "conversational_01_assistance",
                "text": "नमस्ते, मैं आपकी किस प्रकार सहायता कर सकता हूँ? कृपया अपनी समस्या विस्तार से बताइए।",
            },
            {
                "id": "conversational_02_support",
                "text": "मैंने आपकी जानकारी देख ली है। अब मैं आपको अगले चरण के बारे में बताता हूँ।",
            },
            {
                "id": "conversational_03_schedule",
                "text": "जी हाँ, मैं आज शाम तक आपका काम पूरा करने की कोशिश करूँगा।",
            },
            {
                "id": "conversational_04_clarify",
                "text": "ठीक है, पहले हम आपकी ज़रूरत समझ लेते हैं, फिर सही समाधान चुनते हैं।",
            },
            {
                "id": "conversational_05_followup",
                "text": "अगर आपको कोई दिक्कत लगे, तो आप मुझे बताइए, मैं तुरंत मदद करूँगा।",
            },
        ],
        "exaggeration": 0.50,
        "cfg_weight": 0.45,
    },
    "expressive": {
        "audio": REFERENCE_DIR / "ratan_hindi_expressive.wav",
        "transcript": REFERENCE_DIR / "ratan_hindi_expressive.txt",
        "tests": [
            {
                "id": "expressive_01_success",
                "text": "बहुत बढ़िया! हमारा परीक्षण सफल रहा और परिणाम काफी अच्छे दिखाई दे रहे हैं।",
            },
            {
                "id": "expressive_02_progress",
                "text": "मुझे बहुत खुशी है कि हमने यह चरण पूरा कर लिया। अब हम आगे बढ़ सकते हैं!",
            },
            {
                "id": "expressive_03_excited",
                "text": "वाह, यह सच में कमाल का परिणाम है, और आवाज़ पहले से ज़्यादा साफ़ लग रही है!",
            },
            {
                "id": "expressive_04_reassure",
                "text": "चिंता मत कीजिए, हम धीरे-धीरे हर समस्या को समझकर ठीक कर लेंगे।",
            },
            {
                "id": "expressive_05_finish",
                "text": "शानदार, अब हमारा हिंदी वॉइस क्लोन अंतिम जाँच के लिए तैयार है!",
            },
        ],
        "exaggeration": 0.65,
        "cfg_weight": 0.35,
    },
}


def validate_reference(audio_path: Path, transcript_path: Path) -> dict:
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing reference audio: {audio_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript: {transcript_path}")

    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    if audio.size == 0:
        raise RuntimeError(f"Reference audio is empty: {audio_path}")

    transcript = transcript_path.read_text(encoding="utf-8", errors="replace").strip()
    duration = audio.shape[0] / sample_rate
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))

    if not np.isfinite(audio).all():
        raise RuntimeError(f"Reference has non-finite samples: {audio_path}")
    if duration < 3:
        raise RuntimeError(f"Reference is too short: {duration:.2f}s")
    if rms <= 0.001:
        raise RuntimeError(f"Reference appears silent: {audio_path}")

    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_seconds": float(duration),
        "peak": peak,
        "rms": rms,
        "transcript": transcript,
    }


def save_audio(output_path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    ta.save(str(output_path), waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)


def append_csv(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "created_at",
        "model",
        "language",
        "profile",
        "test_id",
        "text",
        "device",
        "status",
        "model_load_seconds",
        "generation_seconds",
        "output_duration_seconds",
        "rtf",
        "sample_rate",
        "channels",
        "peak",
        "rms",
        "output_path",
        "error",
        "similarity_score",
        "naturalness_score",
        "pronunciation_score",
        "metallic_level",
        "missing_words",
        "repeated_words",
        "listening_notes",
        "accepted",
    ]
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hindi zero-shot voice cloning tests with Chatterbox Multilingual."
    )
    parser.add_argument(
        "--profile",
        choices=["neutral", "conversational", "expressive", "all"],
        default="neutral",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    selected_profiles = list(PROFILES) if args.profile == "all" else [args.profile]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Chatterbox Hindi voice-cloning evaluation")
    print("=" * 72)
    print(f"Device: {args.device}")
    print(f"HF cache: {HF_CACHE_DIR}")
    print(f"Profiles: {', '.join(selected_profiles)}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    print("\nLoading Chatterbox Multilingual...")
    load_start = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(device=args.device)
    load_seconds = time.perf_counter() - load_start
    print(f"Model loaded in {load_seconds:.2f} seconds.")

    all_results: list[dict] = []

    for profile_name in selected_profiles:
        config = PROFILES[profile_name]
        reference_stats = validate_reference(config["audio"], config["transcript"])

        print("\n" + "-" * 72)
        print(f"Profile: {profile_name}")
        print(f"Reference: {config['audio']}")
        print(f"Reference duration: {reference_stats['duration_seconds']:.2f}s")
        print(f"Reference sample rate: {reference_stats['sample_rate']} Hz")
        print(f"Reference channels: {reference_stats['channels']}")
        print(f"Reference peak: {reference_stats['peak']:.4f}")
        print(f"Reference RMS: {reference_stats['rms']:.4f}")

        profile_output_dir = OUTPUT_DIR / profile_name

        for test in config["tests"]:
            output_path = profile_output_dir / f"{test['id']}.wav"
            result = {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": "ResembleAI/chatterbox multilingual",
                "language": "hi",
                "profile": profile_name,
                "test_id": test["id"],
                "text": test["text"],
                "reference_audio": str(config["audio"]),
                "reference_transcript": reference_stats["transcript"],
                "exaggeration": config["exaggeration"],
                "cfg_weight": config["cfg_weight"],
                "device": args.device,
                "model_load_seconds": load_seconds,
                "output_path": str(output_path),
                "status": "failed",
            }

            print("\nGenerating:")
            print(f"ID: {test['id']}")
            print(test["text"])

            try:
                generation_start = time.perf_counter()
                waveform = model.generate(
                    test["text"],
                    language_id="hi",
                    audio_prompt_path=str(config["audio"]),
                    exaggeration=config["exaggeration"],
                    cfg_weight=config["cfg_weight"],
                )
                generation_seconds = time.perf_counter() - generation_start

                save_audio(output_path, waveform, model.sr)
                generated, generated_sr = sf.read(str(output_path), always_2d=True)
                duration = generated.shape[0] / generated_sr
                rms = float(np.sqrt(np.mean(np.square(generated))))
                peak = float(np.max(np.abs(generated)))

                if duration <= 0 or rms <= 0.0005:
                    raise RuntimeError("Generated audio is empty or silent.")

                result.update(
                    {
                        "status": "success",
                        "generation_seconds": generation_seconds,
                        "output_duration_seconds": duration,
                        "rtf": generation_seconds / duration,
                        "sample_rate": int(generated_sr),
                        "channels": int(generated.shape[1]),
                        "peak": peak,
                        "rms": rms,
                    }
                )
                print(f"Saved: {output_path}")
                print(f"Generation: {generation_seconds:.2f}s")
                print(f"Duration: {duration:.2f}s")
                print(f"RTF: {result['rtf']:.2f}")

            except Exception as exc:
                result.update({"error": str(exc), "traceback": traceback.format_exc()})
                print(f"FAILED: {exc}")

            all_results.append(result)

    report_path = RESULTS_DIR / f"{args.profile}_hindi_cloning_results.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    append_csv(all_results)

    successful = sum(1 for row in all_results if row.get("status") == "success")
    failed = len(all_results) - successful

    print("\n" + "=" * 72)
    print("Hindi generation completed")
    print("=" * 72)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print(f"CSV: {CSV_PATH}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()



