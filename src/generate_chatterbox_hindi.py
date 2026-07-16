from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "reference_audio"
    / "hindi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "chatterbox"
    / "hindi"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "chatterbox"
    / "hindi"
)


PROFILES = {
    "neutral": {
        "audio": REFERENCE_DIR / "ratan_hindi_neutral.wav",
        "transcript": REFERENCE_DIR / "ratan_hindi_neutral.txt",
        "tests": [
            {
                "id": "neutral_01_identity",
                "text": (
                    "नमस्ते, मेरा नाम रतन ज्योति है। "
                    "यह मेरी हिंदी आवाज़ के क्लोन का परीक्षण है।"
                ),
            },
            {
                "id": "neutral_02_technology",
                "text": (
                    "कृत्रिम बुद्धिमत्ता हमारे काम करने के तरीके "
                    "को तेजी से बदल रही है।"
                ),
            },
        ],
        "exaggeration": 0.45,
        "cfg_weight": 0.5,
    },
    "conversational": {
        "audio": (
            REFERENCE_DIR
            / "ratan_hindi_conversational.wav"
        ),
        "transcript": (
            REFERENCE_DIR
            / "ratan_hindi_conversational.txt"
        ),
        "tests": [
            {
                "id": "conversational_01_assistance",
                "text": (
                    "नमस्ते, मैं आपकी किस प्रकार सहायता कर सकता हूँ? "
                    "कृपया अपनी समस्या विस्तार से बताइए।"
                ),
            },
            {
                "id": "conversational_02_support",
                "text": (
                    "मैंने आपकी जानकारी देख ली है। "
                    "अब मैं आपको अगले चरण के बारे में बताता हूँ।"
                ),
            },
        ],
        "exaggeration": 0.5,
        "cfg_weight": 0.45,
    },
    "expressive": {
        "audio": (
            REFERENCE_DIR
            / "ratan_hindi_expressive.wav"
        ),
        "transcript": (
            REFERENCE_DIR
            / "ratan_hindi_expressive.txt"
        ),
        "tests": [
            {
                "id": "expressive_01_success",
                "text": (
                    "बहुत बढ़िया! हमारा परीक्षण सफल रहा "
                    "और परिणाम काफी अच्छे दिखाई दे रहे हैं।"
                ),
            },
            {
                "id": "expressive_02_progress",
                "text": (
                    "मुझे बहुत खुशी है कि हमने यह चरण पूरा कर लिया। "
                    "अब हम आगे बढ़ सकते हैं!"
                ),
            },
        ],
        "exaggeration": 0.65,
        "cfg_weight": 0.35,
    },
}


def validate_reference(
    audio_path: Path,
    transcript_path: Path,
) -> dict:
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Missing reference audio: {audio_path}"
        )

    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Missing transcript: {transcript_path}"
        )

    audio, sample_rate = sf.read(
        str(audio_path),
        always_2d=True,
    )

    if audio.size == 0:
        raise RuntimeError(
            f"Reference audio is empty: {audio_path}"
        )

    transcript = transcript_path.read_text(
        encoding="utf-8",
    ).strip()

    if not transcript:
        raise RuntimeError(
            f"Transcript is empty: {transcript_path}"
        )

    duration = audio.shape[0] / sample_rate
    peak = float(np.max(np.abs(audio)))
    rms = float(
        np.sqrt(np.mean(np.square(audio)))
    )

    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_seconds": float(duration),
        "peak": peak,
        "rms": rms,
        "transcript": transcript,
    }


def save_audio(
    output_path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    waveform = waveform.detach().cpu()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    ta.save(
        str(output_path),
        waveform,
        sample_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Hindi zero-shot voice-cloning "
            "tests with Chatterbox Multilingual."
        )
    )

    parser.add_argument(
        "--profile",
        choices=[
            "neutral",
            "conversational",
            "expressive",
            "all",
        ],
        default="neutral",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    selected_profiles = (
        list(PROFILES.keys())
        if args.profile == "all"
        else [args.profile]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Chatterbox Hindi voice-cloning evaluation")
    print("=" * 72)
    print(f"Device: {args.device}")
    print(f"Profiles: {selected_profiles}")

    print("\nLoading Chatterbox Multilingual V3...")

    load_start = time.perf_counter()

    model = ChatterboxMultilingualTTS.from_pretrained(
        device=args.device,
        t3_model="v3",
    )

    load_seconds = (
        time.perf_counter() - load_start
    )

    print(
        f"Model loaded in {load_seconds:.2f} seconds."
    )

    all_results: list[dict] = []

    for profile_name in selected_profiles:
        config = PROFILES[profile_name]

        reference_stats = validate_reference(
            config["audio"],
            config["transcript"],
        )

        print("\n" + "-" * 72)
        print(f"Profile: {profile_name}")
        print(f"Reference: {config['audio']}")
        print(
            "Reference duration: "
            f"{reference_stats['duration_seconds']:.2f}s"
        )
        print(
            f"Reference peak: {reference_stats['peak']:.4f}"
        )
        print(
            f"Reference RMS: {reference_stats['rms']:.4f}"
        )

        profile_output_dir = (
            OUTPUT_DIR / profile_name
        )

        for test in config["tests"]:
            output_path = (
                profile_output_dir
                / f"{test['id']}.wav"
            )

            print("\nGenerating:")
            print(test["text"])

            generation_start = time.perf_counter()

            waveform = model.generate(
                test["text"],
                language_id="hi",
                audio_prompt_path=str(
                    config["audio"]
                ),
                exaggeration=config["exaggeration"],
                cfg_weight=config["cfg_weight"],
            )

            generation_seconds = (
                time.perf_counter()
                - generation_start
            )

            save_audio(
                output_path,
                waveform,
                model.sr,
            )

            generated, generated_sr = sf.read(
                str(output_path),
                always_2d=True,
            )

            duration = (
                generated.shape[0]
                / generated_sr
            )

            result = {
                "model": (
                    "Chatterbox Multilingual V3"
                ),
                "language": "hi",
                "profile": profile_name,
                "test_id": test["id"],
                "text": test["text"],
                "reference_audio": str(
                    config["audio"]
                ),
                "reference_transcript": (
                    reference_stats["transcript"]
                ),
                "reference_duration_seconds": (
                    reference_stats[
                        "duration_seconds"
                    ]
                ),
                "exaggeration": (
                    config["exaggeration"]
                ),
                "cfg_weight": (
                    config["cfg_weight"]
                ),
                "device": args.device,
                "model_load_seconds": (
                    load_seconds
                ),
                "generation_seconds": (
                    generation_seconds
                ),
                "output_duration_seconds": (
                    duration
                ),
                "rtf": (
                    generation_seconds / duration
                    if duration > 0
                    else None
                ),
                "sample_rate": int(generated_sr),
                "channels": int(
                    generated.shape[1]
                ),
                "peak": float(
                    np.max(np.abs(generated))
                ),
                "rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(generated)
                        )
                    )
                ),
                "output_path": str(output_path),
            }

            all_results.append(result)

            print(f"Saved: {output_path}")
            print(
                f"Generation: {generation_seconds:.2f}s"
            )
            print(f"Duration: {duration:.2f}s")
            print(
                f"RTF: {result['rtf']:.2f}"
            )

    report_path = (
        RESULTS_DIR
        / "hindi_cloning_results.json"
    )

    report_path.write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Hindi generation completed")
    print("=" * 72)
    print(f"Generated files: {len(all_results)}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()