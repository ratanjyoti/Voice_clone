import argparse
import csv
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "neutts" / "english"
REPORT_DIR = PROJECT_ROOT / "evidence" / "result_snapshots" / "neutts"
SMOKE_REPORT = (
    PROJECT_ROOT
    / "evidence"
    / "result_snapshots"
    / "neutts_english_smoke.json"
)
CSV_PATH = PROJECT_ROOT / "results" / "raw_runs.csv"
VOICE_PROFILE_DIR = PROJECT_ROOT / "data" / "voice_profiles"

MODEL_OPTIONS = {
    "Nano Q8": "neuphonic/neutts-nano-q8-gguf",
    "Nano Q4": "neuphonic/neutts-nano-q4-gguf",
    "Air Q8": "neuphonic/neutts-air-q8-gguf",
    "Air Q4": "neuphonic/neutts-air-q4-gguf",
}

DEFAULT_TEXT = (
    "Hello, welcome to Infinia. "
    "I am ready to help you with your request today."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate English cloned speech with NeuTTS."
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--profile", default="Ratan Neutral")
    parser.add_argument(
        "--model",
        default="Nano Q8",
        choices=sorted(MODEL_OPTIONS),
    )
    parser.add_argument(
        "--mode",
        default="Standard",
        choices=["Standard", "Sentence-by-sentence", "Streaming"],
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Encode and cache the selected voice profile without synthesis.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_voice_profile(profile_name: str) -> dict:
    for path in sorted(VOICE_PROFILE_DIR.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        if profile.get("profile_name") == profile_name:
            profile["_profile_path"] = str(path)
            return profile

    available = [
        json.loads(path.read_text(encoding="utf-8")).get("profile_name")
        for path in sorted(VOICE_PROFILE_DIR.glob("*.json"))
    ]
    raise FileNotFoundError(
        f"Voice profile {profile_name!r} was not found. "
        f"Available profiles: {available}"
    )


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def validate_profile_files(profile: dict) -> tuple[Path, Path, Path, str]:
    reference_audio = resolve_project_path(profile["reference_audio"])
    reference_text = resolve_project_path(profile["reference_text"])
    reference_codes = resolve_project_path(profile["reference_codes"])

    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio missing: {reference_audio}")
    if not reference_text.exists():
        raise FileNotFoundError(f"Reference transcript missing: {reference_text}")

    transcript = reference_text.read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError(f"Reference transcript is empty: {reference_text}")

    audio_info = sf.info(str(reference_audio))
    if audio_info.samplerate != 24000:
        raise ValueError(
            f"Reference audio must be 24000 Hz, got {audio_info.samplerate}: "
            f"{reference_audio}"
        )
    if audio_info.channels != 1:
        raise ValueError(
            f"Reference audio must be mono, got {audio_info.channels} channels: "
            f"{reference_audio}"
        )

    return reference_audio, reference_text, reference_codes, transcript


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def wav_stats(path: Path) -> dict:
    data, sample_rate = sf.read(str(path), always_2d=True)
    if data.size == 0:
        raise RuntimeError("Generated WAV contains no samples")

    duration = data.shape[0] / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(data))))
    peak = float(np.max(np.abs(data)))

    return {
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "frames": int(data.shape[0]),
        "duration_seconds": duration,
        "rms": rms,
        "peak": peak,
        "all_finite": bool(np.isfinite(data).all()),
        "size_bytes": int(path.stat().st_size),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_result_to_csv(result: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pipeline",
        "model",
        "language",
        "sentence_id",
        "text",
        "speaker",
        "device",
        "status",
        "model_load_time_seconds",
        "generation_time_seconds",
        "audio_duration_seconds",
        "rtf",
        "sample_rate",
        "channels",
        "rms",
        "peak",
        "output_path",
        "report_path",
        "error",
    ]

    file_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists or CSV_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(
            {
                "pipeline": "neutts",
                "model": result.get("model", ""),
                "language": "english",
                "sentence_id": result.get("sentence_id", ""),
                "text": result.get("text", ""),
                "speaker": result.get("profile_name", ""),
                "device": "cpu",
                "status": result.get("status", ""),
                "model_load_time_seconds": result.get(
                    "model_load_time_seconds", ""
                ),
                "generation_time_seconds": result.get(
                    "synthesis_time_seconds", ""
                ),
                "audio_duration_seconds": result.get("duration_seconds", ""),
                "rtf": result.get("rtf", ""),
                "sample_rate": result.get("sample_rate", ""),
                "channels": result.get("channels", ""),
                "rms": result.get("rms", ""),
                "peak": result.get("peak", ""),
                "output_path": result.get("output_path", ""),
                "report_path": result.get("report_path", ""),
                "error": result.get("error", ""),
            }
        )


def import_neutts():
    try:
        from neutts import NeuTTS
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The NeuTTS package is not installed in .venv-neutts. "
            "The requested pip package 'neutts[all]' was not available from "
            "the configured package index. Install the official NeuTTS package "
            "or source checkout, then rerun this script."
        ) from exc

    return NeuTTS


def encode_reference(tts, reference_audio: Path, reference_codes: Path):
    reference_codes.parent.mkdir(parents=True, exist_ok=True)

    if reference_codes.exists():
        load_start = time.perf_counter()
        codes = torch.load(str(reference_codes), map_location="cpu")
        return codes, 0.0, time.perf_counter() - load_start, True

    encode_start = time.perf_counter()
    codes = tts.encode_reference(str(reference_audio))
    encode_time = time.perf_counter() - encode_start
    torch.save(codes, str(reference_codes))
    return codes, encode_time, 0.0, False


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_slug = slugify(args.profile)
    model_slug = slugify(args.model)
    mode_slug = slugify(args.mode)
    output_path = OUTPUT_DIR / f"{timestamp}_{profile_slug}_{model_slug}.wav"
    report_path = REPORT_DIR / f"{timestamp}_{profile_slug}_{model_slug}.json"

    result = {
        "status": "failed",
        "pipeline": "neutts",
        "language": "english",
        "profile_name": args.profile,
        "model": args.model,
        "backbone_repo": MODEL_OPTIONS[args.model],
        "codec_repo": "neuphonic/neucodec",
        "mode": args.mode,
        "device": "cpu",
        "text": args.text,
        "sentence_id": f"neutts_{timestamp}",
        "output_path": str(output_path),
        "report_path": str(report_path),
        "reference_status": "Not encoded",
        "model_loaded": False,
    }

    try:
        profile = load_voice_profile(args.profile)
        reference_audio, reference_text, reference_codes, transcript = (
            validate_profile_files(profile)
        )

        result.update(
            {
                "voice_profile": profile,
                "reference_audio": str(reference_audio),
                "reference_text_path": str(reference_text),
                "reference_transcript": transcript,
                "reference_codes_path": str(reference_codes),
                "reference_status": (
                    "Encoded" if reference_codes.exists() else "Needs encoding"
                ),
            }
        )

        NeuTTS = import_neutts()

        print("Loading NeuTTS model...")
        load_start = time.perf_counter()
        tts = NeuTTS(
            backbone_repo=MODEL_OPTIONS[args.model],
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu",
        )
        model_load_time = time.perf_counter() - load_start
        result["model_loaded"] = True
        result["model_load_time_seconds"] = model_load_time

        print("Encoding or loading reference voice...")
        reference_codes_value, encode_time, code_load_time, cache_hit = (
            encode_reference(tts, reference_audio, reference_codes)
        )
        result.update(
            {
                "reference_status": "Encoded",
                "reference_cache_hit": cache_hit,
                "reference_encoding_time_seconds": encode_time,
                "reference_code_load_time_seconds": code_load_time,
            }
        )

        if args.encode_only:
            result["status"] = "success"
            result["generation_status"] = "Skipped"
            save_json(report_path, result)
            save_json(SMOKE_REPORT, result)
            print(f"Reference encoded: {reference_codes}")
            return

        texts = [args.text]
        if args.mode == "Sentence-by-sentence":
            texts = split_sentences(args.text)
        elif args.mode == "Streaming":
            result["streaming_note"] = (
                "Streaming is treated as standard generation until NeuTTS "
                "exposes a streaming API in the installed package."
            )

        print("Generating speech...")
        synthesis_start = time.perf_counter()
        audio_parts = []
        first_audio_time = None

        for text_part in texts:
            audio_part = tts.infer(
                text_part,
                reference_codes_value,
                transcript,
            )
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - synthesis_start
            audio_parts.append(np.asarray(audio_part, dtype=np.float32))

        audio = (
            np.concatenate(audio_parts)
            if len(audio_parts) > 1
            else audio_parts[0]
        )
        sf.write(str(output_path), audio, 24000, subtype="PCM_16")
        synthesis_time = time.perf_counter() - synthesis_start

        stats = wav_stats(output_path)
        if not stats["all_finite"]:
            raise RuntimeError("Generated WAV contains NaN or infinite samples")
        if stats["rms"] <= 0.0005:
            raise RuntimeError("Generated WAV appears silent")

        result.update(
            {
                "status": "success",
                "generation_status": "Completed",
                "synthesis_time_seconds": synthesis_time,
                "time_to_first_audio_seconds": first_audio_time,
                "rtf": synthesis_time / stats["duration_seconds"],
                **stats,
            }
        )

    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "generation_status": "Failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"FAILED: {exc}")

    save_json(report_path, result)
    save_json(SMOKE_REPORT, result)
    append_result_to_csv(result)
    print(f"Report: {report_path}")

    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
