from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_CLONE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "user_voice_clones"
)
EXTERNAL_NEUTTS_DIR = (
    PROJECT_ROOT
    / "external"
    / "neutts"
)

HF_CACHE_DIR = (
    PROJECT_ROOT
    / "models"
    / "huggingface"
)

HF_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
os.environ.setdefault(
    "PHONEMIZER_ESPEAK_LIBRARY",
    r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
)

os.environ.setdefault(
    "ESPEAK_DATA_PATH",
    r"C:\Program Files\eSpeak NG\espeak-ng-data",
)

os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / "models"),
)
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(HF_CACHE_DIR),
)
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    str(HF_CACHE_DIR),
)
os.environ.setdefault(
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "300",
)
os.environ.setdefault(
    "HF_HUB_ETAG_TIMEOUT",
    "300",
)

if EXTERNAL_NEUTTS_DIR.exists():
    neutts_path = str(EXTERNAL_NEUTTS_DIR)

    if neutts_path not in sys.path:
        sys.path.insert(0, neutts_path)
        
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )


def safe_name(value: str) -> str:
    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value.strip(),
    )
    cleaned = cleaned.strip("_")

    return cleaned[:60] or "voice"


def split_long_text(
    text: str,
    max_characters: int = 220,
) -> list[str]:
    text = " ".join(text.split())

    if not text:
        return []

    sentence_parts = re.split(
        r"(?<=[.!?।])\s+",
        text,
    )

    chunks: list[str] = []
    current = ""

    for sentence in sentence_parts:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if len(candidate) <= max_characters:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(sentence) <= max_characters:
            current = sentence
            continue

        # Fallback for unusually long sentences.
        words = sentence.split()
        word_chunk = ""

        for word in words:
            candidate = (
                f"{word_chunk} {word}".strip()
                if word_chunk
                else word
            )

            if len(candidate) <= max_characters:
                word_chunk = candidate
            else:
                if word_chunk:
                    chunks.append(word_chunk)
                word_chunk = word

        if word_chunk:
            current = word_chunk

    if current:
        chunks.append(current)

    return chunks


def load_audio_stats(path: Path) -> dict:
    audio, sample_rate = sf.read(
        str(path),
        always_2d=True,
    )

    if audio.size == 0:
        raise RuntimeError(
            f"Audio contains no samples: {path}"
        )

    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_seconds": float(
            audio.shape[0] / sample_rate
        ),
        "peak": float(
            np.max(np.abs(audio))
        ),
        "rms": float(
            np.sqrt(np.mean(np.square(audio)))
        ),
    }


def validate_reference(
    audio_path: Path,
    transcript: str,
) -> dict:
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {audio_path}"
        )

    transcript = transcript.strip()

    if not transcript:
        raise ValueError(
            "Reference transcript is empty."
        )

    stats = load_audio_stats(audio_path)

    if stats["channels"] != 1:
        raise ValueError(
            "Reference audio must be mono."
        )

    if stats["duration_seconds"] < 5:
        raise ValueError(
            "Reference audio should be at least "
            "5 seconds long."
        )

    if stats["duration_seconds"] > 45:
        raise ValueError(
            "Reference audio should be no longer "
            "than 45 seconds."
        )

    if stats["rms"] <= 0.001:
        raise ValueError(
            "Reference audio appears silent."
        )

    return stats


def concatenate_audio(
    paths: list[Path],
    output_path: Path,
    silence_seconds: float = 0.18,
) -> dict:
    if not paths:
        raise RuntimeError(
            "No generated chunks are available."
        )

    combined: list[np.ndarray] = []
    expected_sample_rate: int | None = None

    for index, path in enumerate(paths):
        audio, sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )

        if expected_sample_rate is None:
            expected_sample_rate = sample_rate

        if sample_rate != expected_sample_rate:
            raise RuntimeError(
                "Generated chunks have different "
                "sample rates."
            )

        if audio.shape[1] > 1:
            audio = np.mean(
                audio,
                axis=1,
                keepdims=True,
            )

        combined.append(audio)

        if index < len(paths) - 1:
            silence_frames = int(
                silence_seconds
                * expected_sample_rate
            )
            combined.append(
                np.zeros(
                    (silence_frames, 1),
                    dtype=np.float32,
                )
            )

    waveform = np.concatenate(
        combined,
        axis=0,
    )

    # Safe headroom without changing the voice
    # unless the generated waveform reaches full scale.
    peak = float(np.max(np.abs(waveform)))

    if peak > 0.98:
        waveform = waveform * (
            0.95 / peak
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(output_path),
        waveform,
        expected_sample_rate,
        subtype="PCM_16",
    )

    return load_audio_stats(output_path)


def generate_hindi(
    chunks: list[str],
    reference_audio: Path,
    output_dir: Path,
    exaggeration: float,
    cfg_weight: float,
) -> tuple[list[Path], float]:
    import torch
    import torchaudio as ta
    from chatterbox.mtl_tts import (
        ChatterboxMultilingualTTS,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Loading Chatterbox Multilingual "
        f"on {device}..."
    )

    load_start = time.perf_counter()

    model = (
        ChatterboxMultilingualTTS
        .from_pretrained(device=device)
    )

    load_seconds = (
        time.perf_counter() - load_start
    )

    generated_paths: list[Path] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        output_path = (
            output_dir
            / "chunks"
            / f"chunk_{index:03d}.wav"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Generating Hindi chunk "
            f"{index}/{len(chunks)}"
        )
        print(chunk)

        waveform = model.generate(
            chunk,
            language_id="hi",
            audio_prompt_path=str(
                reference_audio
            ),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        waveform = waveform.detach().cpu()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        ta.save(
            str(output_path),
            waveform,
            model.sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        generated_paths.append(output_path)

    return generated_paths, load_seconds


def generate_english(
    chunks: list[str],
    reference_audio: Path,
    reference_transcript: str,
    output_dir: Path,
    model_name: str,
) -> tuple[list[Path], float]:
    import torch
    import soundfile as sf
    try:
        from neutts import NeuTTS
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NeuTTS could not be imported. "
            f"Python executable: {sys.executable}. "
            f"Expected repository: {EXTERNAL_NEUTTS_DIR}. "
            "Run this backend with .venv-neutts."
        ) from exc

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    backbone = {
        "air": "neuphonic/neutts-air",
        "nano": "neuphonic/neutts-nano",
    }[model_name]

    print(
        f"Loading {backbone} on {device}..."
    )

    load_start = time.perf_counter()

    tts = NeuTTS(
        backbone_repo=backbone,
        backbone_device=device,
        codec_repo="neuphonic/neucodec",
        codec_device=device,
    )

    load_seconds = (
        time.perf_counter() - load_start
    )

    print("Encoding reference voice...")

    reference_codes = tts.encode_reference(
        str(reference_audio)
    )

    generated_paths: list[Path] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        output_path = (
            output_dir
            / "chunks"
            / f"chunk_{index:03d}.wav"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Generating English chunk "
            f"{index}/{len(chunks)}"
        )
        print(chunk)

        waveform = tts.infer(
            chunk,
            reference_codes,
            reference_transcript,
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        ).squeeze()

        sf.write(
            str(output_path),
            waveform,
            24000,
            subtype="PCM_16",
        )

        generated_paths.append(output_path)

    return generated_paths, load_seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a user-provided English or "
            "Hindi voice clone."
        )
    )

    parser.add_argument(
        "--language",
        choices=["english", "hindi"],
        required=True,
    )
    parser.add_argument(
        "--voice-name",
        required=True,
    )
    parser.add_argument(
        "--reference-audio",
        required=True,
    )
    parser.add_argument(
        "--reference-transcript-file",
        required=True,
    )
    parser.add_argument(
        "--target-text-file",
        required=True,
    )
    parser.add_argument(
        "--english-model",
        choices=["air", "nano"],
        default="air",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.45,
    )

    args = parser.parse_args()

    reference_audio = Path(
        args.reference_audio
    ).resolve()

    transcript_path = Path(
        args.reference_transcript_file
    ).resolve()

    target_text_path = Path(
        args.target_text_file
    ).resolve()

    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript file missing: "
            f"{transcript_path}"
        )

    if not target_text_path.exists():
        raise FileNotFoundError(
            f"Target text file missing: "
            f"{target_text_path}"
        )

    reference_transcript = (
        transcript_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
    )

    target_text = (
        target_text_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
    )

    reference_stats = validate_reference(
        reference_audio,
        reference_transcript,
    )

    chunks = split_long_text(target_text)

    if not chunks:
        raise ValueError(
            "Target text is empty."
        )

    voice_id = safe_name(
        args.voice_name
    )

    run_id = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        USER_CLONE_ROOT
        / voice_id
        / args.language
        / run_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "generation_report.json"
    )

    final_output_path = (
        output_dir
        / "cloned_voice.wav"
    )

    started = time.perf_counter()

    report = {
        "status": "running",
        "voice_name": args.voice_name,
        "voice_id": voice_id,
        "language": args.language,
        "reference_audio": str(
            reference_audio
        ),
        "reference_transcript_file": str(
            transcript_path
        ),
        "target_text_file": str(
            target_text_path
        ),
        "reference_stats": reference_stats,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "output_path": str(
            final_output_path
        ),
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }

    try:
        if args.language == "hindi":
            generated_paths, load_seconds = (
                generate_hindi(
                    chunks=chunks,
                    reference_audio=reference_audio,
                    output_dir=output_dir,
                    exaggeration=args.exaggeration,
                    cfg_weight=args.cfg_weight,
                )
            )
            report["model"] = (
                "ResembleAI Chatterbox "
                "Multilingual"
            )
        else:
            generated_paths, load_seconds = (
                generate_english(
                    chunks=chunks,
                    reference_audio=reference_audio,
                    reference_transcript=(
                        reference_transcript
                    ),
                    output_dir=output_dir,
                    model_name=args.english_model,
                )
            )
            report["model"] = (
                f"NeuTTS {args.english_model.title()}"
            )

        final_stats = concatenate_audio(
            generated_paths,
            final_output_path,
        )

        total_seconds = (
            time.perf_counter() - started
        )

        report.update(
            {
                "status": "success",
                "model_load_seconds": (
                    load_seconds
                ),
                "total_generation_seconds": (
                    total_seconds
                ),
                "final_audio_stats": (
                    final_stats
                ),
                "rtf": (
                    total_seconds
                    / final_stats[
                        "duration_seconds"
                    ]
                ),
                "generated_chunks": [
                    str(path)
                    for path in generated_paths
                ],
            }
        )

        print("=" * 70)
        print("Voice cloning completed")
        print("=" * 70)
        print(
            f"Output: {final_output_path}"
        )
        print(
            f"Report: {report_path}"
        )

    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": (
                    traceback.format_exc()
                ),
            }
        )
        raise

    finally:
        report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()