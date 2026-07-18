from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import os
import re
import shutil
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_NEUTTS_DIR = (
    PROJECT_ROOT
    / "external"
    / "neutts"
)

if EXTERNAL_NEUTTS_DIR.exists():
    external_neutts_text = str(
        EXTERNAL_NEUTTS_DIR
    )

    if external_neutts_text not in sys.path:
        sys.path.insert(
            0,
            external_neutts_text,
        )

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
RESULTS_DIR = PROJECT_ROOT / "results"
VOICE_PROFILE_DIR = PROJECT_ROOT / "data" / "voice_profiles"
HUMAN_EVALUATION_CSV = RESULTS_DIR / "human_voice_evaluations.csv"
AUTOMATIC_EVALUATION_CSV = RESULTS_DIR / "automatic_tts_evaluation.csv"
USER_VOICE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "user_voice_clones"
)

USER_VOICE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "user_voice_clones"
)

USER_CLONE_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "generate_user_voice_clone.py"
)

CHATTERBOX_ENV = PROJECT_ROOT / ".venv"

MMS_ENV = PROJECT_ROOT / ".venv"
MELOTTS_ENV = PROJECT_ROOT / ".venv-melotts"
NEUTTS_ENV = PROJECT_ROOT / ".venv-neutts"
XTTS_ARABIC_ENV = PROJECT_ROOT / ".venv-xtts-arabic"

REFERENCE_AUDIO_OPTIONS = {
    "Ratan Neutral": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "english"
        / "ratan_neutral.wav"
    ),
    "Ratan Conversational": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "english"
        / "ratan_conversational.wav"
    ),
    "Ratan Expressive": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "english"
        / "ratan_expressive.wav"
    ),
    "Hindi cloning reference": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "ratan_reference_22050_mono.wav"
    ),
    "Arabic MSA short": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "arabic"
        / "professional_msa"
        / "arabic_reference_short.wav"
    ),
    "Arabic MSA standard": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "arabic"
        / "professional_msa"
        / "arabic_reference_standard.wav"
    ),
    "Arabic MSA long": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "arabic"
        / "professional_msa"
        / "arabic_reference_long.wav"
    ),
}

REFERENCE_TEXT_OPTIONS = {
    "Ratan Neutral": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "english"
        / "ratan_neutral.txt"
    ),
    "Ratan Conversational": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "english"
        / "ratan_conversational.txt"
    ),
    "Ratan Expressive": (
        PROJECT_ROOT
        / "data"
        / "reference_audio"
        / "english"
        / "ratan_expressive.txt"
    ),
}

NEUTTS_MODELS = ["Nano Q8", "Nano Q4", "Air Q8", "Air Q4"]
NEUTTS_GENERATION_MODES = [
    "Standard",
    "Sentence-by-sentence",
    "Streaming",
]
MODEL_CONFIG = {
    "MMS English": {
        "environment": MMS_ENV,
        "script": PROJECT_ROOT / "src" / "generate_mms.py",
        "report": (
            PROJECT_ROOT
            / "evidence"
            / "result_snapshots"
            / "mms_english_smoke.json"
        ),
        "audio_dir": OUTPUTS_DIR / "mms" / "english",
        "supports_voice_cloning": False,
    },
    "MeloTTS English": {
        "environment": MELOTTS_ENV,
        "script": (
            PROJECT_ROOT
            / "src"
            / "generate_melotts_english.py"
        ),
        "report": (
            PROJECT_ROOT
            / "evidence"
            / "result_snapshots"
            / "melotts_english_smoke.json"
        ),
        "audio_dir": OUTPUTS_DIR / "melotts" / "english",
        "supports_voice_cloning": False,
    },
    "NeuTTS English": {
        "environment": NEUTTS_ENV,
        "script": (
            PROJECT_ROOT
            / "src"
            / "generate_neutts_english.py"
        ),
        "report": (
            PROJECT_ROOT
            / "evidence"
            / "result_snapshots"
            / "neutts_english_smoke.json"
        ),
        "audio_dir": OUTPUTS_DIR / "neutts" / "english",
        "supports_voice_cloning": True,
    },
    "Chatterbox Arabic": {
        "environment": CHATTERBOX_ENV,
        "script": (
            PROJECT_ROOT
            / "src"
            / "generate_chatterbox_arabic.py"
        ),
        "report": (
            PROJECT_ROOT
            / "evidence"
            / "result_snapshots"
            / "chatterbox"
            / "arabic"
            / "chatterbox_arabic_summary.json"
        ),
        "audio_dir": OUTPUTS_DIR / "chatterbox" / "arabic",
        "supports_voice_cloning": True,
        "default_arguments": [
            "--mode", "final",
            "--device", "cpu",
            "--reference", "standard",
            "--seed", "42",
            "--temperature", "0.6",
            "--cfg-weight", "0.5",
            "--exaggeration", "0.5",
            "--repetition-penalty", "2.0",
            "--min-p", "0.05",
            "--top-p", "1.0",
        ],
    },
    "XTTS-v2 Arabic": {
        "environment": XTTS_ARABIC_ENV,
        "script": (
            PROJECT_ROOT
            / "src"
            / "generate_xtts_arabic.py"
        ),
        "report": (
            PROJECT_ROOT
            / "evidence"
            / "result_snapshots"
            / "xtts"
            / "arabic"
            / "xtts_arabic_summary.json"
        ),
        "audio_dir": OUTPUTS_DIR / "xtts" / "arabic",
        "supports_voice_cloning": True,
        "default_arguments": [
            "--mode", "final",
            "--device", "cpu",
            "--reference-strategy", "multi",
            "--seed", "42",
            "--configuration-id", "C",
            "--temperature", "0.65",
            "--top-k", "40",
            "--top-p", "0.80",
            "--repetition-penalty", "5.0",
            "--length-penalty", "1.0",
            "--speed", "1.0",
        ],
    },

}

def get_python_path(environment_path: Path) -> Path:
    return environment_path / "Scripts" / "python.exe"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_text(path: Path) -> str | None:
    if not path.exists():
        return None

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:
        return None

def safe_voice_name(value: str) -> str:
    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value.strip(),
    )
    cleaned = cleaned.strip("_")
    return cleaned[:60] or "voice"


def save_uploaded_reference(
    audio_bytes: bytes,
    voice_name: str,
    language: str,
    transcript: str,
    source_name: str,
) -> dict:
    voice_id = safe_voice_name(
        voice_name
    )

    voice_dir = (
        USER_VOICE_DATA_DIR
        / voice_id
        / language
    )

    voice_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        voice_dir
        / "reference_input"
    )

    extension = (
        Path(source_name).suffix.lower()
        if source_name
        else ".wav"
    )

    if extension not in {
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".flac",
        ".webm",
    }:
        extension = ".wav"

    raw_path = raw_path.with_suffix(
        extension
    )

    raw_path.write_bytes(audio_bytes)

    reference_path = (
        voice_dir
        / "reference.wav"
    )

    transcript_path = (
        voice_dir
        / "reference.txt"
    )

    target_path = (
        voice_dir
        / "target_text.txt"
    )

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(reference_path),
    ]

    completed = subprocess.run(
        ffmpeg_command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Audio conversion failed:\n"
            + completed.stderr[-3000:]
        )

    transcript_path.write_text(
        transcript.strip(),
        encoding="utf-8",
    )

    profile_path = (
        voice_dir
        / "voice_profile.json"
    )

    profile = {
        "voice_name": voice_name,
        "voice_id": voice_id,
        "language": language,
        "reference_audio": project_relative(
            reference_path
        ),
        "reference_text": project_relative(
            transcript_path
        ),
        "source_file": project_relative(
            raw_path
        ),
        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
        "consent_confirmed": True,
    }

    profile_path.write_text(
        json.dumps(
            profile,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "voice_id": voice_id,
        "voice_dir": voice_dir,
        "reference_path": reference_path,
        "transcript_path": transcript_path,
        "target_path": target_path,
        "profile_path": profile_path,
    }


def find_latest_user_clone(
    voice_id: str,
    language: str,
) -> tuple[Path | None, dict | None]:
    root = (
        USER_VOICE_OUTPUT_DIR
        / voice_id
        / language
    )

    if not root.exists():
        return None, None

    candidates = sorted(
        root.rglob("generation_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for report_path in candidates:
        report = load_json(report_path)

        if (
            report
            and report.get("status")
            == "success"
        ):
            audio_path = Path(
                report.get("output_path", "")
            )

            if audio_path.exists():
                return audio_path, report

    return None, None

def load_voice_profiles() -> dict[str, dict]:
    profiles = {}

    if not VOICE_PROFILE_DIR.exists():
        return profiles

    for profile_path in sorted(VOICE_PROFILE_DIR.glob("*.json")):
        profile = load_json(profile_path)
        if not profile:
            continue

        profile_name = profile.get("profile_name")
        if profile_name:
            profile["_profile_path"] = str(profile_path)
            profiles[profile_name] = profile

    return profiles


def resolve_project_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def get_profile_status(profile: dict) -> dict:
    audio_path = resolve_project_path(profile.get("reference_audio"))
    text_path = resolve_project_path(profile.get("reference_text"))
    codes_path = resolve_project_path(profile.get("reference_codes"))

    return {
        "audio_path": audio_path,
        "text_path": text_path,
        "codes_path": codes_path,
        "audio_exists": bool(audio_path and audio_path.exists()),
        "text_exists": bool(text_path and text_path.exists()),
        "codes_exist": bool(codes_path and codes_path.exists()),
        "validated": bool(profile.get("validated")),
    }
def find_latest_json(
    model_name: str,
) -> tuple[Path | None, dict | None]:
    config = MODEL_CONFIG[model_name]
    configured_report = config["report"]

    if configured_report.exists():
        return configured_report, load_json(configured_report)

    snapshot_dir = EVIDENCE_DIR / "result_snapshots"

    if not snapshot_dir.exists():
        return None, None

    keywords = {
        "MMS English": ["mms", "english"],
        "MeloTTS English": ["melotts", "english"],
        "NeuTTS English": ["neutts", "english"],
    }

    matching_files = []

    for path in snapshot_dir.rglob("*.json"):
        lower_name = path.name.lower()

        if all(
            keyword in lower_name
            for keyword in keywords[model_name]
        ):
            matching_files.append(path)

    if not matching_files:
        return None, None

    latest = max(
        matching_files,
        key=lambda path: path.stat().st_mtime,
    )

    return latest, load_json(latest)


def get_wav_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        return []

    return sorted(
        audio_dir.glob("*.wav"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def get_model_status(model_name: str) -> dict:
    config = MODEL_CONFIG[model_name]

    environment = config["environment"]
    python_path = get_python_path(environment)
    script_path = config["script"]
    audio_dir = config["audio_dir"]

    report_path, report = find_latest_json(model_name)
    wav_files = get_wav_files(audio_dir)

    status = "Not started"

    if report:
        report_status = str(
            report.get("status", "")
        ).lower()

        if report_status == "success":
            status = "Working"
        elif report_status == "failed":
            status = "Failed"
        else:
            status = "Report found"

    elif wav_files:
        status = "Audio generated"

    elif script_path.exists() and python_path.exists():
        status = "Ready to run"

    elif not python_path.exists():
        status = "Environment missing"

    elif not script_path.exists():
        status = "Script missing"

    return {
        "name": model_name,
        "status": status,
        "environment_exists": environment.exists(),
        "python_exists": python_path.exists(),
        "script_exists": script_path.exists(),
        "report_path": report_path,
        "report": report,
        "wav_files": wav_files,
    }


def status_icon(status: str) -> str:
    status_lower = status.lower()

    if status_lower == "working":
        return "âœ…"

    if status_lower in {
        "ready to run",
        "audio generated",
    }:
        return "ðŸŸ¡"

    if status_lower == "failed":
        return "âŒ"

    return "âšª"


def run_script(
    python_path: Path,
    script_path: Path,
    arguments: list[str],
):
    command = [
        str(python_path),
        "-u",
        str(script_path),
        *arguments,
    ]

    process =     process_environment = os.environ.copy()

    process_environment["PHONEMIZER_ESPEAK_LIBRARY"] = (
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )
    process_environment["ESPEAK_DATA_PATH"] = (
        r"C:\Program Files\eSpeak NG\espeak-ng-data"
    )

    process_environment.setdefault(
        "HF_HUB_OFFLINE",
        "1",
    )
    process_environment.setdefault(
        "TRANSFORMERS_OFFLINE",
        "1",
    )

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=process_environment,
    )

    return process, command


def show_report(report: dict):
    col1, col2, col3, col4 = st.columns(4)

    load_time = report.get(
        "model_load_time_seconds",
        report.get("load_time_seconds"),
    )

    synthesis_time = report.get(
        "synthesis_time_seconds",
        report.get("generation_time_seconds"),
    )

    duration = report.get(
        "duration_seconds",
        report.get("audio_duration_seconds"),
    )

    rtf = report.get("rtf")

    col1.metric(
        "Model load",
        f"{load_time:.2f}s"
        if isinstance(load_time, (int, float))
        else "N/A",
    )

    col2.metric(
        "Synthesis",
        f"{synthesis_time:.2f}s"
        if isinstance(synthesis_time, (int, float))
        else "N/A",
    )

    col3.metric(
        "Audio duration",
        f"{duration:.2f}s"
        if isinstance(duration, (int, float))
        else "N/A",
    )

    col4.metric(
        "RTF",
        f"{rtf:.3f}"
        if isinstance(rtf, (int, float))
        else "N/A",
    )

    with st.expander("View full JSON report"):
        st.json(report)


def show_audio_file(
    audio_path: Path,
    title: str | None = None,
):
    if title:
        st.markdown(f"#### {title}")

    if not audio_path.exists():
        st.error(f"Audio file does not exist: {audio_path}")
        return

    try:
        audio_bytes = audio_path.read_bytes()
        st.audio(audio_bytes, format="audio/wav")
    except Exception as exc:
        st.error(f"Could not play audio: {exc}")
        return

    size_mb = audio_path.stat().st_size / 1024 / 1024
    modified = datetime.fromtimestamp(
        audio_path.stat().st_mtime
    )

    st.caption(
        f"{audio_path.name} â€¢ "
        f"{size_mb:.2f} MB â€¢ "
        f"{modified:%Y-%m-%d %H:%M:%S}"
    )

    with st.expander("Show audio path"):
        st.code(str(audio_path), language=None)


def select_audio_for_model(
    model_name: str,
    key_prefix: str,
) -> Path | None:
    config = MODEL_CONFIG[model_name]
    wav_files = get_wav_files(config["audio_dir"])

    if not wav_files:
        st.warning(f"No WAV files found for {model_name}.")
        return None

    filenames = [path.name for path in wav_files]

    selected_filename = st.selectbox(
        f"Select {model_name} audio",
        filenames,
        key=f"{key_prefix}_{model_name}",
    )

    return next(
        path
        for path in wav_files
        if path.name == selected_filename
    )


def save_comparison_report(report: dict) -> Path:
    comparison_dir = (
        EVIDENCE_DIR
        / "result_snapshots"
        / "manual_comparisons"
    )

    comparison_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        comparison_dir
        / f"voice_comparison_{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


EVALUATION_COLUMNS = [
    "evaluation_id",
    "sample_id",
    "language",
    "profile",
    "model",
    "reference_name",
    "reference_audio",
    "generated_audio",
    "expected_text",
    "similarity_score",
    "naturalness_score",
    "pronunciation_score",
    "metallic_level",
    "missing_words",
    "repeated_words",
    "listening_notes",
    "accepted",
    "reviewer",
    "created_at",
    "updated_at",
]

EDITABLE_EVALUATION_COLUMNS = [
    "language",
    "profile",
    "model",
    "reference_name",
    "expected_text",
    "similarity_score",
    "naturalness_score",
    "pronunciation_score",
    "metallic_level",
    "missing_words",
    "repeated_words",
    "listening_notes",
    "accepted",
    "reviewer",
]


def project_relative(path: Path | str | None) -> str:
    if path is None:
        return ""

    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return str(path_obj)


def normalize_choice(value: object, allowed: list[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    for choice in allowed:
        if normalized == choice.lower():
            return choice
    return default


def load_evaluations() -> pd.DataFrame:
    if not HUMAN_EVALUATION_CSV.exists():
        return pd.DataFrame(columns=EVALUATION_COLUMNS)

    try:
        dataframe = pd.read_csv(
            HUMAN_EVALUATION_CSV,
            keep_default_na=False,
        )
    except Exception:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)

    if "metallic_level" not in dataframe.columns and "metallic" in dataframe.columns:
        dataframe["metallic_level"] = dataframe["metallic"]
    if "listening_notes" not in dataframe.columns and "notes" in dataframe.columns:
        dataframe["listening_notes"] = dataframe["notes"]

    for column in EVALUATION_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[EVALUATION_COLUMNS].copy()

    for score_column in [
        "similarity_score",
        "naturalness_score",
        "pronunciation_score",
    ]:
        dataframe[score_column] = pd.to_numeric(
            dataframe[score_column], errors="coerce"
        ).fillna(0).astype(int).clip(0, 5)

    dataframe["metallic_level"] = dataframe["metallic_level"].apply(
        lambda value: normalize_choice(
            value, ["None", "Mild", "Strong"], "None"
        )
    )
    for column in ["missing_words", "repeated_words", "accepted"]:
        dataframe[column] = dataframe[column].apply(
            lambda value: normalize_choice(value, ["Yes", "No"], "No")
        )

    return dataframe


def save_evaluations(dataframe: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clean = dataframe.copy()

    for column in EVALUATION_COLUMNS:
        if column not in clean.columns:
            clean[column] = ""

    clean = clean[EVALUATION_COLUMNS]
    temporary_path = HUMAN_EVALUATION_CSV.with_suffix(".tmp.csv")
    clean.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(HUMAN_EVALUATION_CSV)


def evaluation_id_for(generated_audio: Path) -> str:
    relative_path = project_relative(generated_audio)
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]


def discover_reference_audio() -> dict[str, Path]:
    reference_root = PROJECT_ROOT / "data" / "reference_audio"
    if not reference_root.exists():
        return {}

    references: dict[str, Path] = {}
    for path in sorted(reference_root.rglob("*.wav")):
        label = project_relative(path)
        references[label] = path
    return references


def discover_generated_audio() -> list[dict]:
    if not OUTPUTS_DIR.exists():
        return []

    rows: list[dict] = []
    for path in sorted(
        OUTPUTS_DIR.rglob("*.wav"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        relative = path.relative_to(OUTPUTS_DIR)
        parts = list(relative.parts)
        model = parts[0] if len(parts) >= 2 else "Unknown"
        language = parts[1] if len(parts) >= 3 else "Unknown"
        profile = parts[2] if len(parts) >= 4 else "General"

        rows.append(
            {
                "path": path,
                "relative_path": project_relative(path),
                "sample_id": path.stem,
                "model": model.replace("_", " ").title(),
                "language": language.replace("_", " ").title(),
                "profile": profile.replace("_", " ").title(),
                "modified": datetime.fromtimestamp(path.stat().st_mtime),
            }
        )

    return rows


def infer_best_reference(
    generated: dict,
    references: dict[str, Path],
) -> str | None:
    if not references:
        return None

    generated_tokens = {
        token.lower()
        for token in (
            generated.get("language", "")
            + " "
            + generated.get("profile", "")
            + " "
            + generated.get("sample_id", "")
        ).replace("_", " ").split()
        if token
    }

    best_label = None
    best_score = -1
    for label, path in references.items():
        candidate_tokens = {
            token.lower()
            for token in (
                label.replace("/", " ")
                + " "
                + path.stem.replace("_", " ")
            ).split()
            if token
        }
        score = len(generated_tokens.intersection(candidate_tokens))

        if "hindi" in generated_tokens and "hindi" in candidate_tokens:
            score += 5
        if "arabic" in generated_tokens and "arabic" in candidate_tokens:
            score += 5
        if "english" in generated_tokens and "english" in candidate_tokens:
            score += 5
        if "conversational" in generated_tokens and "conversational" in candidate_tokens:
            score += 4
        if "neutral" in generated_tokens and "neutral" in candidate_tokens:
            score += 4
        if "expressive" in generated_tokens and "expressive" in candidate_tokens:
            score += 4

        if score > best_score:
            best_label = label
            best_score = score

    return best_label


def find_expected_text(generated_audio: Path) -> str:
    candidates = [
        generated_audio.with_suffix(".txt"),
        generated_audio.parent / f"{generated_audio.stem}.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        if candidate.suffix.lower() == ".txt":
            return load_text(candidate) or ""

        report = load_json(candidate)
        if report:
            for key in ["text", "expected_text", "input_text", "sentence"]:
                value = report.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return ""


def load_automatic_metrics() -> pd.DataFrame:
    if not AUTOMATIC_EVALUATION_CSV.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            AUTOMATIC_EVALUATION_CSV,
            keep_default_na=False,
        )
    except Exception:
        return pd.DataFrame()


def get_automatic_metrics_for(
    generated_audio: Path,
    sample_id: str,
) -> dict:
    automatic = load_automatic_metrics()
    if automatic.empty:
        return {}

    path_columns = [
        column
        for column in ["generated_audio", "generated_path", "audio_path"]
        if column in automatic.columns
    ]

    target_relative = project_relative(generated_audio).replace("\\", "/").lower()
    for column in path_columns:
        normalized = (
            automatic[column]
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.lower()
        )
        matches = automatic[normalized == target_relative]
        if not matches.empty:
            return matches.iloc[-1].to_dict()

    if "sample_id" in automatic.columns:
        matches = automatic[
            automatic["sample_id"].astype(str) == str(sample_id)
        ]
        if not matches.empty:
            return matches.iloc[-1].to_dict()

    return {}


def automatic_acceptance_recommendation(
    similarity: int,
    naturalness: int,
    pronunciation: int,
    metallic: str,
    missing_words: str,
    repeated_words: str,
) -> bool:
    return (
        similarity >= 4
        and naturalness >= 4
        and pronunciation >= 4
        and metallic in {"None", "Mild"}
        and missing_words == "No"
        and repeated_words == "No"
    )


def upsert_evaluation(record: dict) -> tuple[str, pd.DataFrame]:
    dataframe = load_evaluations()
    evaluation_id = record["evaluation_id"]
    now = datetime.now().isoformat(timespec="seconds")

    matching_index = dataframe.index[
        dataframe["evaluation_id"].astype(str) == evaluation_id
    ].tolist()

    if matching_index:
        index = matching_index[0]
        record["created_at"] = dataframe.at[index, "created_at"] or now
        record["updated_at"] = now
        for key, value in record.items():
            dataframe.at[index, key] = value
        action = "updated"
    else:
        record["created_at"] = now
        record["updated_at"] = now
        dataframe = pd.concat(
            [dataframe, pd.DataFrame([record])],
            ignore_index=True,
        )
        action = "saved"

    save_evaluations(dataframe)
    return action, dataframe


def dataframe_to_excel_bytes(dataframe: pd.DataFrame) -> bytes | None:
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            dataframe.to_excel(
                writer,
                index=False,
                sheet_name="Voice evaluations",
            )
        return output.getvalue()
    except Exception:
        return None


def show_score_guide() -> None:
    with st.expander("Scoring guide", expanded=False):
        st.markdown(
            """
            **Similarity:** 5 = almost exactly the same speaker; 1 = different speaker.  
            **Naturalness:** 5 = human and fluid; 1 = unusable.  
            **Pronunciation:** 5 = every word is correct; 1 = difficult to understand.  
            **Accepted:** similarity, naturalness and pronunciation are at least 4; metallic is None or Mild; no missing or repeated words.
            """
        )


def render_review_workspace() -> None:
    st.markdown(
        """
        <div class="review-hero">
          <div>
            <div class="eyebrow">HUMAN LISTENING BENCHMARK</div>
            <h1>Voice Review Studio</h1>
            <p>Listen to the reference and cloned speech side by side, score every quality dimension, and persist results in one editable evaluation sheet.</p>
          </div>
          <div class="hero-badge">Reproducible â€¢ Auditable â€¢ Multilingual</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    generated_rows = discover_generated_audio()
    references = discover_reference_audio()
    evaluations = load_evaluations()

    reviewed_paths = set(
        evaluations["generated_audio"].astype(str).tolist()
        if not evaluations.empty
        else []
    )
    accepted_count = int(
        (evaluations["accepted"] == "Yes").sum()
        if not evaluations.empty
        else 0
    )
    total_outputs = len(generated_rows)
    reviewed_count = len(reviewed_paths)
    pending_count = max(total_outputs - reviewed_count, 0)
    acceptance_rate = (
        accepted_count / reviewed_count * 100
        if reviewed_count
        else 0.0
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Generated clips", total_outputs)
    metric_columns[1].metric("Reviewed", reviewed_count)
    metric_columns[2].metric("Pending", pending_count)
    metric_columns[3].metric("Acceptance rate", f"{acceptance_rate:.0f}%")

    progress = reviewed_count / total_outputs if total_outputs else 0.0
    st.progress(progress, text=f"Review progress: {reviewed_count}/{total_outputs}")

    review_tab, saved_tab, insights_tab = st.tabs(
        ["Review queue", "Saved evaluations", "Insights"]
    )

    with review_tab:
        if not generated_rows:
            st.warning(
                f"No generated WAV files were found under {OUTPUTS_DIR}."
            )
            return

        filter_container = st.container(border=True)
        with filter_container:
            st.markdown("#### Find a sample")
            filter_columns = st.columns([1, 1, 1, 1, 2])

            languages = sorted({row["language"] for row in generated_rows})
            models = sorted({row["model"] for row in generated_rows})
            profiles = sorted({row["profile"] for row in generated_rows})

            selected_language = filter_columns[0].selectbox(
                "Language", ["All", *languages], key="review_language_filter"
            )
            selected_model = filter_columns[1].selectbox(
                "Model", ["All", *models], key="review_model_filter"
            )
            selected_profile = filter_columns[2].selectbox(
                "Profile", ["All", *profiles], key="review_profile_filter"
            )
            selected_status = filter_columns[3].selectbox(
                "Status",
                ["Pending", "Reviewed", "All"],
                key="review_status_filter",
            )
            search_text = filter_columns[4].text_input(
                "Search filename",
                placeholder="conversation, neutral, assistance...",
                key="review_search_filter",
            )

        filtered_rows = []
        for row in generated_rows:
            is_reviewed = row["relative_path"] in reviewed_paths
            if selected_language != "All" and row["language"] != selected_language:
                continue
            if selected_model != "All" and row["model"] != selected_model:
                continue
            if selected_profile != "All" and row["profile"] != selected_profile:
                continue
            if selected_status == "Pending" and is_reviewed:
                continue
            if selected_status == "Reviewed" and not is_reviewed:
                continue
            if search_text and search_text.lower() not in row["relative_path"].lower():
                continue
            filtered_rows.append(row)

        if not filtered_rows:
            st.info("No clips match the selected filters.")
            return

        row_by_label = {
            (
                f"{'âœ“' if row['relative_path'] in reviewed_paths else 'â—‹'} "
                f"{row['sample_id']}  Â·  {row['model']} / "
                f"{row['language']} / {row['profile']}"
            ): row
            for row in filtered_rows
        }

        selected_label = st.selectbox(
            "Select generated sample",
            list(row_by_label.keys()),
            key="review_sample_selector",
        )
        selected = row_by_label[selected_label]
        generated_audio = selected["path"]
        evaluation_id = evaluation_id_for(generated_audio)

        existing_record = None
        if not evaluations.empty:
            matches = evaluations[
                evaluations["evaluation_id"].astype(str) == evaluation_id
            ]
            if not matches.empty:
                existing_record = matches.iloc[0].to_dict()

        inferred_reference = infer_best_reference(selected, references)
        current_reference = (
            existing_record.get("reference_audio")
            if existing_record
            else inferred_reference
        )
        if current_reference not in references:
            current_reference = inferred_reference or next(iter(references), None)

        status_badge = "Reviewed" if existing_record else "Pending review"
        st.markdown(
            f"""
            <div class="sample-strip">
              <div><span class="status-pill">{status_badge}</span></div>
              <div><strong>{selected['sample_id']}</strong><br><span>{selected['relative_path']}</span></div>
              <div><strong>{selected['model']}</strong><br><span>{selected['language']} Â· {selected['profile']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metadata_columns = st.columns([1, 1, 1])
        language_value = metadata_columns[0].text_input(
            "Language",
            value=str(existing_record.get("language", selected["language"]) if existing_record else selected["language"]),
            key=f"review_language_{evaluation_id}",
        )
        profile_value = metadata_columns[1].text_input(
            "Voice profile",
            value=str(existing_record.get("profile", selected["profile"]) if existing_record else selected["profile"]),
            key=f"review_profile_{evaluation_id}",
        )
        model_value = metadata_columns[2].text_input(
            "Model",
            value=str(existing_record.get("model", selected["model"]) if existing_record else selected["model"]),
            key=f"review_model_{evaluation_id}",
        )

        if not references:
            st.error(
                "No reference WAV files were found under data/reference_audio."
            )
            return

        reference_labels = list(references.keys())
        reference_index = (
            reference_labels.index(current_reference)
            if current_reference in reference_labels
            else 0
        )
        selected_reference_label = st.selectbox(
            "Reference voice",
            reference_labels,
            index=reference_index,
            key=f"review_reference_{evaluation_id}",
        )
        reference_audio = references[selected_reference_label]

        default_expected_text = (
            str(existing_record.get("expected_text", ""))
            if existing_record
            else find_expected_text(generated_audio)
        )
        expected_text = st.text_area(
            "Expected sentence",
            value=default_expected_text,
            height=90,
            placeholder="Paste the exact sentence that should be spoken.",
            key=f"review_expected_text_{evaluation_id}",
        )

        playback_left, playback_right = st.columns(2)
        with playback_left:
            with st.container(border=True):
                st.markdown("### 1 Â· Original reference")
                st.caption(selected_reference_label)
                show_audio_file(reference_audio)
        with playback_right:
            with st.container(border=True):
                st.markdown("### 2 Â· Generated clone")
                st.caption(selected["relative_path"])
                show_audio_file(generated_audio)

        automatic_metrics = get_automatic_metrics_for(
            generated_audio,
            selected["sample_id"],
        )
        if automatic_metrics:
            with st.container(border=True):
                st.markdown("#### Automatic evidence")
                metric_keys = [
                    ("speaker_cosine", "Speaker cosine"),
                    ("predicted_mos", "Predicted MOS"),
                    ("wer", "WER"),
                    ("cer", "CER"),
                    ("rtf", "RTF"),
                    ("latency_sec", "Latency"),
                ]
                available = [item for item in metric_keys if item[0] in automatic_metrics]
                if available:
                    columns = st.columns(len(available))
                    for column, (key, label) in zip(columns, available):
                        value = automatic_metrics.get(key)
                        column.metric(label, value if value != "" else "N/A")
                with st.expander("View automatic result row"):
                    st.json(automatic_metrics)
        else:
            st.caption(
                "Automatic metrics will appear here when "
                "results/automatic_tts_evaluation.csv is available."
            )

        show_score_guide()

        default_similarity = int(existing_record.get("similarity_score", 3)) if existing_record else 3
        default_naturalness = int(existing_record.get("naturalness_score", 3)) if existing_record else 3
        default_pronunciation = int(existing_record.get("pronunciation_score", 3)) if existing_record else 3
        default_metallic = str(existing_record.get("metallic_level", "None")) if existing_record else "None"
        default_missing = str(existing_record.get("missing_words", "No")) if existing_record else "No"
        default_repeated = str(existing_record.get("repeated_words", "No")) if existing_record else "No"
        default_notes = str(existing_record.get("listening_notes", "")) if existing_record else ""
        default_reviewer = str(existing_record.get("reviewer", "")) if existing_record else ""

        st.markdown("### Human scores")
        score_columns = st.columns(3)
        similarity = score_columns[0].slider(
            "Speaker similarity",
            1,
            5,
            default_similarity,
            key=f"review_similarity_{evaluation_id}",
        )
        naturalness = score_columns[1].slider(
            "Naturalness (MOS)",
            1,
            5,
            default_naturalness,
            key=f"review_naturalness_{evaluation_id}",
        )
        pronunciation = score_columns[2].slider(
            "Pronunciation",
            1,
            5,
            default_pronunciation,
            key=f"review_pronunciation_{evaluation_id}",
        )

        issue_columns = st.columns(3)
        metallic_options = ["None", "Mild", "Strong"]
        metallic = issue_columns[0].selectbox(
            "Metallic artifacts",
            metallic_options,
            index=metallic_options.index(
                default_metallic if default_metallic in metallic_options else "None"
            ),
            key=f"review_metallic_{evaluation_id}",
        )
        yes_no = ["No", "Yes"]
        missing_words = issue_columns[1].selectbox(
            "Missing words",
            yes_no,
            index=yes_no.index(default_missing if default_missing in yes_no else "No"),
            key=f"review_missing_{evaluation_id}",
        )
        repeated_words = issue_columns[2].selectbox(
            "Repeated words",
            yes_no,
            index=yes_no.index(default_repeated if default_repeated in yes_no else "No"),
            key=f"review_repeated_{evaluation_id}",
        )

        recommendation = automatic_acceptance_recommendation(
            similarity,
            naturalness,
            pronunciation,
            metallic,
            missing_words,
            repeated_words,
        )

        decision_columns = st.columns([1, 1, 2])
        decision_columns[0].metric(
            "Average human score",
            f"{(similarity + naturalness + pronunciation) / 3:.2f}/5",
        )
        decision_columns[1].metric(
            "Rule recommendation",
            "Accept" if recommendation else "Reject",
        )
        accepted_default = (
            str(existing_record.get("accepted", "No")) == "Yes"
            if existing_record
            else recommendation
        )
        accepted = decision_columns[2].toggle(
            "Accepted",
            value=accepted_default,
            help="You may override the rule recommendation, but explain the reason in Notes.",
            key=f"review_accepted_{evaluation_id}",
        )

        reviewer_columns = st.columns([1, 2])
        reviewer = reviewer_columns[0].text_input(
            "Reviewer",
            value=default_reviewer,
            placeholder="Name or listener ID",
            key=f"review_reviewer_{evaluation_id}",
        )
        notes = reviewer_columns[1].text_area(
            "Listening notes",
            value=default_notes,
            height=110,
            placeholder=(
                "Example: Voice is close and conversational, but the ending "
                "is mildly metallic and one word sounds compressed."
            ),
            key=f"review_notes_{evaluation_id}",
        )

        save_label = "Update evaluation" if existing_record else "Save evaluation"
        if st.button(
            save_label,
            type="primary",
            use_container_width=True,
            key=f"save_review_{evaluation_id}",
        ):
            record = {
                "evaluation_id": evaluation_id,
                "sample_id": selected["sample_id"],
                "language": language_value.strip(),
                "profile": profile_value.strip(),
                "model": model_value.strip(),
                "reference_name": Path(selected_reference_label).stem,
                "reference_audio": selected_reference_label,
                "generated_audio": selected["relative_path"],
                "expected_text": expected_text.strip(),
                "similarity_score": similarity,
                "naturalness_score": naturalness,
                "pronunciation_score": pronunciation,
                "metallic_level": metallic,
                "missing_words": missing_words,
                "repeated_words": repeated_words,
                "listening_notes": notes.strip(),
                "accepted": "Yes" if accepted else "No",
                "reviewer": reviewer.strip(),
            }
            action, _ = upsert_evaluation(record)
            st.success(
                f"Evaluation {action}. Spreadsheet: "
                f"{project_relative(HUMAN_EVALUATION_CSV)}"
            )
            st.rerun()

    with saved_tab:
        st.markdown("### Evaluation spreadsheet")
        st.caption(
            "Edit saved values directly, tick Delete for unwanted rows, then save the table."
        )

        evaluations = load_evaluations()
        if evaluations.empty:
            st.info("No evaluations have been saved yet.")
        else:
            table = evaluations.copy()
            table.insert(0, "Delete", False)
            edited = st.data_editor(
                table,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                disabled=[
                    "evaluation_id",
                    "sample_id",
                    "reference_audio",
                    "generated_audio",
                    "created_at",
                    "updated_at",
                ],
                key="saved_evaluation_editor",
            )

            action_columns = st.columns([1, 1, 1, 3])
            if action_columns[0].button(
                "Save table changes",
                type="primary",
                use_container_width=True,
            ):
                cleaned = edited[~edited["Delete"]].drop(columns=["Delete"])
                cleaned["updated_at"] = datetime.now().isoformat(timespec="seconds")
                save_evaluations(cleaned)
                st.success("Spreadsheet changes saved.")
                st.rerun()

            csv_bytes = evaluations.to_csv(index=False).encode("utf-8-sig")
            action_columns[1].download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="human_voice_evaluations.csv",
                mime="text/csv",
                use_container_width=True,
            )

            excel_bytes = dataframe_to_excel_bytes(evaluations)
            action_columns[2].download_button(
                "Download Excel",
                data=excel_bytes or b"",
                file_name="human_voice_evaluations.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                disabled=excel_bytes is None,
                use_container_width=True,
            )

            st.caption(f"Stored at: {HUMAN_EVALUATION_CSV}")

    with insights_tab:
        st.markdown("### Review insights")
        evaluations = load_evaluations()
        if evaluations.empty:
            st.info("Save at least one evaluation to view insights.")
        else:
            summary_columns = st.columns(4)
            summary_columns[0].metric("Rows", len(evaluations))
            summary_columns[1].metric(
                "Mean similarity",
                f"{evaluations['similarity_score'].mean():.2f}/5",
            )
            summary_columns[2].metric(
                "Mean naturalness",
                f"{evaluations['naturalness_score'].mean():.2f}/5",
            )
            summary_columns[3].metric(
                "Mean pronunciation",
                f"{evaluations['pronunciation_score'].mean():.2f}/5",
            )

            by_language = (
                evaluations.groupby("language", dropna=False)
                .agg(
                    samples=("evaluation_id", "count"),
                    similarity=("similarity_score", "mean"),
                    naturalness=("naturalness_score", "mean"),
                    pronunciation=("pronunciation_score", "mean"),
                    accepted=("accepted", lambda values: (values == "Yes").mean() * 100),
                )
                .reset_index()
            )
            st.markdown("#### Language summary")
            st.dataframe(
                by_language.style.format(
                    {
                        "similarity": "{:.2f}",
                        "naturalness": "{:.2f}",
                        "pronunciation": "{:.2f}",
                        "accepted": "{:.0f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            model_scores = (
                evaluations.groupby("model")[[
                    "similarity_score",
                    "naturalness_score",
                    "pronunciation_score",
                ]]
                .mean()
                .sort_values("naturalness_score", ascending=False)
            )
            st.markdown("#### Mean scores by model")
            st.bar_chart(model_scores)

            failure_columns = st.columns(3)
            failure_columns[0].metric(
                "Metallic clips",
                int((evaluations["metallic_level"] != "None").sum()),
            )
            failure_columns[1].metric(
                "Missing-word clips",
                int((evaluations["missing_words"] == "Yes").sum()),
            )
            failure_columns[2].metric(
                "Repeated-word clips",
                int((evaluations["repeated_words"] == "Yes").sum()),
            )



def load_csv_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            keep_default_na=False,
            encoding="utf-8",
        )
    except Exception:
        return pd.DataFrame()


def numeric_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def render_arabic_results() -> None:
    st.header("Arabic voice-cloning results")

    comparison_path = RESULTS_DIR / "arabic_model_comparison.csv"
    comparison_summary_path = (
        EVIDENCE_DIR
        / "result_snapshots"
        / "arabic_model_comparison_summary.json"
    )
    xtts_summary_path = (
        EVIDENCE_DIR
        / "result_snapshots"
        / "xtts"
        / "arabic"
        / "xtts_arabic_summary.json"
    )
    chatterbox_summary_path = (
        EVIDENCE_DIR
        / "result_snapshots"
        / "chatterbox"
        / "arabic"
        / "chatterbox_arabic_summary.json"
    )

    comparison = load_csv_dataframe(comparison_path)
    comparison_summary = load_json(comparison_summary_path) or {}
    xtts_summary = load_json(xtts_summary_path) or {}
    chatterbox_summary = load_json(chatterbox_summary_path) or {}

    if comparison.empty:
        st.warning(
            "Arabic model comparison has not been generated yet."
        )
        st.code(str(comparison_path), language=None)
        return

    winner = comparison_summary.get(
        "automatic_winner",
        "Not available",
    )

    xtts_row = comparison[
        comparison["model"].astype(str) == "XTTS-v2"
    ]
    chatterbox_row = comparison[
        comparison["model"].astype(str) == "Chatterbox multilingual"
    ]

    xtts_metrics = (
        xtts_row.iloc[0].to_dict()
        if not xtts_row.empty
        else {}
    )
    chatterbox_metrics = (
        chatterbox_row.iloc[0].to_dict()
        if not chatterbox_row.empty
        else {}
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Automatic winner", winner)
    metric_columns[1].metric(
        "XTTS avg WER",
        f"{numeric_value(xtts_metrics.get('average_wer')) * 100:.2f}%",
    )
    metric_columns[2].metric(
        "XTTS WER pass",
        f"{int(numeric_value(xtts_metrics.get('wer_pass_count')))}/5",
    )
    metric_columns[3].metric(
        "XTTS avg RTF",
        f"{numeric_value(xtts_metrics.get('average_rtf')):.2f}",
    )
    metric_columns[4].metric(
        "Clipping pass",
        f"{int(numeric_value(xtts_metrics.get('clipping_pass_count')))}/5",
    )

    st.caption(
        "Automatic WER is a screening proxy. Native Arabic listening review is still required."
    )

    comparison_tab, samples_tab, review_tab, evidence_tab = st.tabs(
        [
            "Model comparison",
            "Final samples",
            "Native review",
            "Evidence",
        ]
    )

    with comparison_tab:
        display = comparison.copy()
        for column in [
            "average_wer",
            "median_wer",
            "maximum_wer",
        ]:
            if column in display.columns:
                display[column] = (
                    pd.to_numeric(display[column], errors="coerce")
                    * 100
                ).round(2)
        for column in ["average_generation_time", "average_rtf"]:
            if column in display.columns:
                display[column] = pd.to_numeric(
                    display[column], errors="coerce"
                ).round(3)

        st.dataframe(display, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### XTTS-v2")
            st.json(xtts_summary)
        with col2:
            st.markdown("#### Chatterbox")
            st.json(chatterbox_summary)

    with samples_tab:
        xtts_eval = load_csv_dataframe(
            RESULTS_DIR / "xtts_arabic_evaluation.csv"
        )
        chatterbox_eval = load_csv_dataframe(
            RESULTS_DIR / "chatterbox_arabic_evaluation.csv"
        )

        if xtts_eval.empty:
            st.warning("XTTS Arabic evaluation CSV is missing.")
            return

        for _, xtts_sample in xtts_eval.iterrows():
            test_id = str(xtts_sample.get("test_id", ""))
            with st.container(border=True):
                st.subheader(test_id)
                expected_text = str(
                    xtts_sample.get("expected_text", "")
                    or xtts_sample.get("reference_text", "")
                )
                if expected_text:
                    st.info(expected_text)

                columns = st.columns(2)
                with columns[0]:
                    st.markdown("#### XTTS-v2 Arabic")
                    xtts_audio = Path(
                        str(xtts_sample.get("output_path", ""))
                    )
                    if xtts_audio.exists():
                        show_audio_file(xtts_audio)
                    st.metric(
                        "WER",
                        f"{numeric_value(xtts_sample.get('wer_percentage')):.2f}%",
                    )
                    st.metric(
                        "RTF",
                        f"{numeric_value(xtts_sample.get('rtf')):.2f}",
                    )

                with columns[1]:
                    st.markdown("#### Chatterbox Arabic")
                    chatter_match = chatterbox_eval[
                        chatterbox_eval["test_id"].astype(str) == test_id
                    ] if not chatterbox_eval.empty else pd.DataFrame()
                    if not chatter_match.empty:
                        chatter_sample = chatter_match.iloc[0]
                        chatter_audio = Path(
                            str(chatter_sample.get("output_path", ""))
                        )
                        if chatter_audio.exists():
                            show_audio_file(chatter_audio)
                        chatter_wer = chatter_sample.get(
                            "wer_percent",
                            chatter_sample.get("wer_percentage", ""),
                        )
                        st.metric(
                            "WER",
                            f"{numeric_value(chatter_wer):.2f}%",
                        )
                        st.metric(
                            "RTF",
                            f"{numeric_value(chatter_sample.get('rtf')):.2f}",
                        )
                    else:
                        st.warning("No matching Chatterbox row found.")

    with review_tab:
        st.markdown("#### Review package")
        st.write(
            "Use these artifacts for native Arabic listening review. Human score fields are intentionally blank."
        )
        review_paths = [
            docs_path
            for docs_path in [
                PROJECT_ROOT / "docs" / "arabic_native_review_guide.md",
                RESULTS_DIR / "arabic_native_review_sheet.csv",
                RESULTS_DIR / "xtts_arabic_evaluation.csv",
                RESULTS_DIR / "chatterbox_arabic_evaluation.csv",
            ]
        ]
        for review_path in review_paths:
            status = "available" if review_path.exists() else "missing"
            st.write(f"**{review_path.name}:** {status}")
            st.code(str(review_path), language=None)

    with evidence_tab:
        st.markdown("#### Evidence files")
        evidence_paths = [
            comparison_path,
            comparison_summary_path,
            xtts_summary_path,
            chatterbox_summary_path,
            EVIDENCE_DIR
            / "result_snapshots"
            / "xtts"
            / "arabic"
            / "xtts_completion_manifest.csv",
            PROJECT_ROOT / "docs" / "xtts_arabic_experiment_notes.md",
            PROJECT_ROOT / "docs" / "licenses" / "xtts_v2_license_notes.md",
        ]
        for evidence_path in evidence_paths:
            status = "available" if evidence_path.exists() else "missing"
            st.write(f"**{evidence_path.name}:** {status}")
            st.code(str(evidence_path), language=None)
st.set_page_config(
    page_title="Infinia Voice Lab",
    page_icon="ðŸŽ™ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #111827;
        --muted: #526071;
        --panel: #ffffff;
        --panel-soft: #f8fafc;
        --line: #d8e0ea;
        --brand: #2563eb;
        --brand-2: #0891b2;
        --accent: #f59e0b;
    }
    .stApp {
        color: var(--ink);
        background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 44%, #f8fafc 100%);
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, .92) !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
    }
    .review-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        padding: 26px 28px;
        margin: 4px 0 22px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(135deg, #ffffff 0%, #eaf4ff 58%, #eefdfa 100%);
        box-shadow: 0 14px 38px rgba(15, 23, 42, .08);
    }
    .review-hero h1 { color: #0f172a; margin: 4px 0 8px; font-size: 2.1rem; letter-spacing: 0; }
    .review-hero p { color: var(--muted); max-width: 760px; margin: 0; }
    .eyebrow { font-size: .72rem; letter-spacing: .12em; color: #1d4ed8; font-weight: 800; }
    .hero-badge, .status-pill {
        white-space: nowrap;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
        padding: 8px 12px;
        border-radius: 999px;
        color: #1e3a8a !important;
        font-size: .82rem;
        font-weight: 700;
    }
    .sample-strip {
        display: grid;
        grid-template-columns: auto 1.7fr 1fr;
        gap: 18px;
        align-items: center;
        margin: 16px 0;
        padding: 16px 18px;
        border: 1px solid var(--line);
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
    }
    .sample-strip span { color: var(--muted); font-size: .82rem; }
        .sample-strip strong { color: #0f172a; }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100% !important;
        width: 100% !important;
    }    
    .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 650; }
    div[data-baseweb="tab-list"] { gap: 10px; }
    div[data-baseweb="tab"] { border-radius: 8px; padding: 10px 16px; }
    div[data-baseweb="select"] > div, input, textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-color: var(--line) !important;
    }
    @media (max-width: 900px) {
        .review-hero { align-items: flex-start; flex-direction: column; }
        .sample-strip { grid-template-columns: 1fr; }
    }
</style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## ðŸŽ™ï¸ Infinia Voice Lab")
st.caption(
    "Run open-source TTS experiments, inspect evidence, and build an auditable multilingual voice benchmark."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Clone Your Voice",
        "Run Experiment",
        "Generated Audio",
        "Voice Comparison",
        "Arabic Results",
        "Review Workspace",
        "Reports",
    ],
)


if page == "Overview":
    st.header("Pipeline overview")

    statuses = [
        get_model_status(model_name)
        for model_name in MODEL_CONFIG
    ]

    columns = st.columns(len(statuses))

    for column, model_status in zip(columns, statuses):
        with column:
            icon = status_icon(model_status["status"])

            st.subheader(
                f"{icon} {model_status['name']}"
            )

            st.write(
                f"**Status:** {model_status['status']}"
            )

            st.write(
                "**Voice cloning:**",
                (
                    "âœ… Supported"
                    if MODEL_CONFIG[
                        model_status["name"]
                    ]["supports_voice_cloning"]
                    else "âŒ Not supported"
                ),
            )

            st.write(
                "Environment:",
                (
                    "âœ…"
                    if model_status["environment_exists"]
                    else "âŒ"
                ),
            )

            st.write(
                "Python executable:",
                (
                    "âœ…"
                    if model_status["python_exists"]
                    else "âŒ"
                ),
            )

            st.write(
                "Generation script:",
                (
                    "âœ…"
                    if model_status["script_exists"]
                    else "âŒ"
                ),
            )

            st.write(
                f"Audio files: "
                f"{len(model_status['wav_files'])}"
            )

            if model_status["report"]:
                show_report(model_status["report"])

            if model_status["report_path"]:
                st.code(
                    str(model_status["report_path"]),
                    language=None,
                )

    st.divider()

    st.subheader("Reference voice status")

    for reference_name, reference_path in (
        REFERENCE_AUDIO_OPTIONS.items()
    ):
        if reference_path.exists():
            st.success(
                f"{reference_name}: available"
            )
        else:
            st.warning(
                f"{reference_name}: not found at "
                f"{reference_path}"
            )

    st.divider()

    st.subheader("What the statuses mean")

    st.markdown(
        """
- **Working:** a successful JSON report was found.
- **Ready to run:** environment and script exist, but no result exists.
- **Audio generated:** WAV exists, but a matching report was not found.
- **Failed:** the latest report says the experiment failed.
- **Environment missing:** its virtual environment does not exist.
- **Script missing:** the required generation script does not exist.
        """
    )
elif page == "Clone Your Voice":
    st.header("Clone Your Voice")

    st.write(
        "Upload or record a clear voice sample, "
        "provide its exact transcript, and generate "
        "new speech in the same voice."
    )

    st.warning(
        "Only clone your own voice or a voice for "
        "which you have explicit permission."
    )

    consent = st.checkbox(
        "I confirm that this is my voice, or I have "
        "clear permission from the speaker to clone it.",
        value=False,
    )

    setup_container = st.container(
        border=True
    )

    with setup_container:
        st.markdown(
            "### 1 Â· Voice and language"
        )

        setup_columns = st.columns(2)

        voice_name = setup_columns[0].text_input(
            "Voice name",
            placeholder=(
                "Example: Ratan Hindi Natural"
            ),
        )

        selected_language_label = (
            setup_columns[1].selectbox(
                "Language",
                [
                    "English",
                    "Hindi",
                ],
            )
        )

        selected_language = (
            selected_language_label.lower()
        )

        st.caption(
            "English uses NeuTTS. "
            "Hindi uses Chatterbox Multilingual."
        )

    audio_container = st.container(
        border=True
    )

    with audio_container:
        st.markdown(
            "### 2 Â· Add reference recording"
        )

        input_method = st.radio(
            "Recording method",
            [
                "Upload audio",
                "Record with microphone",
            ],
            horizontal=True,
        )

        reference_source = None
        source_name = ""

        if input_method == "Upload audio":
            uploaded_audio = st.file_uploader(
                "Upload reference audio",
                type=[
                    "wav",
                    "mp3",
                    "m4a",
                    "aac",
                    "ogg",
                    "flac",
                    "webm",
                ],
                help=(
                    "Use 10â€“30 seconds of clear, "
                    "continuous speech."
                ),
            )

            if uploaded_audio is not None:
                reference_source = (
                    uploaded_audio.getvalue()
                )
                source_name = uploaded_audio.name

                st.audio(
                    reference_source
                )

        else:
            if hasattr(st, "audio_input"):
                recorded_audio = st.audio_input(
                    "Record through your laptop microphone"
                )

                if recorded_audio is not None:
                    reference_source = (
                        recorded_audio.getvalue()
                    )
                    source_name = (
                        recorded_audio.name
                        or "microphone.wav"
                    )

                    st.audio(
                        reference_source
                    )
            else:
                st.error(
                    "Your Streamlit version does not "
                    "support microphone recording. "
                    "Upgrade Streamlit or upload a WAV."
                )

        st.info(
            "For best cloning: record 10â€“30 seconds, "
            "use one speaker, avoid music, echo, fan "
            "noise and automatic voice effects."
        )

    transcript_container = st.container(
        border=True
    )

    with transcript_container:
        st.markdown(
            "### 3 Â· Exact reference transcript"
        )

        reference_transcript = st.text_area(
            "Write exactly what was spoken",
            height=150,
            placeholder=(
                "Every word, pause-related filler and "
                "name must match the recording."
            ),
        )

        st.caption(
            f"Transcript characters: "
            f"{len(reference_transcript)}"
        )

    target_container = st.container(
        border=True
    )

    with target_container:
        st.markdown(
            "### 4 Â· Text to generate"
        )

        target_text = st.text_area(
            "Long target text",
            height=260,
            placeholder=(
                "Enter the complete text that should "
                "be spoken in the cloned voice."
            ),
        )

        st.caption(
            f"Target characters: "
            f"{len(target_text)}. "
            "Long text will be generated sentence "
            "by sentence and joined automatically."
        )

        english_model = "air"
        exaggeration = 0.50
        cfg_weight = 0.45

        if selected_language == "english":
            english_model = st.selectbox(
                "English model",
                ["air", "nano"],
                index=0,
                format_func=lambda value: (
                    "NeuTTS Air â€” better quality"
                    if value == "air"
                    else "NeuTTS Nano â€” smaller model"
                ),
            )

        if selected_language == "hindi":
            parameter_columns = st.columns(2)

            exaggeration = (
                parameter_columns[0].slider(
                    "Expression",
                    min_value=0.25,
                    max_value=0.80,
                    value=0.50,
                    step=0.05,
                )
            )

            cfg_weight = (
                parameter_columns[1].slider(
                    "Voice guidance",
                    min_value=0.20,
                    max_value=0.70,
                    value=0.45,
                    step=0.05,
                )
            )

    validation_errors = []

    if not consent:
        validation_errors.append(
            "Speaker consent is required."
        )

    if not voice_name.strip():
        validation_errors.append(
            "Enter a voice name."
        )

    if reference_source is None:
        validation_errors.append(
            "Upload or record reference audio."
        )

    if not reference_transcript.strip():
        validation_errors.append(
            "Enter the exact reference transcript."
        )

    if not target_text.strip():
        validation_errors.append(
            "Enter the target text."
        )

    environment = (
        NEUTTS_ENV
        if selected_language == "english"
        else CHATTERBOX_ENV
    )

    python_path = get_python_path(
        environment
    )

    if not python_path.exists():
        validation_errors.append(
            f"Required environment is missing: "
            f"{python_path}"
        )

    if not USER_CLONE_SCRIPT.exists():
        validation_errors.append(
            f"Generation script is missing: "
            f"{USER_CLONE_SCRIPT}"
        )

    if validation_errors:
        with st.expander(
            "Requirements before generation",
            expanded=True,
        ):
            for error in validation_errors:
                st.write(f"â€¢ {error}")

    generate_button = st.button(
        "Generate cloned voice",
        type="primary",
        use_container_width=True,
        disabled=bool(validation_errors),
    )

    if generate_button:
        try:
            saved = save_uploaded_reference(
                audio_bytes=reference_source,
                voice_name=voice_name,
                language=selected_language,
                transcript=reference_transcript,
                source_name=source_name,
            )

            saved["target_path"].write_text(
                target_text.strip(),
                encoding="utf-8",
            )

            arguments = [
                "--language",
                selected_language,
                "--voice-name",
                voice_name,
                "--reference-audio",
                str(saved["reference_path"]),
                "--reference-transcript-file",
                str(saved["transcript_path"]),
                "--target-text-file",
                str(saved["target_path"]),
            ]

            if selected_language == "english":
                arguments.extend(
                    [
                        "--english-model",
                        english_model,
                    ]
                )
            else:
                arguments.extend(
                    [
                        "--exaggeration",
                        str(exaggeration),
                        "--cfg-weight",
                        str(cfg_weight),
                    ]
                )

            process, command = run_script(
                python_path,
                USER_CLONE_SCRIPT,
                arguments,
            )

            st.code(
                subprocess.list2cmdline(
                    command
                ),
                language="powershell",
            )

            log_placeholder = st.empty()
            status_placeholder = st.empty()

            lines = []
            started = time.perf_counter()

            while True:
                line = process.stdout.readline()

                if line:
                    lines.append(
                        line.rstrip()
                    )

                    log_placeholder.code(
                        "\n".join(
                            lines[-100:]
                        ),
                        language=None,
                    )

                return_code = process.poll()

                status_placeholder.info(
                    "Generating cloned voice â€” "
                    f"{time.perf_counter() - started:.1f}s"
                )

                if return_code is not None:
                    remaining = (
                        process.stdout.read()
                    )

                    if remaining:
                        lines.extend(
                            remaining.splitlines()
                        )

                    log_placeholder.code(
                        "\n".join(
                            lines[-150:]
                        ),
                        language=None,
                    )

                    if return_code != 0:
                        status_placeholder.error(
                            "Voice cloning failed. "
                            f"Exit code: {return_code}"
                        )
                    else:
                        status_placeholder.success(
                            "Voice cloning completed."
                        )

                    break

                time.sleep(0.1)

            if process.returncode == 0:
                voice_id = saved["voice_id"]

                cloned_audio, report = (
                    find_latest_user_clone(
                        voice_id,
                        selected_language,
                    )
                )

                if (
                    cloned_audio
                    and cloned_audio.exists()
                ):
                    st.markdown(
                        "### Generated cloned voice"
                    )

                    st.audio(
                        cloned_audio.read_bytes(),
                        format="audio/wav",
                    )

                    st.download_button(
                        "Download cloned WAV",
                        data=cloned_audio.read_bytes(),
                        file_name=(
                            f"{voice_id}_"
                            f"{selected_language}_clone.wav"
                        ),
                        mime="audio/wav",
                        use_container_width=True,
                    )

                    st.success(
                        "Files stored under: "
                        f"{project_relative(cloned_audio.parent)}"
                    )

                    if report:
                        show_report(report)
                else:
                    st.warning(
                        "The process completed, but "
                        "the final WAV was not found."
                    )

        except Exception as exc:
            st.error(
                f"Could not generate clone: {exc}"
            )
            st.exception(exc)

elif page == "Run Experiment":
    st.header("Run a model")

    selected_model = st.selectbox(
        "Select pipeline",
        list(MODEL_CONFIG.keys()),
    )

    config = MODEL_CONFIG[selected_model]

    python_path = get_python_path(config["environment"])
    script_path = config["script"]

    st.write(f"**Python:** `{python_path}`")
    st.write(f"**Script:** `{script_path}`")

    default_text = (
        "Hello, welcome to Infinia. "
        "How can I help you today?"
    )

    input_text = st.text_area(
        "Input text",
        value=default_text,
        height=100,
    )

    can_run = python_path.exists() and script_path.exists()

    if not python_path.exists():
        st.error(
            f"Python environment is missing: {python_path}"
        )

    if not script_path.exists():
        st.error(
            f"Generation script is missing: {script_path}"
        )

    neutts_profile_name = None
    neutts_model_name = None
    neutts_generation_mode = None
    encode_only = False

    if selected_model == "NeuTTS English":
        voice_profiles = load_voice_profiles()

        if not voice_profiles:
            st.error(
                f"No voice profiles found at: {VOICE_PROFILE_DIR}"
            )
            can_run = False
        else:
            neutts_profile_name = st.selectbox(
                "Voice profile",
                list(voice_profiles.keys()),
            )
            neutts_model_name = st.selectbox(
                "Model",
                NEUTTS_MODELS,
            )
            neutts_generation_mode = st.selectbox(
                "Generation mode",
                NEUTTS_GENERATION_MODES,
            )

            selected_profile = voice_profiles[
                neutts_profile_name
            ]
            profile_status = get_profile_status(
                selected_profile
            )

            status_columns = st.columns(4)
            status_columns[0].metric(
                "Reference status",
                (
                    "Encoded"
                    if profile_status["codes_exist"]
                    else "Needs encoding"
                ),
            )
            status_columns[1].metric(
                "Reference profile",
                neutts_profile_name,
            )
            status_columns[2].metric(
                "Model loaded",
                "No",
            )
            status_columns[3].metric(
                "Generation status",
                "Idle",
            )

            if profile_status["audio_exists"]:
                show_audio_file(
                    profile_status["audio_path"],
                    "Reference voice used for cloning",
                )
            else:
                st.error(
                    "Reference audio is missing: "
                    f"{profile_status['audio_path']}"
                )
                can_run = False

            if not profile_status["text_exists"]:
                st.error(
                    "Reference transcript is missing: "
                    f"{profile_status['text_path']}"
                )
                can_run = False

            if st.button(
                "Validate reference",
                disabled=not (
                    profile_status["audio_exists"]
                    and profile_status["text_exists"]
                ),
            ):
                st.success(
                    "Reference audio and transcript are present."
                )
                st.json(selected_profile)

            encode_only = st.checkbox(
                "Encode voice only",
                value=False,
                help=(
                    "Cache reference codes without generating "
                    "target speech."
                ),
            )
    if st.button(
        "Run selected experiment",
        type="primary",
        disabled=not can_run,
    ):
        st.info(
            "The model is running. Live output appears below."
        )

        log_placeholder = st.empty()
        status_placeholder = st.empty()

        arguments = list(config.get("default_arguments", []))

        if selected_model == "MeloTTS English":
            arguments = ["--text", input_text]

        if selected_model == "NeuTTS English":
            arguments = [
                "--text",
                input_text,
                "--profile",
                neutts_profile_name,
                "--model",
                neutts_model_name,
                "--mode",
                neutts_generation_mode,
            ]

            if encode_only:
                arguments.append("--encode-only")

        process, command = run_script(
            python_path,
            script_path,
            arguments,
        )

        st.code(
            subprocess.list2cmdline(command),
            language="powershell",
        )

        lines = []
        start_time = time.perf_counter()

        while True:
            line = process.stdout.readline()

            if line:
                lines.append(line.rstrip())

                log_placeholder.code(
                    "\n".join(lines[-150:]),
                    language=None,
                )

            return_code = process.poll()
            elapsed = time.perf_counter() - start_time

            status_placeholder.write(
                f"Running for {elapsed:.1f} seconds..."
            )

            if return_code is not None:
                remaining = process.stdout.read()

                if remaining:
                    lines.extend(remaining.splitlines())

                log_placeholder.code(
                    "\n".join(lines[-200:]),
                    language=None,
                )

                if return_code == 0:
                    status_placeholder.success(
                        "Experiment completed successfully."
                    )
                else:
                    status_placeholder.error(
                        f"Experiment failed with "
                        f"exit code {return_code}."
                    )

                break

            time.sleep(0.1)

        st.session_state["latest_run_logs"] = lines
        st.session_state["latest_return_code"] = (
            process.returncode
        )

        st.rerun()


elif page == "Generated Audio":
    st.header("Generated audio")

    selected_model = st.selectbox(
        "Select pipeline",
        list(MODEL_CONFIG.keys()),
        key="audio_model",
    )

    config = MODEL_CONFIG[selected_model]
    audio_dir = config["audio_dir"]

    if not audio_dir.exists():
        st.warning(
            f"Audio directory does not exist: {audio_dir}"
        )

    else:
        wav_files = get_wav_files(audio_dir)

        if not wav_files:
            st.info("No generated WAV files found.")

        else:
            st.write(
                f"Found **{len(wav_files)}** audio files."
            )

            for wav_file in wav_files:
                with st.container(border=True):
                    st.subheader(wav_file.name)

                    show_audio_file(wav_file)

                    quality_key = (
                        f"quality_{selected_model}_"
                        f"{wav_file.name}"
                    )

                    st.selectbox(
                        "Voice quality",
                        [
                            "Not evaluated",
                            "Natural",
                            "Slightly robotic",
                            "metallic_level",
                            "Very robotic",
                            "Unclear",
                        ],
                        key=quality_key,
                    )

                    notes_key = (
                        f"notes_{selected_model}_"
                        f"{wav_file.name}"
                    )

                    st.text_area(
                        "Listening notes",
                        key=notes_key,
                        placeholder=(
                            "Example: clear pronunciation, "
                            "but metallic voice and "
                            "unnatural pauses."
                        ),
                    )


elif page == "Voice Comparison":
    st.header("Voice comparison")

    st.write(
        "Compare the original reference voice against "
        "generated outputs from each pipeline."
    )

    available_references = {
        name: path
        for name, path in REFERENCE_AUDIO_OPTIONS.items()
        if path.exists()
    }

    if not available_references:
        st.error(
            "No reference voice was found. Add a WAV file "
            "inside data/reference_audio."
        )

    else:
        selected_reference_name = st.selectbox(
            "Select original reference voice",
            list(available_references.keys()),
        )

        selected_reference_path = (
            available_references[
                selected_reference_name
            ]
        )

        reference_column, info_column = st.columns(
            [2, 1]
        )

        with reference_column:
            with st.container(border=True):
                show_audio_file(
                    selected_reference_path,
                    "Original reference voice",
                )

        with info_column:
            st.markdown("#### Reference information")

            st.write(
                f"**Used for cloning:** "
                f"{selected_reference_name}"
            )

            transcript_path = (
                REFERENCE_TEXT_OPTIONS.get(
                    selected_reference_name
                )
            )

            if (
                transcript_path
                and transcript_path.exists()
            ):
                transcript = load_text(transcript_path)

                st.write("**Transcript:**")
                st.info(
                    transcript
                    or "Transcript is empty."
                )
            else:
                st.warning(
                    "No transcript file was found for "
                    "this reference."
                )

        st.divider()

        st.subheader("Select generated outputs")

        selected_audio = {}

        selection_columns = st.columns(
            len(MODEL_CONFIG)
        )

        for column, model_name in zip(
            selection_columns,
            MODEL_CONFIG.keys(),
        ):
            with column:
                st.markdown(f"### {model_name}")

                selected_path = select_audio_for_model(
                    model_name,
                    key_prefix="comparison",
                )

                selected_audio[model_name] = (
                    selected_path
                )

        st.divider()

        st.subheader("Listen side by side")

        playback_columns = st.columns(
            1 + len(MODEL_CONFIG)
        )

        with playback_columns[0]:
            with st.container(border=True):
                show_audio_file(
                    selected_reference_path,
                    "Original voice",
                )

        for column, (
            model_name,
            audio_path,
        ) in zip(
            playback_columns[1:],
            selected_audio.items(),
        ):
            with column:
                with st.container(border=True):
                    if audio_path:
                        show_audio_file(
                            audio_path,
                            model_name,
                        )
                    else:
                        st.markdown(
                            f"#### {model_name}"
                        )
                        st.warning(
                            "No audio selected."
                        )

        st.divider()

        st.subheader("Manual listening evaluation")

        comparison_scores = {}

        evaluation_columns = st.columns(
            len(MODEL_CONFIG)
        )

        for column, model_name in zip(
            evaluation_columns,
            MODEL_CONFIG.keys(),
        ):
            with column:
                with st.container(border=True):
                    st.markdown(f"### {model_name}")

                    audio_path = selected_audio.get(
                        model_name
                    )

                    if audio_path is None:
                        st.warning(
                            "Generate or select audio first."
                        )
                        continue

                    similarity = st.slider(
                        "Voice similarity",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=(
                            f"similarity_{model_name}_"
                            f"{audio_path.name}"
                        ),
                        help=(
                            "1 means completely different "
                            "from the reference. "
                            "5 means highly similar."
                        ),
                    )

                    naturalness = st.slider(
                        "Naturalness",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=(
                            f"naturalness_{model_name}_"
                            f"{audio_path.name}"
                        ),
                    )

                    clarity = st.slider(
                        "Clarity",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=(
                            f"clarity_{model_name}_"
                            f"{audio_path.name}"
                        ),
                    )

                    pronunciation = st.slider(
                        "Pronunciation",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=(
                            f"pronunciation_{model_name}_"
                            f"{audio_path.name}"
                        ),
                    )

                    metallic_artifacts = st.selectbox(
                        "Metallic or robotic artifacts",
                        [
                            "None",
                            "Mild",
                            "Moderate",
                            "Severe",
                        ],
                        key=(
                            f"metallic_{model_name}_"
                            f"{audio_path.name}"
                        ),
                    )

                    notes = st.text_area(
                        "listening_notes",
                        key=(
                            f"comparison_notes_{model_name}_"
                            f"{audio_path.name}"
                        ),
                        placeholder=(
                            "Example: voice is clear but "
                            "does not resemble the reference."
                        ),
                    )

                    overall_score = (
                        similarity
                        + naturalness
                        + clarity
                        + pronunciation
                    ) / 4

                    st.metric(
                        "Average score",
                        f"{overall_score:.2f}/5",
                    )

                    comparison_scores[model_name] = {
                        "audio_path": str(audio_path),
                        "voice_similarity": similarity,
                        "naturalness": naturalness,
                        "clarity": clarity,
                        "pronunciation": pronunciation,
                        "metallic_artifacts": (
                            metallic_artifacts
                        ),
                        "average_score": (
                            overall_score
                        ),
                        "notes": notes,
                    }

        st.divider()

        valid_models = [
            model_name
            for model_name, audio_path
            in selected_audio.items()
            if audio_path is not None
        ]

        preferred_model = st.selectbox(
            "Which output sounds best?",
            ["Not decided", *valid_models],
        )

        overall_notes = st.text_area(
            "Overall comparison notes",
            placeholder=(
                "Example: NeuTTS is closest to the "
                "original voice, while MeloTTS sounds "
                "metallic and MMS sounds robotic."
            ),
        )

        if st.button(
            "Save comparison report",
            type="primary",
            disabled=not bool(comparison_scores),
        ):
            comparison_report = {
                "created_at": datetime.now().isoformat(),
                "reference_name": (
                    selected_reference_name
                ),
                "reference_audio_path": str(
                    selected_reference_path
                ),
                "preferred_model": preferred_model,
                "overall_notes": overall_notes,
                "model_scores": comparison_scores,
            }

            saved_path = save_comparison_report(
                comparison_report
            )

            st.success(
                "Comparison report saved successfully."
            )

            st.code(
                str(saved_path),
                language=None,
            )

            st.json(comparison_report)



elif page == "Arabic Results":
    render_arabic_results()

elif page == "Review Workspace":
    render_review_workspace()


elif page == "Reports":
    st.header("Reports and metrics")

    result_csv = RESULTS_DIR / "raw_runs.csv"

    if result_csv.exists():
        try:
            dataframe = pd.read_csv(result_csv)

            st.subheader("Raw experiment runs")

            st.dataframe(
                dataframe,
                use_container_width=True,
            )

        except Exception as exc:
            st.error(
                f"Could not read raw_runs.csv: {exc}"
            )

    else:
        st.info(
            f"No raw metrics file found at: "
            f"{result_csv}"
        )

    st.divider()
    st.subheader("JSON result files")

    snapshot_dir = EVIDENCE_DIR / "result_snapshots"

    if snapshot_dir.exists():
        json_files = sorted(
            snapshot_dir.rglob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for json_file in json_files:
            with st.expander(
                str(
                    json_file.relative_to(
                        snapshot_dir
                    )
                )
            ):
                report = load_json(json_file)

                if report is None:
                    st.error(
                        "Could not read this JSON file."
                    )
                else:
                    st.json(report)

                st.code(
                    str(json_file),
                    language=None,
                )

    else:
        st.info(
            "No result snapshot directory found."
        )

    st.divider()
    st.subheader("Terminal logs")

    logs_dir = EVIDENCE_DIR / "terminal_logs"

    if logs_dir.exists():
        log_files = sorted(
            logs_dir.glob("*.txt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for log_file in log_files:
            with st.expander(log_file.name):
                try:
                    content = log_file.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )

                    st.code(
                        content[-20000:],
                        language=None,
                    )

                except Exception as exc:
                    st.error(
                        f"Could not read log: {exc}"
                    )

    else:
        st.info(
            "No terminal logs directory found."
        )









