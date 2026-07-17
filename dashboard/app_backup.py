import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
RESULTS_DIR = PROJECT_ROOT / "results"
VOICE_PROFILE_DIR = PROJECT_ROOT / "data" / "voice_profiles"

MMS_ENV = PROJECT_ROOT / ".venv"
MELOTTS_ENV = PROJECT_ROOT / ".venv-melotts"
NEUTTS_ENV = PROJECT_ROOT / ".venv-neutts"

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

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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


st.set_page_config(
    page_title="TTS Experiment Dashboard",
    page_icon="ðŸŽ™ï¸",
    layout="wide",
)

st.title("ðŸŽ™ï¸ TTS Experiment Dashboard")

st.caption(
    "Monitor, run, listen to, and compare MMS, "
    "MeloTTS, and NeuTTS experiments."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Run Experiment",
        "Generated Audio",
        "Voice Comparison",
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

        arguments = []

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
                            "Metallic",
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
                        "Notes",
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



