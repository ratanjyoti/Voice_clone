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

# ==============================================================================
# PROJECT PATHS & CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_NEUTTS_DIR = PROJECT_ROOT / "external" / "neutts"

if EXTERNAL_NEUTTS_DIR.exists():
    external_neutts_text = str(EXTERNAL_NEUTTS_DIR)
    if external_neutts_text not in sys.path:
        sys.path.insert(0, external_neutts_text)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_DIR / "raw"
RESULTS_SUMMARY_DIR = RESULTS_DIR / "summary"
VOICE_PROFILE_DIR = PROJECT_ROOT / "data" / "voice_profiles"
HUMAN_EVALUATION_CSV = RESULTS_RAW_DIR / "human_voice_evaluations.csv"
AUTOMATIC_EVALUATION_CSV = RESULTS_RAW_DIR / "automatic_tts_evaluation.csv"
USER_VOICE_DATA_DIR = PROJECT_ROOT / "data" / "user_voice_clones"
USER_VOICE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "user_voice_clones"
USER_CLONE_SCRIPT = PROJECT_ROOT / "src" / "pipelines" / "english" / "generate_user_voice_clone.py"

# Environment Paths
VENVS_DIR = PROJECT_ROOT / ".venvs"
CHATTERBOX_ENV = VENVS_DIR / ".venv"
MMS_ENV = VENVS_DIR / ".venv"
MELOTTS_ENV = VENVS_DIR / ".venv-melotts"
NEUTTS_ENV = VENVS_DIR / ".venv-neutts"
XTTS_ARABIC_ENV = VENVS_DIR / ".venv-xtts-arabic"

REFERENCE_AUDIO_OPTIONS = {
    "Ratan Neutral": PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_neutral.wav",
    "Ratan Conversational": PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_conversational.wav",
    "Ratan Expressive": PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_expressive.wav",
    "Hindi cloning reference": PROJECT_ROOT / "data" / "reference_audio" / "ratan_reference_22050_mono.wav",
    "Arabic MSA short": PROJECT_ROOT / "data" / "reference_audio" / "arabic" / "professional_msa" / "arabic_reference_short.wav",
    "Arabic MSA standard": PROJECT_ROOT / "data" / "reference_audio" / "arabic" / "professional_msa" / "arabic_reference_standard.wav",
    "Arabic MSA long": PROJECT_ROOT / "data" / "reference_audio" / "arabic" / "professional_msa" / "arabic_reference_long.wav",
}

REFERENCE_TEXT_OPTIONS = {
    "Ratan Neutral": PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_neutral.txt",
    "Ratan Conversational": PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_conversational.txt",
    "Ratan Expressive": PROJECT_ROOT / "data" / "reference_audio" / "english" / "ratan_expressive.txt",
}

NEUTTS_MODELS = ["Nano Q8", "Nano Q4", "Air Q8", "Air Q4"]
NEUTTS_GENERATION_MODES = ["Standard", "Sentence-by-sentence", "Streaming"]

MODEL_CONFIG = {
    "MMS English": {
        "environment": MMS_ENV,
        "script": PROJECT_ROOT / "src" / "pipelines" / "english" / "mms.py",
        "report": EVIDENCE_DIR / "result_snapshots" / "mms_english_smoke.json",
        "audio_dir": OUTPUTS_DIR / "english" / "mms",
        "supports_voice_cloning": False,
    },
    "MeloTTS English": {
        "environment": MELOTTS_ENV,
        "script": PROJECT_ROOT / "src" / "pipelines" / "english" / "melotts.py",
        "report": EVIDENCE_DIR / "result_snapshots" / "melotts_english_smoke.json",
        "audio_dir": OUTPUTS_DIR / "english" / "melotts",
        "supports_voice_cloning": False,
    },
    "NeuTTS English": {
        "environment": NEUTTS_ENV,
        "script": PROJECT_ROOT / "src" / "pipelines" / "english" / "neutts.py",
        "report": EVIDENCE_DIR / "result_snapshots" / "neutts_english_smoke.json",
        "audio_dir": OUTPUTS_DIR / "neutts" / "english",
        "supports_voice_cloning": True,
    },
    "Chatterbox Arabic": {
        "environment": CHATTERBOX_ENV,
        "script": PROJECT_ROOT / "src" / "pipelines" / "arabic" / "chatterbox.py",
        "report": EVIDENCE_DIR / "result_snapshots" / "chatterbox" / "arabic" / "chatterbox_arabic_summary.json",
        "audio_dir": OUTPUTS_DIR / "chatterbox" / "arabic",
        "supports_voice_cloning": True,
        "default_arguments": ["--mode", "final", "--device", "cpu", "--reference", "standard", "--seed", "42", "--temperature", "0.6", "--cfg-weight", "0.5", "--exaggeration", "0.5", "--repetition-penalty", "2.0", "--min-p", "0.05", "--top-p", "1.0"],
    },
    "XTTS-v2 Arabic": {
        "environment": XTTS_ARABIC_ENV,
        "script": PROJECT_ROOT / "src" / "pipelines" / "arabic" / "xtts.py",
        "report": EVIDENCE_DIR / "result_snapshots" / "xtts" / "arabic" / "xtts_arabic_summary.json",
        "audio_dir": OUTPUTS_DIR / "arabic" / "xtts",
        "supports_voice_cloning": True,
        "default_arguments": ["--mode", "final", "--device", "cpu", "--reference-strategy", "multi", "--seed", "42", "--configuration-id", "C", "--temperature", "0.65", "--top-k", "40", "--top-p", "0.80", "--repetition-penalty", "5.0", "--length-penalty", "1.0", "--speed", "1.0"],
    },
    "MMS Hindi": {
        "environment": MMS_ENV,
        "script": PROJECT_ROOT / "src" / "pipelines" / "hindi" / "mms.py",
        "report": None,
        "audio_dir": OUTPUTS_DIR / "hindi" / "mms",
        "supports_voice_cloning": False,
    },
    "Chatterbox Hindi": {
        "environment": CHATTERBOX_ENV,
        "script": PROJECT_ROOT / "src" / "pipelines" / "hindi" / "chatterbox.py",
        "report": OUTPUTS_DIR / "chatterbox" / "hindi" / "final" / "generation_summary.json",
        "audio_dir": OUTPUTS_DIR / "chatterbox" / "hindi",
        "supports_voice_cloning": True,
    },
    "IndicF5 Hindi": {
        "environment": PROJECT_ROOT / ".venvs" / ".venv-indicf5",
        "script": PROJECT_ROOT / "src" / "pipelines" / "hindi" / "indicf5.py",
        "report": EVIDENCE_DIR / "result_snapshots" / "indicf5_hindi_initial_generation_success.json",
        "audio_dir": OUTPUTS_DIR / "hindi" / "indicf5",
        "supports_voice_cloning": True,
    },
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def get_python_path(environment_path: Path) -> Path:
    return environment_path / "Scripts" / "python.exe"

def load_json(path: Path) -> dict | None:
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None

def load_text(path: Path) -> str | None:
    if not path.exists(): return None
    try: return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception: return None

def safe_voice_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:60] or "voice"

def project_relative(path: Path | str | None) -> str:
    if path is None: return ""
    path_obj = Path(path)
    try: return path_obj.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception: return str(path_obj)

def load_csv_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path, keep_default_na=False, encoding="utf-8")
    except Exception: return pd.DataFrame()

def result_csv_path(filename: str, summary: bool = False) -> Path:
    preferred = RESULTS_SUMMARY_DIR if summary else RESULTS_RAW_DIR
    candidates = [preferred / filename, RESULTS_RAW_DIR / filename, RESULTS_SUMMARY_DIR / filename, RESULTS_DIR / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return preferred / filename

# ==============================================================================
# RECRUITER SUMMARY HELPERS
# ==============================================================================
def value_or_pending(value: object, suffix: str = "") -> str:
    if value is None: return "Pending"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}: return "Pending"
    return f"{text}{suffix}"

def safe_round(value: object, digits: int = 2) -> str:
    try: return str(round(float(value), digits))
    except Exception: return "Pending"

def percent_from_value(value: object) -> str:
    try:
        number = float(value)
        if number <= 1: number *= 100
        return f"{number:.2f}%"
    except Exception: return "Pending"

def get_first_row(dataframe: pd.DataFrame) -> dict:
    if dataframe.empty: return {}
    return dataframe.iloc[0].to_dict()

def read_melotts_summary() -> dict:
    candidates = [result_csv_path("melotts_english_summary.csv", True), result_csv_path("melotts_english_summary.csv")]
    for path in candidates:
        df = load_csv_dataframe(path)
        if not df.empty: return get_first_row(df)
    return {}

def read_hindi_chatterbox_summary() -> dict:
    df = load_csv_dataframe(result_csv_path("chatterbox_hindi_evaluation.csv"))
    if df.empty: return {}
    wer_col = next((c for c in df.columns if "wer_percent" in c.lower()), None)
    rtf_col = next((c for c in df.columns if "rtf" in c.lower()), None)
    res = {"samples": len(df)}
    if wer_col: res["average_wer_percent"] = pd.to_numeric(df[wer_col], errors="coerce").mean()
    if rtf_col: res["average_rtf"] = pd.to_numeric(df[rtf_col], errors="coerce").mean()
    return res

def read_arabic_xtts_summary() -> dict:
    df = load_csv_dataframe(result_csv_path("arabic_model_comparison.csv", summary=True))
    if df.empty: return {}
    if "model" in df.columns:
        xtts_rows = df[df["model"].astype(str).str.contains("XTTS", case=False, na=False)]
        if not xtts_rows.empty: df = xtts_rows
    summary = {"samples": len(df)}
    if "WER_percent" in df.columns: summary["average_wer_percent"] = pd.to_numeric(df["WER_percent"], errors="coerce").mean()
    elif "average_wer" in df.columns: summary["average_wer"] = pd.to_numeric(df["average_wer"], errors="coerce").mean()
    if "RTF" in df.columns: summary["average_rtf"] = pd.to_numeric(df["RTF"], errors="coerce").mean()
    elif "average_rtf" in df.columns: summary["average_rtf"] = pd.to_numeric(df["average_rtf"], errors="coerce").mean()
    return summary

def get_final_decision_rows() -> list[dict]:
    melotts = read_melotts_summary()
    arabic_xtts = read_arabic_xtts_summary()
    hindi_chatterbox = read_hindi_chatterbox_summary()

    m_wer = melotts.get("average_WER_percent") or melotts.get("average_wer_percent") or melotts.get("WER_percent")
    m_rtf = melotts.get("average_RTF") or melotts.get("average_rtf") or melotts.get("RTF")
    a_wer = arabic_xtts.get("average_wer") or arabic_xtts.get("average_WER") or arabic_xtts.get("wer") or arabic_xtts.get("average_wer_percent")
    a_rtf = arabic_xtts.get("average_rtf") or arabic_xtts.get("average_RTF") or arabic_xtts.get("RTF") or arabic_xtts.get("average_rtf")
    h_wer = hindi_chatterbox.get("average_wer_percent")
    h_rtf = hindi_chatterbox.get("average_rtf")

    return [
        {"Language": "English", "Final model / voice": "MeloTTS — EN-US speaker", "WER": percent_from_value(m_wer), "RTF": safe_round(m_rtf, 3), "Status": "Final English winner", "Reason": "Best balance of naturalness and speed."},
        {"Language": "Arabic", "Final model / voice": "XTTS-v2 — Arabic MSA", "WER": percent_from_value(a_wer), "RTF": safe_round(a_rtf, 3), "Status": "Best intelligibility", "Reason": "Passed WER targets; CPU latency higher than ideal."},
        {"Language": "Hindi", "Final model / voice": "Chatterbox (Best tested)", "WER": percent_from_value(h_wer), "RTF": safe_round(h_rtf, 3), "Status": "Unresolved", "Reason": "MMS robotic; Chatterbox slow; IndicF5 prompt leakage detected."}
    ]

# ==============================================================================
# CORE RENDER FUNCTIONS (Recruiter Facing)
# ==============================================================================
def render_recruiter_overview() -> None:
    st.markdown("""<div class="review-hero"><div><div class="eyebrow">FINAL SUBMISSION DASHBOARD</div><h1>Open-Source Multilingual TTS Benchmark</h1><p>Recruiter-facing summary of the final model choices, metrics, audio clips, evidence logs, and failure analysis for English, Arabic, and Hindi.</p></div><div class="hero-badge">Runs • Numbers • Clips • Messy Parts</div></div>""", unsafe_allow_html=True)
    final_df = pd.DataFrame(get_final_decision_rows())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Languages tested", "3")
    metric_cols[1].metric("Final architecture", "Language router")
    metric_cols[2].metric("Audio evidence", "WAV + JSON")
    metric_cols[3].metric("Evaluation", "WER + RTF + MOS")
    st.subheader("Final model choices")
    st.dataframe(final_df, use_container_width=True, hide_index=True)
    
    tabs = st.tabs(["Final recommendation", "Metric targets", "Failure summary", "Submission checklist"])
    with tabs[0]:
        st.markdown("### Final architecture\n```text\nInput text → Language router → (English: MeloTTS / Arabic: XTTS-v2 / Hindi: MMS/Chatterbox) → Audio validation → ASR evaluation → Manual review\n```")
    with tabs[1]:
        targets = pd.DataFrame([{"Metric": "WER", "Target": "Eng ≤10%, Ar/Hi ≤15%", "Meaning": "Intelligibility"}, {"Metric": "RTF", "Target": "≤0.5", "Meaning": "Speed"}, {"Metric": "MOS", "Target": "≥4.0", "Meaning": "Naturalness"}])
        st.dataframe(targets, use_container_width=True, hide_index=True)
    with tabs[2]:
        failures = pd.DataFrame([{"Area": "Dependency hell", "What happened": "Python 3.12 conflicts with Coqui TTS"}, {"Area": "IndicF5", "What happened": "Prompt leakage and meta-device errors"}, {"Area": "Hindi Quality", "What happened": "No production-ready open-source model found"}])
        st.dataframe(failures, use_container_width=True, hide_index=True)
    with tabs[3]:
        checklist = [("README", PROJECT_ROOT / "README.md"), ("Engineering Journey", PROJECT_ROOT / "docs" / "engineering_journey.md"), ("Results", RESULTS_DIR)]
        for label, path in checklist:
            st.success(f"{label}: available") if path.exists() else st.error(f"{label}: missing")

def render_final_results_page() -> None:
    st.header("Final Results")
    final_df = pd.DataFrame(get_final_decision_rows())
    st.dataframe(final_df, use_container_width=True, hide_index=True)
    tabs = st.tabs(["English", "Arabic", "Hindi", "All Metrics"])
    with tabs[0]:
        st.subheader("English: MeloTTS")
        st.markdown("MeloTTS selected as winner for naturalness vs speed tradeoff.")
    with tabs[1]:
        st.subheader("Arabic: XTTS-v2")
        st.markdown("XTTS-v2 winner for intelligibility, though CPU latency is a bottleneck.")
    with tabs[2]:
        st.subheader("Hindi: Unresolved")
        st.markdown("Documented failures for IndicF5 and Chatterbox. MMS used as baseline.")
    with tabs[3]:
        all_csvs = sorted(RESULTS_DIR.rglob("*.csv"))
        selected_csv = st.selectbox("Select result CSV", all_csvs, format_func=lambda p: project_relative(p))
        if selected_csv: st.dataframe(load_csv_dataframe(selected_csv))

def render_recruiter_audio_clips() -> None:
    st.header("Final Audio Clips")

    st.info(
        "These clips include the final winners, speed baselines, and rejected-but-important "
        "naturalness/failure cases. NeuTTS is shown as an English naturalness candidate, "
        "not as the final winner."
    )

    clip_groups = {
        "English — MeloTTS Winner": [
            OUTPUTS_DIR / "english" / "melotts",
            OUTPUTS_DIR / "melotts" / "english",
        ],
        "English — NeuTTS Naturalness Candidate": [
            OUTPUTS_DIR / "english" / "neutts",
            OUTPUTS_DIR / "neutts" / "english",
        ],
        "English — MMS Speed Baseline": [
            OUTPUTS_DIR / "english" / "mms",
            OUTPUTS_DIR / "mms" / "english",
        ],
        "Arabic — XTTS-v2 Quality Candidate": [
            OUTPUTS_DIR / "arabic" / "xtts",
            OUTPUTS_DIR / "xtts" / "arabic",
        ],
        "Arabic — MMS Speed Baseline": [
            OUTPUTS_DIR / "arabic" / "mms",
            OUTPUTS_DIR / "mms" / "arabic",
        ],
        "Arabic — Chatterbox Rejected Candidate": [
            OUTPUTS_DIR / "arabic" / "chatterbox",
            OUTPUTS_DIR / "chatterbox" / "arabic",
        ],
        "Hindi — MMS Stable Baseline": [
            OUTPUTS_DIR / "hindi" / "mms",
            OUTPUTS_DIR / "mms" / "hindi",
        ],
        "Hindi — Chatterbox Best Tested Candidate": [
            OUTPUTS_DIR / "hindi" / "chatterbox",
            OUTPUTS_DIR / "chatterbox" / "hindi",
        ],
    }

    for group_name, directories in clip_groups.items():
        with st.container(border=True):
            st.subheader(group_name)

            wav_files = []
            for directory in directories:
                if directory.exists():
                    wav_files.extend(directory.rglob("*.wav"))

            unique_files = sorted(
                {str(path.resolve()): path for path in wav_files}.values(),
                key=lambda p: p.name,
            )

            if not unique_files:
                st.warning("No clips found.")
                st.caption(
                    "Checked: "
                    + ", ".join(project_relative(directory) for directory in directories)
                )
                continue

            selected = st.selectbox(
                "Select clip",
                unique_files,
                format_func=lambda p: project_relative(p),
                key=f"clip_{group_name}",
            )

            show_audio_file(selected)

            expected = find_expected_text(selected)
            if expected:
                st.info(expected)

def render_evidence_and_logs_page() -> None:
    st.header("Evidence & Logs")
    tabs = st.tabs(["Result CSVs", "Terminal logs", "Snapshots", "Docs"])
    with tabs[0]:
        csv_files = sorted(RESULTS_DIR.rglob("*.csv")) if RESULTS_DIR.exists() else []
        selected = st.selectbox("Select CSV", csv_files, format_func=lambda p: project_relative(p)) if csv_files else None
        if selected: st.dataframe(load_csv_dataframe(selected))
    with tabs[1]:
        log_files = sorted((EVIDENCE_DIR / "terminal_logs").glob("*.txt")) if (EVIDENCE_DIR / "terminal_logs").exists() else []
        selected = st.selectbox("Select Log", log_files, format_func=lambda p: p.name) if log_files else None
        if selected: st.code(load_text(selected) or "")
    with tabs[2]:
        snapshots = sorted((EVIDENCE_DIR / "result_snapshots").rglob("*.json")) if (EVIDENCE_DIR / "result_snapshots").exists() else []
        selected = st.selectbox("Select JSON", snapshots, format_func=lambda p: p.name) if snapshots else None
        if selected: st.json(load_json(selected))
    with tabs[3]:
        docs = sorted((PROJECT_ROOT / "docs").glob("*.md")) if (PROJECT_ROOT / "docs").exists() else []
        selected = st.selectbox("Select Doc", docs, format_func=lambda p: p.name) if docs else None
        if selected: st.markdown(load_text(selected) or "")

# ==============================================================================
# EXISTING FUNCTIONALITY (MODIFIED FOR NAVIGATION)
# ==============================================================================
def get_model_status(model_name: str) -> dict:
    config = MODEL_CONFIG[model_name]
    env = config["environment"]
    py_path = get_python_path(env)
    script_path = config["script"]
    audio_dir = config["audio_dir"]
    report_path, report = find_latest_json(model_name)
    wav_files = get_wav_files(audio_dir)
    status = "Not started"
    if report: status = "Report found"
    elif wav_files: status = "Audio generated"
    elif script_path.exists() and py_path.exists(): status = "Ready to run"
    elif not py_path.exists(): status = "Environment missing"
    else: status = "Script missing"
    return {"name": model_name, "status": status, "environment_exists": env.exists(), "python_exists": py_path.exists(), "script_exists": script_path.exists(), "script_path": script_path, "python_path": py_path, "report_path": report_path, "report": report, "wav_files": wav_files}

def status_icon(status: str) -> str:
    s = status.lower()
    if s == "working": return "✅"
    if s in {"ready to run", "audio generated"}: return "⏳"
    if s == "failed": return "❌"
    return "⚙️"

def find_latest_json(model_name: str) -> tuple[Path | None, object | None]:
    config = MODEL_CONFIG[model_name]
    report_path = config.get("report")
    if report_path and report_path.exists(): return report_path, load_json(report_path)
    snapshot_dir = EVIDENCE_DIR / "result_snapshots"
    if not snapshot_dir.exists(): return None, None
    keywords = [part.lower() for part in re.split(r"[^A-Za-z0-9]+", model_name) if part and part.lower() not in {"v2"}]
    matching = [p for p in snapshot_dir.rglob("*.json") if all(k in p.name.lower() for k in keywords[:2])]
    if not matching: return None, None
    latest = max(matching, key=lambda p: p.stat().st_mtime)
    return latest, load_json(latest)

def get_wav_files(audio_dir: Path) -> list[Path]:
    dirs = [audio_dir]
    reverse_map = {
        ("english", "mms"): OUTPUTS_DIR / "mms" / "english",
        ("english", "melotts"): OUTPUTS_DIR / "melotts" / "english",
        ("arabic", "xtts"): OUTPUTS_DIR / "xtts" / "arabic",
        ("arabic", "mms"): OUTPUTS_DIR / "mms" / "arabic",
        ("hindi", "mms"): OUTPUTS_DIR / "mms" / "hindi",
        ("hindi", "indicf5"): OUTPUTS_DIR / "indicf5" / "hindi",
    }
    try:
        rel = audio_dir.relative_to(OUTPUTS_DIR).parts
        if len(rel) >= 2 and (rel[0], rel[1]) in reverse_map:
            dirs.append(reverse_map[(rel[0], rel[1])])
    except Exception:
        pass
    files = []
    for directory in dirs:
        if directory.exists(): files.extend(directory.rglob("*.wav"))
    unique = {str(path.resolve()): path for path in files}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)

def show_report(report: dict):
    col1, col2, col3, col4 = st.columns(4)
    load_t = report.get("model_load_time_seconds", report.get("load_time_seconds", "N/A"))
    synth_t = report.get("synthesis_time_seconds", report.get("generation_time_seconds", "N/A"))
    dur = report.get("duration_seconds", report.get("audio_duration_seconds", "N/A"))
    rtf = report.get("rtf", "N/A")
    col1.metric("Load", f"{load_t}s"); col2.metric("Synth", f"{synth_t}s"); col3.metric("Dur", f"{dur}s"); col4.metric("RTF", f"{rtf}")

def resolve_existing_audio_path(audio_path: Path | str) -> Path:
    path = Path(audio_path)
    if path.exists():
        return path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    legacy = {
        ("mms", "english"): ("english", "mms"), ("mms", "arabic"): ("arabic", "mms"),
        ("mms", "hindi"): ("hindi", "mms"), ("melotts", "english"): ("english", "melotts"),
        ("xtts", "arabic"): ("arabic", "xtts"), ("indicf5", "hindi"): ("hindi", "indicf5"),
    }
    try:
        rel = path.relative_to(PROJECT_ROOT)
        parts = rel.parts
        if len(parts) >= 4 and parts[0] == "outputs" and (parts[1], parts[2]) in legacy:
            candidate = OUTPUTS_DIR.joinpath(*legacy[(parts[1], parts[2])], *parts[3:])
            if candidate.exists():
                return candidate
    except Exception:
        pass
    matches = list(OUTPUTS_DIR.rglob(path.name)) if OUTPUTS_DIR.exists() else []
    return matches[0] if matches else path

def show_audio_file(audio_path: Path | str, title: str | None = None):
    audio_path = resolve_existing_audio_path(audio_path)
    if title:
        st.markdown(f"#### {title}")
    if not audio_path.exists():
        st.error(f"File missing: {audio_path}")
        return
    try:
        st.audio(str(audio_path), format="audio/wav")
    except Exception:
        st.audio(audio_path.read_bytes(), format="audio/wav")
    st.caption(project_relative(audio_path))

def run_script(python_path: Path, script_path: Path, arguments: list[str]):
    command = [str(python_path), "-u", str(script_path), *arguments]
    env = os.environ.copy()
    env["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    env["ESPEAK_DATA_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng-data"
    env.setdefault("HF_HUB_OFFLINE", "1"); env.setdefault("TRANSFORMERS_OFFLINE", "1")
    process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    return process, command

# [LIBRARIES AND HELPER FUNCTIONS like load_evaluations, save_evaluations, render_review_workspace, etc. remain exactly as in your original code]
# (I have omitted them here for brevity, but KEEP them in your file)

def load_evaluations() -> pd.DataFrame:
    if not HUMAN_EVALUATION_CSV.exists(): return pd.DataFrame(columns=EVALUATION_COLUMNS)
    try: return pd.read_csv(HUMAN_EVALUATION_CSV, keep_default_na=False)
    except Exception: return pd.DataFrame(columns=EVALUATION_COLUMNS)

def save_evaluations(dataframe: pd.DataFrame) -> None:
    HUMAN_EVALUATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = HUMAN_EVALUATION_CSV.with_suffix(".tmp.csv")
    dataframe.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(HUMAN_EVALUATION_CSV)

def evaluation_id_for(generated_audio: Path) -> str:
    return hashlib.sha1(project_relative(generated_audio).encode("utf-8")).hexdigest()[:16]

def discover_generated_audio() -> list[dict]:
    if not OUTPUTS_DIR.exists(): return []
    rows = []
    languages = {"english", "arabic", "hindi"}
    for path in sorted(OUTPUTS_DIR.rglob("*.wav"), key=lambda i: i.stat().st_mtime, reverse=True):
        parts = list(path.relative_to(OUTPUTS_DIR).parts)
        if parts and parts[0].lower() in languages:
            language = parts[0]; model = parts[1] if len(parts) > 1 else "unknown"; profile = parts[2] if len(parts) > 2 else "general"
        else:
            model = parts[0] if parts else "unknown"; language = parts[1] if len(parts) > 1 else "unknown"; profile = parts[2] if len(parts) > 2 else "general"
        rows.append({"path": path, "relative_path": project_relative(path), "sample_id": path.stem, "model": model.replace("_", " ").title(), "language": language.replace("_", " ").title(), "profile": profile.replace("_", " ").title(), "modified": datetime.fromtimestamp(path.stat().st_mtime)})
    return rows

def discover_reference_audio() -> dict[str, Path]:
    root = PROJECT_ROOT / "data" / "reference_audio"
    if not root.exists(): return {}
    return {project_relative(p): p for p in sorted(root.rglob("*.wav"))}

def infer_best_reference(generated: dict, references: dict[str, Path]) -> str | None:
    if not references: return None
    gen_tokens = {t.lower() for t in (generated.get("language", "") + " " + generated.get("profile", "") + " " + generated.get("sample_id", "")).replace("_", " ").split() if t}
    best_label, best_score = None, -1
    for label, path in references.items():
        can_tokens = {t.lower() for t in (label.replace("/", " ") + " " + path.stem.replace("_", " ")).split() if t}
        score = len(gen_tokens.intersection(can_tokens))
        if "hindi" in gen_tokens and "hindi" in can_tokens: score += 5
        if score > best_score: best_label, best_score = label, score
    return best_label

def find_expected_text(generated_audio: Path) -> str:
    for candidate in [generated_audio.with_suffix(".txt"), generated_audio.parent / f"{generated_audio.stem}.json"]:
        if candidate.exists():
            if candidate.suffix.lower() == ".txt": return load_text(candidate) or ""
            report = load_json(candidate)
            if report:
                for k in ["text", "expected_text", "input_text", "sentence"]:
                    if isinstance(report.get(k), str): return report[k].strip()
    return ""

def upsert_evaluation(record: dict) -> tuple[str, pd.DataFrame]:
    df = load_evaluations()
    eid = record["evaluation_id"]
    now = datetime.now().isoformat(timespec="seconds")
    matching = df.index[df["evaluation_id"].astype(str) == eid].tolist()
    if matching:
        idx = matching[0]
        record["created_at"] = df.at[idx, "created_at"] or now
        record["updated_at"] = now
        for k, v in record.items(): df.at[idx, k] = v
        action = "updated"
    else:
        record["created_at"] = now; record["updated_at"] = now
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        action = "saved"
    save_evaluations(df)
    return action, df

# [Include all other render functions: render_review_workspace, render_arabic_results, etc. exactly as in your original code]
def render_review_workspace():
    # ... (Existing code from your original post) ...
    pass

def render_arabic_results():
    # ... (Existing code from your original post) ...
    pass



def render_generated_audio_page() -> None:
    st.header("Generated Audio")
    rows = discover_generated_audio()
    if not rows:
        st.warning("No generated WAV files found under outputs.")
        return

    df = pd.DataFrame(rows)
    languages = ["All"] + sorted(df["language"].unique().tolist())
    models = ["All"] + sorted(df["model"].unique().tolist())
    c1, c2 = st.columns(2)
    language = c1.selectbox("Language", languages)
    model = c2.selectbox("Model", models)

    filtered = df.copy()
    if language != "All":
        filtered = filtered[filtered["language"] == language]
    if model != "All":
        filtered = filtered[filtered["model"] == model]

    st.dataframe(filtered.drop(columns=["path"]), use_container_width=True, hide_index=True)
    if filtered.empty:
        st.info("No audio matches the selected filters.")
        return

    st.subheader("Audio Players")
    for idx, row in filtered.reset_index(drop=True).iterrows():
        audio_path = Path(row["path"])
        label = f'{idx + 1}. {row.get("language", "unknown")} / {row.get("model", "unknown")} / {audio_path.name}'
        with st.expander(label, expanded=idx < 3):
            st.caption(project_relative(audio_path))
            show_audio_file(audio_path)
            expected = find_expected_text(audio_path)
            if expected:
                st.info(expected)


def render_review_workspace() -> None:
    st.header("Review Workspace")
    rows = discover_generated_audio()
    refs = discover_reference_audio()
    if not rows:
        st.warning("No generated WAV files found.")
        return
    audio_paths = [row["path"] for row in rows]
    selected_audio = st.selectbox("Generated sample", audio_paths, format_func=lambda path: project_relative(path))
    row = next(item for item in rows if item["path"] == selected_audio)
    show_audio_file(selected_audio)
    expected_text = find_expected_text(selected_audio)
    st.text_area("Expected text", value=expected_text, height=90)
    ref_label = infer_best_reference(row, refs) or (next(iter(refs)) if refs else "")
    reference_name = st.selectbox("Reference", list(refs.keys()), index=list(refs.keys()).index(ref_label) if ref_label in refs else 0) if refs else ""
    if reference_name: show_audio_file(refs[reference_name], "Reference")
    c1, c2, c3 = st.columns(3)
    similarity = c1.slider("Similarity", 0, 5, 0)
    naturalness = c2.slider("Naturalness", 0, 5, 0)
    pronunciation = c3.slider("Pronunciation", 0, 5, 0)
    metallic = st.selectbox("Metallic level", ["None", "Mild", "Strong"])
    missing_words = st.selectbox("Missing words", ["No", "Yes"])
    repeated_words = st.selectbox("Repeated words", ["No", "Yes"])
    accepted = st.selectbox("Accepted", ["No", "Yes"])
    reviewer = st.text_input("Reviewer", value="manual")
    notes = st.text_area("Listening notes")
    if st.button("Save review", type="primary"):
        record = {"evaluation_id": evaluation_id_for(selected_audio), "sample_id": row["sample_id"], "language": row["language"], "profile": row["profile"], "model": row["model"], "reference_name": reference_name, "reference_audio": project_relative(refs.get(reference_name)) if reference_name else "", "generated_audio": project_relative(selected_audio), "expected_text": expected_text, "similarity_score": similarity, "naturalness_score": naturalness, "pronunciation_score": pronunciation, "metallic_level": metallic, "missing_words": missing_words, "repeated_words": repeated_words, "listening_notes": notes, "accepted": accepted, "reviewer": reviewer}
        action, _ = upsert_evaluation(record)
        st.success(f"Review {action}: {project_relative(HUMAN_EVALUATION_CSV)}")
    existing = load_evaluations()
    if not existing.empty:
        st.subheader("Saved reviews")
        st.dataframe(existing, use_container_width=True)


def render_arabic_results() -> None:
    st.header("Arabic Results")
    files = [result_csv_path("arabic_model_comparison.csv", summary=True), result_csv_path("xtts_arabic_evaluation.csv"), result_csv_path("chatterbox_arabic_evaluation.csv"), result_csv_path("arabic_native_review_sheet.csv")]
    tabs = st.tabs([path.name for path in files])
    for tab, path in zip(tabs, files):
        with tab:
            df = load_csv_dataframe(path)
            if df.empty:
                st.warning(f"Missing or empty: {project_relative(path)}")
            else:
                st.dataframe(df, use_container_width=True)
                audio_cols = [col for col in df.columns if "output_path" in col or "audio" in col]
                if audio_cols:
                    audio_values = [resolve_existing_audio_path(value) for value in df[audio_cols[0]].astype(str).tolist() if value]
                    audio_values = [path for path in audio_values if path and path.exists()]
                    if audio_values:
                        selected = st.selectbox("Play related audio", audio_values, format_func=lambda p: p.name, key=f"ar_{path.name}")
                        show_audio_file(selected)


def render_run_experiment_page() -> None:
    st.header("Run a model")
    selected = st.selectbox("Model", list(MODEL_CONFIG.keys()))
    status = get_model_status(selected)
    st.json({k: str(v) for k, v in status.items() if k not in {"report", "wav_files"}})
    args = st.text_input("Arguments", value="")
    can_run = status["python_exists"] and status["script_exists"]
    if st.button("Run", disabled=not can_run):
        process, command = run_script(status["python_path"], status["script_path"], args.split())
        st.code(subprocess.list2cmdline(command), language="powershell")
        output = []
        box = st.empty()
        while True:
            line = process.stdout.readline() if process.stdout else ""
            if line:
                output.append(line.rstrip())
                box.code("\n".join(output[-150:]), language=None)
            if process.poll() is not None:
                rest = process.stdout.read() if process.stdout else ""
                if rest: output.extend(rest.splitlines())
                box.code("\n".join(output[-200:]), language=None)
                break
            time.sleep(0.1)
        st.success("Experiment completed") if process.returncode == 0 else st.error(f"Exit code {process.returncode}")


def render_voice_comparison_page() -> None:
    st.header("Voice Comparison")
    refs = {name: path for name, path in REFERENCE_AUDIO_OPTIONS.items() if path.exists()}
    if refs:
        reference = st.selectbox("Reference", list(refs.keys()))
        show_audio_file(refs[reference], "Reference")
    selected_models = st.multiselect("Models", list(MODEL_CONFIG.keys()), default=["MeloTTS English", "XTTS-v2 Arabic"])
    cols = st.columns(max(1, len(selected_models)))
    for col, model_name in zip(cols, selected_models):
        with col:
            st.subheader(model_name)
            wavs = get_model_status(model_name)["wav_files"]
            if not wavs:
                st.warning("No WAV files found")
                continue
            selected = st.selectbox("Clip", wavs, format_func=lambda path: path.name, key=f"compare_{model_name}")
            show_audio_file(selected)


def result_csvs_for_language(language: str) -> list[Path]:
    keywords = {
        "English": ("english", "melotts", "neutts"),
        "Hindi": ("hindi", "indicf5"),
        "Arabic": ("arabic", "xtts"),
    }[language]
    csv_files = sorted(RESULTS_DIR.rglob("*.csv")) if RESULTS_DIR.exists() else []
    return [
        path for path in csv_files
        if any(keyword in path.name.lower() for keyword in keywords)
    ]


def render_results_by_language_page() -> None:
    st.header("Results")
    tabs = st.tabs(["English", "Hindi", "Arabic"] )
    for tab, language in zip(tabs, ["English", "Hindi", "Arabic"]):
        with tab:
            st.subheader(language)
            files = result_csvs_for_language(language)
            if not files:
                st.warning(f"No CSV results found for {language}.")
                continue
            for csv_path in files:
                with st.expander(project_relative(csv_path), expanded=False):
                    df = load_csv_dataframe(csv_path)
                    if df.empty:
                        st.warning("This CSV is missing or empty.")
                    else:
                        st.dataframe(df, use_container_width=True)


def render_reports_page() -> None:
    st.header("Reports and metrics")
    csv_files = sorted(RESULTS_DIR.rglob("*.csv")) if RESULTS_DIR.exists() else []
    if csv_files:
        selected = st.selectbox("Result CSV", csv_files, format_func=lambda path: project_relative(path))
        st.dataframe(load_csv_dataframe(selected), use_container_width=True)
    logs_dir = EVIDENCE_DIR / "terminal_logs"
    log_files = sorted(logs_dir.glob("*.txt"), key=lambda path: path.stat().st_mtime, reverse=True) if logs_dir.exists() else []
    if log_files:
        selected_log = st.selectbox("Terminal log", log_files, format_func=lambda path: path.name)
        st.code((load_text(selected_log) or "")[-20000:], language=None)



def render_final_results_page() -> None:
    st.header("Final Results")
    st.dataframe(pd.DataFrame(get_final_decision_rows()), use_container_width=True, hide_index=True)

    st.subheader("Detailed Evaluation CSVs")
    detailed_files = {
        "English": [
            result_csv_path("english_detailed_evaluation.csv"),
            result_csv_path("melotts_english_evaluation.csv"),
            result_csv_path("mms_english_evaluation.csv"),
            result_csv_path("neutts_english_evaluation.csv"),
        ],
        "Hindi": [
            result_csv_path("chatterbox_hindi_evaluation.csv"),
            result_csv_path("mms_hindi_evaluation.csv"),
            result_csv_path("hindi_model_comparison.csv", True),
        ],
        "Arabic": [
            result_csv_path("xtts_arabic_evaluation.csv"),
            result_csv_path("chatterbox_arabic_evaluation.csv"),
            result_csv_path("arabic_model_comparison.csv", True),
        ],
    }

    tabs = st.tabs(["English", "Hindi", "Arabic"])
    for tab, language in zip(tabs, ["English", "Hindi", "Arabic"]):
        with tab:
            for csv_path in detailed_files[language]:
                expanded = csv_path.name == "english_detailed_evaluation.csv"
                with st.expander(csv_path.name, expanded=expanded):
                    df = load_csv_dataframe(csv_path)
                    if df.empty:
                        st.warning(f"Missing or empty: {project_relative(csv_path)}")
                    else:
                        st.caption(project_relative(csv_path))
                        st.dataframe(df, use_container_width=True)


# ==============================================================================
# MAIN APP ENTRY
# ==============================================================================
st.set_page_config(page_title="Infinia Voice Lab", page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
    :root { --ink: #111827; --muted: #526071; --panel: #ffffff; --panel-soft: #f8fafc; --line: #d8e0ea; --brand: #2563eb; --brand-2: #0891b2; --accent: #f59e0b; }
    .stApp { color: var(--ink); background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 44%, #f8fafc 100%); }
    .review-hero { display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 26px 28px; margin: 4px 0 22px; border: 1px solid var(--line); border-radius: 8px; background: linear-gradient(135deg, #ffffff 0%, #eaf4ff 58%, #eefdfa 100%); box-shadow: 0 14px 38px rgba(15, 23, 42, .08); }
    .review-hero h1 { color: #0f172a; margin: 4px 0 8px; font-size: 2.1rem; }
    .eyebrow { font-size: .72rem; letter-spacing: .12em; color: #1d4ed8; font-weight: 800; }
    .hero-badge { white-space: nowrap; border: 1px solid #bfdbfe; background: #eff6ff; padding: 8px 12px; border-radius: 999px; color: #1e3a8a !important; font-size: .82rem; font-weight: 700; }
    .sample-strip { display: grid; grid-template-columns: auto 1.7fr 1fr; gap: 18px; align-items: center; margin: 16px 0; padding: 16px 18px; border: 1px solid var(--line); background: #ffffff; border-radius: 8px; }
</style>""", unsafe_allow_html=True)

st.markdown("## 🎙️ Infinia Voice Lab")
st.caption("Run open-source TTS experiments, inspect evidence, and build an auditable multilingual voice benchmark.")

# UPDATED NAVIGATION
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Final Results",
        "Audio Clips",
        "Evidence & Logs",
        "Generated Audio",
        "Results",
        "Voice Comparison",
    ],
)

# PAGE ROUTING
if page == "Overview":
    render_recruiter_overview()
elif page == "Final Results":
    render_final_results_page()
elif page == "Audio Clips":
    render_recruiter_audio_clips()
elif page == "Evidence & Logs":
    render_evidence_and_logs_page()
elif page == "Generated Audio":
    render_generated_audio_page()
elif page == "Results":
    render_results_by_language_page()
elif page == "Voice Comparison":
    render_voice_comparison_page()
