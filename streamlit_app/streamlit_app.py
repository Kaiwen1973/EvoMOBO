"""Streamlit interface for the EvoMOBO optimization workflow."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "evomobo" / "assets" / "input_learning"
OUTPUT_DIR = PROJECT_ROOT / "evomobo" / "assets" / "active_learning"
CHECKPOINT_ROOT = PROJECT_ROOT / "evomobo_checkpoints"
ARCHIVE_ROOT = PROJECT_ROOT / "data_archiving"
OPT_SCRIPT = PROJECT_ROOT / "scripts" / "black_box_opt.py"

MANAGED_INPUT_FILES = (
    "wt.fasta",
    "known_data.csv",
    "seed_data.csv",
    "pool.csv",
    "exam.csv",
    "par_cand_extra.csv",
)

ESM_VARIANTS = (
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
)
ESMC_VARIANTS = ("esmc_300m", "esmc_600m")
PROTBERT_VARIANTS = ("Rostlab/prot_bert_bfd", "Rostlab/prot_bert")

TASKS = {
    "Single objective": "single_objective_task",
    "Double objective": "double_objective_task",
    "Multi objective": "multi_objective_task",
}
ENCODERS = {
    "ESM-2": "esm",
    "ESM-C": "esmc",
    "ProtBERT": "protbert",
    "mCNN": "mcnn",
    "One-hot": "onehot",
}
SURROGATES = (
    "multi_task_exact_gp",
    "multi_task_gaussian_mlp",
    "multi_task_ibnn",
    "multi_task_vbll",
)
ACQUISITIONS = (
    "exploitative_qnehvi",
    "qnehvi",
    "qnparego",
    "qehvi",
    "qucb",
)
MULTI_SELECTION_MODES = (
    "greedy_mean_hvi",
    "rank1_batch_only",
    "ranked_batch",
)
SINGLE_SELECTION_MODES = (
    "greedy_mean_ei",
    "batch_ei",
    "rank1_batch_only",
)

ARCHIVABLE_DIRECTORIES = {
    "active_learning": OUTPUT_DIR,
    "input_learning": INPUT_DIR,
    "evomobo_checkpoints": CHECKPOINT_ROOT,
}

APP_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem;
}
.deployment-card {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 0.65rem;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
}
.evomobo-table {
    overflow-x: auto;
    margin: 0.35rem 0 0.75rem;
}
.evomobo-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.evomobo-table th,
.evomobo-table td {
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    padding: 0.35rem 0.5rem;
    text-align: left;
    white-space: nowrap;
}
.upload-guide-offset {
    height: 1.8rem;
}
.data-action-caption {
    color: rgba(49, 51, 63, 0.6);
    font-size: 0.875rem;
    line-height: 1.35;
    min-height: 2.8rem;
    margin-bottom: 0.35rem;
}
</style>
"""

RUNNING_LOCK_CSS = """
<style>
[data-testid="stAppViewContainer"] button:not([data-testid="stBaseButton-primary"]),
[data-testid="stSidebar"] button {
    pointer-events: none !important;
    cursor: not-allowed !important;
    opacity: 0.55 !important;
}
</style>
"""

LATER_ROUND_UPLOAD_CSS = """
<style>
[data-testid="stFileUploaderFile"] {
    display: none !important;
}
</style>
"""


def hydra_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def hydra_list(values: Iterable[object]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def render_frame(frame: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    display_frame = frame if max_rows is None else frame.head(max_rows)
    table = display_frame.to_html(
        index=False,
        border=0,
        escape=True,
        na_rep="",
    )
    st.markdown(
        f'<div class="evomobo-table">{table}</div>',
        unsafe_allow_html=True,
    )


class OptimizationJob:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._lines: list[str] = []
        self._return_code: Optional[int] = None
        self._job_id = 0
        self._stopped = False
        self._before_rounds: set[str] = set()
        self._checkpoint_name = "evomobo"

    def start(
        self,
        command: list[str],
        environment: dict[str, str],
        before_rounds: set[str],
        checkpoint_name: str,
    ) -> int:
        with self._lock:
            if self.is_running():
                raise RuntimeError("Another optimization process is already running.")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._process = process
            self._lines = []
            self._return_code = None
            self._stopped = False
            self._before_rounds = set(before_rounds)
            self._checkpoint_name = checkpoint_name
            self._job_id += 1
            job_id = self._job_id
            self._reader = threading.Thread(
                target=self._read_output,
                args=(process, job_id),
                daemon=True,
            )
            self._reader.start()
            return job_id

    def _read_output(self, process: subprocess.Popen, job_id: int) -> None:
        if process.stdout is not None:
            for line in process.stdout:
                with self._lock:
                    if job_id != self._job_id:
                        return
                    self._lines.append(line.rstrip())
                    if len(self._lines) > 10000:
                        del self._lines[:1000]
        return_code = process.wait()
        with self._lock:
            if job_id == self._job_id:
                self._return_code = return_code

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self._stopped = True
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            return_code = self._return_code
            if not running and return_code is None and self._process is not None:
                return_code = self._process.poll()
            return {
                "running": running,
                "lines": list(self._lines),
                "return_code": return_code,
                "job_id": self._job_id,
                "stopped": self._stopped,
                "before_rounds": set(self._before_rounds),
                "checkpoint_name": self._checkpoint_name,
            }


@st.cache_resource
def optimization_job() -> OptimizationJob:
    return OptimizationJob()


def deployment_settings() -> tuple[str, str, int]:
    mode = os.environ.get("EVOMOBO_STREAMLIT_MODE", "local").strip().lower()
    address = os.environ.get(
        "EVOMOBO_STREAMLIT_ADDRESS",
        str(st.get_option("server.address") or "localhost"),
    )
    port = int(
        os.environ.get(
            "EVOMOBO_STREAMLIT_PORT",
            str(st.get_option("server.port") or 6789),
        )
    )
    return mode, address, port


def render_deployment_status() -> None:
    mode, _, port = deployment_settings()
    public_host = os.environ.get("EVOMOBO_STREAMLIT_PUBLIC_HOST", "").strip()
    st.sidebar.header("Server")
    st.sidebar.metric("Mode", "Remote" if mode == "remote" else "Local")
    if mode == "remote":
        if public_host:
            st.sidebar.caption(
                f"Remote client URL: http://{public_host}:{port}"
            )
        st.sidebar.warning(
            f"Remote mode requires TCP port {port} to be open in the server "
            "firewall or cloud security group. Restrict access to trusted IPs."
        )
        st.warning(
            f"Remote mode is active on port {port}. Ensure that TCP port {port} "
            "is open and that access is limited to trusted users. This interface "
            "can upload files and start optimization jobs on the server."
        )
    else:
        st.sidebar.caption(f"Server-local URL: http://127.0.0.1:{port}")
        st.sidebar.success("Local mode is limited to this machine.")


def find_rounds() -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    numbered: list[tuple[int, Path]] = []
    for path in OUTPUT_DIR.iterdir():
        match = re.fullmatch(r"round(\d+)", path.name)
        if path.is_dir() and match:
            numbered.append((int(match.group(1)), path))
    return [path for _, path in sorted(numbered, reverse=True)]


def input_status_frame(
    filenames: Iterable[str],
    uploads: Optional[dict[str, object]] = None,
) -> pd.DataFrame:
    uploads = uploads or {}
    rows = []
    for filename in filenames:
        path = INPUT_DIR / filename
        upload = uploads.get(filename)
        available = upload is not None or path.is_file()
        size = (
            len(uploaded_bytes(upload))
            if upload is not None
            else path.stat().st_size if path.is_file() else None
        )
        rows.append(
            {
                "File": filename,
                "Status": (
                    "Selected"
                    if upload is not None
                    else "Available" if path.is_file() else "Missing"
                ),
                "Size (KB)": (
                    round(size / 1024, 1) if available and size is not None else None
                ),
            }
        )
    return pd.DataFrame(rows)


def objective_columns(frame: pd.DataFrame) -> list[str]:
    indexed = []
    for column in frame.columns:
        match = re.fullmatch(r"objective(\d+)", str(column))
        if match:
            indexed.append((int(match.group(1)), str(column)))
    indexed.sort()
    columns = [column for _, column in indexed]
    expected = [f"objective{idx}" for idx in range(len(columns))]
    if columns != expected:
        raise ValueError(
            "Objective columns must be consecutive from objective0; "
            f"found {columns}."
        )
    return columns


def read_csv_bytes(data: bytes, filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(data), engine="python")
    except Exception as error:
        raise ValueError(f"Unable to read {filename}: {error}") from error


def validate_fasta_bytes(data: bytes, filename: str = "wt.fasta") -> None:
    text = data.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise ValueError(
            f"{filename} must contain a FASTA header and a protein sequence."
        )
    sequence = "".join(lines[1:]).replace(" ", "")
    if not sequence or not sequence.isalpha():
        raise ValueError(f"{filename} does not contain a valid protein sequence.")


def validate_sequence_frame(
    frame: pd.DataFrame,
    filename: str,
    expected_objectives: int,
    require_rows: bool = True,
) -> None:
    required = {"Name", "mutant_sequence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {', '.join(missing)}.")
    columns = objective_columns(frame)
    if len(columns) != expected_objectives:
        raise ValueError(
            f"{filename} contains {len(columns)} objective columns, but the "
            f"selected task expects {expected_objectives}."
        )
    if require_rows and frame.empty:
        raise ValueError(f"{filename} contains no rows.")
    if frame.empty:
        return
    if frame["mutant_sequence"].isna().any():
        raise ValueError(f"{filename} contains an empty mutant_sequence.")
    if frame.loc[:, columns].isna().any().any():
        raise ValueError(f"{filename} contains missing objective values.")


def uploaded_bytes(uploaded_file: object) -> bytes:
    return bytes(uploaded_file.getbuffer())


def save_uploads(uploads: dict[str, object]) -> list[Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for filename, upload in uploads.items():
        if upload is None:
            continue
        destination = INPUT_DIR / filename
        temporary = destination.with_name(destination.name + ".uploading")
        temporary.write_bytes(uploaded_bytes(upload))
        os.replace(temporary, destination)
        saved.append(destination)
    return saved


def preview_file_data(filename: str, data: bytes) -> None:
    st.markdown(f"**{filename}**")
    if filename.endswith(".csv"):
        try:
            frame = read_csv_bytes(data, filename)
            render_frame(frame, max_rows=20)
        except ValueError as error:
            st.error(str(error))
    else:
        st.code(data.decode("utf-8", errors="replace")[:5000])


def uploader_key(stage: str, filename: str) -> str:
    revision_key = f"upload_revision_{stage}_{filename}"
    revision = int(st.session_state.get(revision_key, 0))
    return f"upload_{stage}_{filename}_{revision}"


def delete_input_file(stage: str, filename: str) -> None:
    path = INPUT_DIR / filename
    if path.is_file() or path.is_symlink():
        path.unlink()
    revision_key = f"upload_revision_{stage}_{filename}"
    st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1


def render_upload_actions(
    stage: str,
    filename: str,
    upload: object,
    disabled: bool,
) -> None:
    path = INPUT_DIR / filename
    data = uploaded_bytes(upload) if upload is not None else (
        path.read_bytes() if path.is_file() else None
    )
    preview_column, delete_column = st.columns(2)
    preview_clicked = preview_column.button(
        "Preview",
        key=f"preview_{stage}_{filename}",
        disabled=disabled or data is None,
        use_container_width=True,
    )
    delete_clicked = delete_column.button(
        "Delete",
        key=f"delete_{stage}_{filename}",
        disabled=disabled or data is None,
        use_container_width=True,
    )
    if delete_clicked:
        delete_input_file(stage, filename)
        st.rerun()
    if preview_clicked and data is not None:
        preview_file_data(filename, data)


def render_input_panel(disabled: bool = False) -> tuple[str, dict[str, object]]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    st.subheader("1. Input files")
    stage = st.radio(
        "Optimization stage",
        ("First round", "Later round"),
        horizontal=True,
        disabled=disabled,
    )
    if stage == "Later round":
        st.markdown(LATER_ROUND_UPLOAD_CSS, unsafe_allow_html=True)

    uploads: dict[str, object] = {}
    if stage == "First round":
        specifications = (
            ("known_data.csv", "Known evaluated sequences", ["csv"], True),
            ("seed_data.csv", "Additional training sequences", ["csv"], False),
            ("pool.csv", "Additional mutation-source pool", ["csv"], False),
            ("wt.fasta", "Wild-type FASTA", ["fasta"], True),
        )
    else:
        specifications = (
            ("exam.csv", "Completed evaluation results", ["csv"], True),
        )

    upload_column, guide_column = st.columns([1.15, 0.85])
    with upload_column:
        for filename, label, extensions, required in specifications:
            marker = "required" if required else "optional"
            uploads[filename] = st.file_uploader(
                f"{label} ({marker})",
                type=extensions,
                key=uploader_key(stage, filename),
                disabled=disabled,
            )
            render_upload_actions(
                stage,
                filename,
                uploads[filename],
                disabled,
            )
    with guide_column:
        st.markdown(
            '<div class="upload-guide-offset"></div>',
            unsafe_allow_html=True,
        )
        if stage == "First round":
            st.info(
                "known_data.csv and wt.fasta are required. seed_data.csv and "
                "pool.csv are optional."
            )
            render_frame(
                input_status_frame(
                    ("known_data.csv", "wt.fasta", "seed_data.csv", "pool.csv"),
                    uploads,
                )
            )
            st.markdown("**CSV format**")
            st.write(
                "CSV files must contain Name, mutant_sequence, and consecutive "
                "objective columns. Double-objective example:"
            )
            st.code(
                "Name,mutant_sequence,objective0,objective1\n"
                "E25G,SKPKRLCQVCGDHASGFHYGVWSCGGC...,7.14,9.15\n"
                "E25L,SKPKRLCQVCGDHASGFHYGVWSCLGC...,7.23,8.49\n"
                "E25C,SKPKRLCQVCGDHASGFHYGVWSCCGC...,7.43,8.08\n"
                "...",
                language="text",
            )
        else:
            st.info(
                "Before uploading a new completed exam.csv, delete the previous "
                "exam.csv or upload the new file directly to overwrite it. "
                "Existing first-round files are reused."
            )
            render_frame(
                input_status_frame(
                    ("known_data.csv", "wt.fasta", "exam.csv"),
                    uploads,
                )
            )
    return stage, uploads


def render_objective_settings(
    task_label: str,
    disabled: bool = False,
) -> tuple[int, list[str], list[float]]:
    if task_label == "Single objective":
        count = 1
    elif task_label == "Double objective":
        count = 2
    else:
        count = int(
            st.number_input(
                "Number of objectives",
                min_value=3,
                max_value=12,
                value=3,
                step=1,
                disabled=disabled,
            )
        )

    st.markdown("**Objective definitions**")
    directions = []
    weights = []
    for start in range(0, count, 4):
        columns = st.columns(min(4, count - start))
        for offset, column in enumerate(columns):
            index = start + offset
            with column:
                directions.append(
                    st.selectbox(
                        f"objective{index} direction",
                        ("min", "max"),
                        index=0 if index == 0 else 1,
                        key=f"objective_direction_{index}",
                        disabled=disabled,
                    )
                )
                weights.append(
                    float(
                        st.number_input(
                            f"objective{index} weight",
                            min_value=0.001,
                            value=1.0,
                            step=0.1,
                            key=f"objective_weight_{index}",
                            disabled=disabled,
                        )
                    )
                )
    return count, directions, weights


def render_encoder_settings(
    disabled: bool = False,
) -> tuple[str, Optional[str]]:
    encoder_label = st.selectbox(
        "Encoder（Before selecting an encoder, activate its corresponding Conda environment.）",
        tuple(ENCODERS),
        disabled=disabled,
    )
    variant = None
    if encoder_label == "ESM-2":
        variant = st.selectbox(
            "ESM-2 checkpoint",
            ESM_VARIANTS,
            index=3,
            disabled=disabled,
        )
    elif encoder_label == "ESM-C":
        variant = st.selectbox(
            "ESM-C checkpoint",
            ESMC_VARIANTS,
            disabled=disabled,
        )
    elif encoder_label == "ProtBERT":
        variant = st.selectbox(
            "ProtBERT checkpoint",
            PROTBERT_VARIANTS,
            disabled=disabled,
        )
    return ENCODERS[encoder_label], variant


def render_model_settings(
    task_label: str,
    disabled: bool = False,
) -> tuple[Optional[str], Optional[str], str]:
    if task_label == "Single objective":
        st.text_input(
            "Surrogate",
            value="single_task_exact_gp",
            disabled=True,
        )
        st.text_input(
            "Acquisition",
            value="single_objective_ei",
            disabled=True,
        )
        selection_label = st.selectbox(
            "Candidate selection",
            SINGLE_SELECTION_MODES,
            disabled=disabled,
        )
        return None, None, selection_label

    surrogate = st.selectbox("Surrogate", SURROGATES, disabled=disabled)
    acquisition = st.selectbox("Acquisition", ACQUISITIONS, disabled=disabled)
    selection_mode = st.selectbox(
        "Candidate selection",
        MULTI_SELECTION_MODES,
        disabled=disabled,
    )
    return surrogate, acquisition, selection_mode


def render_configuration_panel(disabled: bool = False) -> dict[str, object]:
    st.subheader("2. Optimization configuration")
    general_column, model_column = st.columns(2)
    with general_column:
        task_label = st.selectbox(
            "Task",
            tuple(TASKS),
            index=1,
            disabled=disabled,
        )
        project_name = st.text_input(
            "Project name",
            value="EVOMOBO",
            disabled=disabled,
        )
        checkpoint_name = st.text_input(
            "Checkpoint name",
            value="evomobo",
            disabled=disabled,
        )
        mutable_positions = st.text_input(
            "Mutable positions",
            value="25,26,28,29",
            disabled=disabled,
        )
        encoder, encoder_variant = render_encoder_settings(disabled)
    with model_column:
        surrogate, acquisition, selection_mode = render_model_settings(
            task_label,
            disabled,
        )
        batch_size = int(
            st.number_input(
                "Sequences per candidate batch",
                min_value=1,
                value=16,
                step=1,
                disabled=disabled,
            )
        )
        suggestions = int(
            st.number_input(
                "Final sequences to evaluate",
                min_value=1,
                value=16,
                step=1,
                disabled=disabled,
            )
        )
        num_gens = int(
            st.number_input(
                "Candidate batches",
                min_value=1,
                value=64,
                step=1,
                disabled=disabled,
            )
        )

    objective_count, directions, weights = render_objective_settings(
        task_label,
        disabled,
    )

    return {
        "task_label": task_label,
        "task": TASKS[task_label],
        "objective_count": objective_count,
        "directions": directions,
        "weights": weights,
        "project_name": project_name,
        "checkpoint_name": checkpoint_name,
        "mutable_positions": mutable_positions,
        "encoder": encoder,
        "encoder_variant": encoder_variant,
        "surrogate": surrogate,
        "acquisition": acquisition,
        "selection_mode": selection_mode,
        "batch_size": batch_size,
        "suggestions": suggestions,
        "num_gens": num_gens,
    }


def path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def path_contains_data(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        return True
    return path.is_dir() and any(path.iterdir())


def safe_archive_component(value: object, fallback: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    component = component.strip("._-")
    return component or fallback


def next_archive_directory(settings: dict[str, object]) -> Path:
    project = safe_archive_component(settings["project_name"], "EVOMOBO")
    task = safe_archive_component(settings["task"], "task")
    date = datetime.now().strftime("%Y-%m-%d")
    base = ARCHIVE_ROOT / f"{project}_{task}_{date}"
    candidate = base
    suffix = 2
    while path_exists(candidate):
        candidate = ARCHIVE_ROOT / f"{base.name}_{suffix}"
        suffix += 1
    return candidate


def reset_upload_widgets() -> None:
    for stage in ("First round", "Later round"):
        for filename in MANAGED_INPUT_FILES:
            revision_key = f"upload_revision_{stage}_{filename}"
            st.session_state[revision_key] = (
                int(st.session_state.get(revision_key, 0)) + 1
            )


def archive_current_data(settings: dict[str, object]) -> Path:
    sources = {
        name: path
        for name, path in ARCHIVABLE_DIRECTORIES.items()
        if path_contains_data(path)
    }
    if not sources:
        raise ValueError("No optimization data is available to archive.")

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archive_dir = next_archive_directory(settings)
    archive_dir.mkdir()
    for name, source in sources.items():
        shutil.move(str(source), str(archive_dir / name))
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return archive_dir


def archive_directories() -> list[Path]:
    if not ARCHIVE_ROOT.is_dir():
        return []
    return sorted(
        (path for path in ARCHIVE_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )


def restore_archived_data(archive_dir: Path) -> list[str]:
    available = {
        name: archive_dir / name
        for name in ARCHIVABLE_DIRECTORIES
        if path_exists(archive_dir / name)
    }
    if not available:
        raise ValueError("The selected archive contains no reusable data directories.")

    conflicts = [
        name
        for name in available
        if path_contains_data(ARCHIVABLE_DIRECTORIES[name])
    ]
    if conflicts:
        raise FileExistsError(
            "Current data already exists in: "
            + ", ".join(conflicts)
            + ". Use Data archiving before Data reuse."
        )

    restored = []
    for name, source in available.items():
        destination = ARCHIVABLE_DIRECTORIES[name]
        if destination.is_dir() and not destination.is_symlink():
            destination.rmdir()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            symlinks=True,
        )
        restored.append(name)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return restored


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def delete_archive_directory(archive_dir: Path) -> None:
    if not path_exists(archive_dir):
        raise FileNotFoundError(f"Archive not found: {archive_dir.name}")
    remove_path(archive_dir)


def clear_current_data() -> list[str]:
    removed = []
    for name, path in ARCHIVABLE_DIRECTORIES.items():
        if path_exists(path):
            remove_path(path)
            removed.append(name)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return removed


def render_data_management(
    settings: dict[str, object],
    disabled: bool = False,
) -> None:
    st.subheader("3. Data archiving and reuse")
    notice = st.session_state.pop("data_management_notice", None)
    if notice:
        st.success(notice)

    archives = archive_directories()
    selected_name = st.selectbox(
        "Archived dataset",
        [path.name for path in archives] if archives else ["No archive available"],
        disabled=disabled or not archives,
    )

    archive_column, reuse_column = st.columns(2)
    with archive_column:
        st.markdown("**Data archiving**")
        st.markdown(
            '<div class="data-action-caption">Move the three current data '
            "directories into data_archiving.</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Data archiving",
            disabled=disabled,
            use_container_width=True,
        ):
            try:
                archive_dir = archive_current_data(settings)
            except (OSError, ValueError) as error:
                st.error(str(error))
            else:
                reset_upload_widgets()
                st.session_state["data_management_notice"] = (
                    f"Data moved to {archive_dir.relative_to(PROJECT_ROOT)}."
                )
                st.rerun()

    with reuse_column:
        st.markdown("**Data reuse**")
        st.markdown(
            '<div class="data-action-caption">Copy data directories from the '
            "selected archive into the workspace.</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Data reuse",
            disabled=disabled or not archives,
            use_container_width=True,
        ):
            selected = next(path for path in archives if path.name == selected_name)
            try:
                restored = restore_archived_data(selected)
            except (OSError, ValueError) as error:
                st.error(str(error))
            else:
                reset_upload_widgets()
                st.session_state["data_management_notice"] = (
                    "Restored: " + ", ".join(restored) + "."
                )
                st.rerun()

    cleanup_column, delete_column = st.columns(2)
    with cleanup_column:
        st.markdown("**Clean current data**")
        st.markdown(
            '<div class="data-action-caption">Remove the three current data '
            "directories.</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Clean current data",
            disabled=disabled,
            use_container_width=True,
        ):
            try:
                removed = clear_current_data()
            except OSError as error:
                st.error(str(error))
            else:
                reset_upload_widgets()
                st.session_state["data_management_notice"] = (
                    "Removed: " + ", ".join(removed) + "."
                    if removed
                    else "No current optimization data was present."
                )
                st.rerun()

    with delete_column:
        st.markdown("**Delete archive**")
        st.markdown(
            '<div class="data-action-caption">Permanently delete the selected '
            "archive directory.</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Delete selected archive",
            disabled=disabled or not archives,
            use_container_width=True,
        ):
            selected = next(path for path in archives if path.name == selected_name)
            try:
                delete_archive_directory(selected)
            except OSError as error:
                st.error(str(error))
            else:
                st.session_state["data_management_notice"] = (
                    f"Deleted archive: {selected.name}."
                )
                st.rerun()


def build_command(settings: dict[str, object]) -> list[str]:
    command = [
        sys.executable,
        str(OPT_SCRIPT),
        f"task={settings['task']}",
        f"encoder={settings['encoder']}",
    ]
    if settings["task_label"] != "Single objective":
        command.extend(
            [
                f"surrogate={settings['surrogate']}",
                f"acquisition={settings['acquisition']}",
            ]
        )

    encoder = str(settings["encoder"])
    variant = settings["encoder_variant"]
    if encoder == "esm":
        command.append(f"encoder.model.esm_variant={variant}")
    elif encoder == "esmc":
        command.append(f"encoder.model.model_name={variant}")
    elif encoder == "protbert":
        command.append(f"encoder.model.model_name={variant}")

    if settings["task_label"] == "Single objective":
        command.append(f"task.objective_direction={settings['directions'][0]}")
    else:
        command.append(
            "task.objective_directions="
            + hydra_list(settings["directions"])
        )
        command.append(
            "optimizer.obj_weights=" + hydra_list(settings["weights"])
        )

    command.extend(
        [
            "optimizer.mutable_positions="
            + hydra_string(str(settings["mutable_positions"]).strip()),
            f"task.batch_size={settings['batch_size']}",
            f"optimizer.num_gens={settings['num_gens']}",
            f"optimizer.offline_selection_mode={settings['selection_mode']}",
            f"optimizer.offline_num_suggestions={settings['suggestions']}",
            "optimizer.ckpt_name="
            + hydra_string(str(settings["checkpoint_name"]).strip()),
            "project_name="
            + hydra_string(str(settings["project_name"]).strip() or "EVOMOBO"),
        ]
    )
    return command


def display_command(command: list[str]) -> str:
    display = ["python", "scripts/black_box_opt.py", *command[2:]]
    return shlex.join(display)


def source_bytes(filename: str, uploads: dict[str, object]) -> Optional[bytes]:
    upload = uploads.get(filename)
    if upload is not None:
        return uploaded_bytes(upload)
    path = INPUT_DIR / filename
    return path.read_bytes() if path.is_file() else None


def validate_run_inputs(
    stage: str,
    uploads: dict[str, object],
    settings: dict[str, object],
) -> list[str]:
    errors = []
    expected_objectives = int(settings["objective_count"])
    known_data = source_bytes("known_data.csv", uploads)
    if known_data is None:
        errors.append("known_data.csv is required.")
    else:
        try:
            validate_sequence_frame(
                read_csv_bytes(known_data, "known_data.csv"),
                "known_data.csv",
                expected_objectives,
            )
        except ValueError as error:
            errors.append(str(error))

    wild_type = source_bytes("wt.fasta", uploads)
    if wild_type is None:
        errors.append("wt.fasta is required.")
    else:
        try:
            validate_fasta_bytes(wild_type)
        except ValueError as error:
            errors.append(str(error))

    if stage == "Later round" and uploads.get("exam.csv") is None:
        errors.append("A new completed exam.csv upload is required.")

    for filename in ("exam.csv", "seed_data.csv", "pool.csv", "par_cand_extra.csv"):
        data = source_bytes(filename, uploads)
        if data is None:
            continue
        if filename == "exam.csv" and stage == "First round" and uploads.get(filename) is None:
            continue
        try:
            validate_sequence_frame(
                read_csv_bytes(data, filename),
                filename,
                expected_objectives,
                require_rows=filename == "exam.csv",
            )
        except ValueError as error:
            errors.append(str(error))

    if not str(settings["mutable_positions"]).strip():
        errors.append("At least one mutable position is required.")
    if not str(settings["checkpoint_name"]).strip():
        errors.append("Checkpoint name cannot be empty.")
    return errors


def optimization_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    return environment


def sync_completed_job(job: OptimizationJob) -> None:
    snapshot = job.snapshot()
    if snapshot["running"] or snapshot["return_code"] is None:
        return
    job_id = int(snapshot["job_id"])
    if int(st.session_state.get("handled_job_id", -1)) == job_id:
        return

    before = set(snapshot["before_rounds"])
    after = find_rounds()
    new_rounds = [path for path in after if str(path.resolve()) not in before]
    lines = snapshot["lines"]
    st.session_state["last_log"] = "\n".join(str(line) for line in lines)
    st.session_state["last_return_code"] = int(snapshot["return_code"])
    st.session_state["last_checkpoint_name"] = str(snapshot["checkpoint_name"])
    st.session_state["last_round"] = str(new_rounds[0]) if new_rounds else ""
    st.session_state["last_stopped"] = bool(snapshot["stopped"])
    st.session_state["handled_job_id"] = job_id


def render_running_status(job: OptimizationJob) -> None:
    snapshot = job.snapshot()
    if not snapshot["running"]:
        return
    st.markdown(RUNNING_LOCK_CSS, unsafe_allow_html=True)
    st.warning(
        "EvoMOBO is running. All controls are disabled until the process "
        "finishes or is stopped."
    )
    if st.button("Stop optimization", type="primary", key="stop_optimization"):
        try:
            job.stop()
        except (OSError, subprocess.TimeoutExpired) as error:
            st.error(f"Unable to stop the optimization process: {error}")
        st.rerun()
    lines = snapshot["lines"]
    if lines:
        st.code("\n".join(str(line) for line in lines[-300:]), language=None)


def render_command_and_run(
    stage: str,
    uploads: dict[str, object],
    settings: dict[str, object],
    job: OptimizationJob,
    disabled: bool = False,
) -> None:
    st.subheader("4. Review and run")
    command = build_command(settings)
    st.code(display_command(command), language="bash")
    st.caption(
        "The command runs in the same Python environment that started Streamlit."
    )

    if not st.button(
        "Run one optimization round",
        type="primary",
        disabled=disabled,
    ):
        return_code = st.session_state.get("last_return_code")
        if return_code is None:
            return
        if st.session_state.get("last_stopped", False):
            st.warning("The last optimization process was stopped by the user.")
        elif int(return_code) == 0 and st.session_state.get("last_round"):
            last_round = Path(str(st.session_state["last_round"]))
            st.success(
                f"Optimization completed. Results were saved to {last_round.name}."
            )
        elif int(return_code) == 0:
            st.error(
                "The process exited without creating a committed round directory. "
                "Review the log for the underlying EvoMOBO error."
            )
        else:
            st.error(f"The process exited with code {return_code}.")
        return

    errors = validate_run_inputs(stage, uploads, settings)
    if errors:
        for error in errors:
            st.error(error)
        return

    if job.is_running():
        st.error("Another optimization process is already running.")
        return

    try:
        saved = save_uploads(uploads)
        if saved:
            st.info("Saved: " + ", ".join(path.name for path in saved))
        before = {str(path.resolve()) for path in find_rounds()}
        job_id = job.start(
            command,
            optimization_environment(),
            before,
            str(settings["checkpoint_name"]),
        )
        st.session_state["handled_job_id"] = job_id - 1
        st.rerun()
    except (OSError, RuntimeError) as error:
        st.error(str(error))


def render_configuration_tab(
    job: OptimizationJob,
    disabled: bool = False,
) -> None:
    stage, uploads = render_input_panel(disabled)
    st.divider()
    settings = render_configuration_panel(disabled)
    st.divider()
    render_data_management(settings, disabled)
    st.divider()
    render_command_and_run(stage, uploads, settings, job, disabled)


def render_csv(path: Path) -> None:
    try:
        frame = pd.read_csv(path)
        render_frame(frame, max_rows=100)
    except Exception as error:
        st.warning(f"Unable to preview {path.name}: {error}")


def download_file(
    path: Path,
    label: Optional[str] = None,
    disabled: bool = False,
) -> None:
    st.download_button(
        label or f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv" if path.suffix.lower() == ".csv" else "application/octet-stream",
        use_container_width=True,
        key=f"download_{path.resolve()}",
        disabled=disabled,
    )


def render_results_tab(disabled: bool = False) -> None:
    st.subheader("Committed optimization rounds")
    rounds = find_rounds()
    if not rounds:
        st.info("No committed round directory is available.")
    else:
        selected_name = st.selectbox(
            "Round",
            [path.name for path in rounds],
            disabled=disabled,
        )
        selected = next(path for path in rounds if path.name == selected_name)
        st.caption(str(selected.relative_to(PROJECT_ROOT)))

        primary_files = (
            selected / "to_eval.csv",
            selected / "exam_template.csv",
        )
        for path in primary_files:
            if path.is_file():
                st.markdown(f"#### {path.name}")
                render_csv(path)
                download_file(path, disabled=disabled)

        additional = sorted(
            path
            for path in selected.iterdir()
            if path.is_file() and path not in primary_files
        )
        if additional:
            with st.expander("Additional round files"):
                for path in additional:
                    column1, column2 = st.columns([3, 1])
                    column1.write(path.name)
                    with column2:
                        download_file(path, disabled=disabled)

    st.divider()
    st.subheader("Checkpoint")
    checkpoint_name = str(
        st.session_state.get("last_checkpoint_name", "evomobo")
    )
    checkpoint_dir = CHECKPOINT_ROOT / checkpoint_name
    latest_checkpoint = checkpoint_dir / "latest.pt"
    if latest_checkpoint.exists() or latest_checkpoint.is_symlink():
        st.success(f"Latest checkpoint: {latest_checkpoint}")
    else:
        st.info(f"No latest checkpoint found under {checkpoint_dir}.")

    st.divider()
    st.subheader("Last session log")
    last_log = st.session_state.get("last_log")
    if last_log:
        st.code("\n".join(str(last_log).splitlines()[-500:]), language=None)
    else:
        st.info("No optimization log is available in this browser session.")


def main() -> None:
    st.set_page_config(
        page_title="EvoMOBO",
        layout="wide",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    job = optimization_job()
    sync_completed_job(job)
    running = job.is_running()
    render_deployment_status()
    st.title("EvoMOBO")
    st.caption(
        "Sequence optimization with configurable protein encoders, surrogate "
        "models, acquisition functions, and offline candidate selection."
    )
    render_running_status(job)
    configuration_tab, results_tab = st.tabs(
        ("Configuration & Run", "Results & Logs")
    )
    with configuration_tab:
        render_configuration_tab(job, disabled=running)
    with results_tab:
        render_results_tab(disabled=running)

    if running:
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
