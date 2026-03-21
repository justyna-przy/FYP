from __future__ import annotations

import contextlib
import io
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

from src.config import CONFIG


TeacherWindowRow = Dict[str, Any]


@dataclass(frozen=True)
class PreprocessingPaths:
    data_dir: Path
    raw_dir: Path
    manifest_dir: Path
    clips_dir: Path
    species_clips_dir: Path
    nonbird_clips_dir: Path


def resolve_preprocessing_paths(repo_root: Path) -> PreprocessingPaths:
    """Resolve dataset paths with the same fallback behaviour as the notebook."""
    data_dir_cfg = Path(CONFIG.paths.data_dir)
    data_dir_candidates = [
        (repo_root / data_dir_cfg).resolve(),
        (Path.cwd() / data_dir_cfg).resolve(),
        (repo_root / "src" / data_dir_cfg).resolve(),
    ]

    data_dir = next((p for p in data_dir_candidates if p.exists()), data_dir_candidates[0])
    raw_dir = data_dir / CONFIG.paths.raw_dir
    manifest_dir = data_dir / CONFIG.paths.manifests_dir
    clips_dir = data_dir / CONFIG.paths.clips_dir

    return PreprocessingPaths(
        data_dir=data_dir,
        raw_dir=raw_dir,
        manifest_dir=manifest_dir,
        clips_dir=clips_dir,
        species_clips_dir=clips_dir / "species",
        nonbird_clips_dir=clips_dir / "non_bird",
    )


def ensure_preprocessing_output_dirs(paths: PreprocessingPaths) -> None:
    for p in [paths.raw_dir, paths.manifest_dir, paths.species_clips_dir, paths.nonbird_clips_dir]:
        p.mkdir(parents=True, exist_ok=True)


def create_birdnet_teacher_analyzer() -> Analyzer:
    analyzer = Analyzer()
    if analyzer is None:
        raise RuntimeError("Teacher analyzer not initialized.")
    return analyzer


def compute_rms_dbfs_for_waveform(waveform: np.ndarray, eps: float) -> float:
    rms = float(np.sqrt(np.mean(np.square(waveform)) + eps))
    return float(20.0 * math.log10(rms + eps))


def compute_window_start_times_for_recording(
    recording_len_s: float,
    *,
    clip_len_s: float,
    stride_s: float,
    skip_first_s: float,
) -> List[float]:
    starts: List[float] = []
    s = float(skip_first_s)
    while s + clip_len_s <= recording_len_s + 1e-9:
        starts.append(float(s))
        s += float(stride_s)
    return starts


def build_candidate_windows_from_recording(
    waveform_16k: np.ndarray,
    *,
    sample_rate_model: int,
    clip_len_s: float,
    stride_s: float,
    skip_first_s: float,
    rms_eps: float,
) -> List[TeacherWindowRow]:
    recording_len_s = float(len(waveform_16k) / sample_rate_model)
    starts = compute_window_start_times_for_recording(
        recording_len_s,
        clip_len_s=clip_len_s,
        stride_s=stride_s,
        skip_first_s=skip_first_s,
    )

    windows: List[TeacherWindowRow] = []
    clip_samples = int(round(clip_len_s * sample_rate_model))

    for start_s in starts:
        start_i = int(round(start_s * sample_rate_model))
        end_i = start_i + clip_samples
        if end_i > len(waveform_16k):
            continue

        clip_wave = waveform_16k[start_i:end_i]
        windows.append(
            {
                "start_s": float(start_s),
                "end_s": float(start_s + clip_len_s),
                "rms_db": compute_rms_dbfs_for_waveform(clip_wave, rms_eps),
                "wave_16k": clip_wave,
            }
        )

    return windows


def apply_adaptive_rms_gate_to_windows(
    windows: List[TeacherWindowRow],
    *,
    rms_keep_percentile: float,
    rms_abs_min_db: float,
) -> tuple[List[TeacherWindowRow], float]:
    if not windows:
        return [], float(rms_abs_min_db)

    rms_vals = np.array([w["rms_db"] for w in windows], dtype=float)
    gate_thr_db = max(float(rms_abs_min_db), float(np.percentile(rms_vals, rms_keep_percentile)))
    gated = [w for w in windows if float(w["rms_db"]) >= gate_thr_db]

    if not gated:
        gated = [windows[int(np.argmax(rms_vals))]]

    return gated, gate_thr_db


def classify_window_with_birdnet_teacher(
    window_wave_16k: np.ndarray,
    *,
    target_sci_name: str,
    analyzer: Analyzer,
    sample_rate_model: int,
    sample_rate_teacher: int,
    bird_conf_thr: float,
    species_conf_thr: float,
) -> Dict[str, Any]:
    # BirdNET expects 48 kHz input audio.
    y_48k = librosa.resample(window_wave_16k, orig_sr=sample_rate_model, target_sr=sample_rate_teacher)

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        sf.write(tmp_path, y_48k, sample_rate_teacher, subtype="PCM_16")
        rec = Recording(analyzer, tmp_path, min_conf=bird_conf_thr)
        with contextlib.redirect_stdout(io.StringIO()):
            rec.analyze()
        detections = rec.detections or []
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    top = None
    max_conf = 0.0
    for d in detections:
        conf = float(d.get("confidence", 0.0))
        if conf > max_conf:
            max_conf = conf
            top = d

    top_sci = (top.get("scientific_name") if top else "") or ""
    top_common = (top.get("common_name") if top else "") or ""
    top_conf = float(top.get("confidence", 0.0)) if top else 0.0

    target_norm = target_sci_name.strip().lower()
    is_target = bool(top_sci) and (top_sci.strip().lower() == target_norm) and (top_conf >= species_conf_thr)

    if len(detections) == 0:
        decision = "non_bird"
    elif is_target:
        decision = "species"
    else:
        decision = "drop"

    return {
        "detections": detections,
        "top_sci": top_sci,
        "top_common": top_common,
        "top_conf": top_conf,
        "max_conf": float(max_conf),
        "decision": decision,
    }


def resolve_downloaded_audio_path_from_manifest(
    local_path_value: str | Path,
    *,
    repo_root: Path,
    data_dir: Path,
) -> Path | None:
    raw = str(local_path_value).strip()
    if not raw:
        return None

    candidates: List[Path] = []
    for candidate_str in [raw, raw.replace("\\", "/")]:
        p = Path(candidate_str)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend(
                [
                    Path.cwd() / p,
                    repo_root / p,
                    data_dir.parent / p,
                    data_dir / p,
                ]
            )

    seen = set()
    for p in candidates:
        rp = p.resolve(strict=False)
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        if rp.exists():
            return rp
    return None


def save_selected_clip_as_wav(
    clip_wave_16k: np.ndarray,
    *,
    out_dir: Path,
    xc_id: str,
    start_s: float,
    end_s: float,
    sample_rate_model: int,
) -> str:
    start_ms = int(round(start_s * 1000))
    end_ms = int(round(end_s * 1000))
    fname = f"XC{xc_id}__s{start_ms}__e{end_ms}.wav"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    sf.write(str(out_path), clip_wave_16k, sample_rate_model, subtype="PCM_16")
    return str(out_path)


def _stable_seed_for_recording(xc_id: str, base_seed: int) -> int:
    try:
        numeric_id = int(float(xc_id))
    except Exception:
        numeric_id = abs(hash(xc_id)) % (2**31)
    return int(base_seed + numeric_id)


def select_training_windows_per_recording(
    teacher_labeled_windows: List[TeacherWindowRow],
    *,
    xc_id: str,
    max_species_clips_per_rec: int,
    nonbird_clips_per_rec: int,
    seed: int,
) -> tuple[List[TeacherWindowRow], List[TeacherWindowRow]]:
    species_positive = [w for w in teacher_labeled_windows if w.get("teacher_decision") == "species"]
    non_bird = [w for w in teacher_labeled_windows if w.get("teacher_decision") == "non_bird"]

    rng = random.Random(_stable_seed_for_recording(xc_id, seed))
    rng.shuffle(species_positive)
    selected_species = species_positive[:max_species_clips_per_rec]

    non_bird_sorted = sorted(non_bird, key=lambda x: float(x.get("teacher_max_conf", 0.0)))
    selected_nonbird = non_bird_sorted[:nonbird_clips_per_rec]
    return selected_species, selected_nonbird


def list_species_labels_with_downloaded_manifests(manifest_dir: Path) -> List[str]:
    files = sorted(manifest_dir.glob("*_downloaded.csv"))
    species: List[str] = []
    for f in files:
        match = re.match(r"(.+)_downloaded\.csv$", f.name)
        if match:
            species.append(match.group(1))
    return species


def load_downloaded_manifest_for_species(
    manifest_dir: Path,
    species_label: str,
    *,
    max_recordings: int | None = None,
) -> pd.DataFrame:
    in_csv = manifest_dir / f"{species_label}_downloaded.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing: {in_csv}")

    df = pd.read_csv(in_csv)
    if max_recordings is not None:
        df = df.head(max_recordings).copy()
    return df


def label_windows_with_birdnet_teacher(
    windows: List[TeacherWindowRow],
    *,
    target_sci_name: str,
    analyzer: Analyzer,
    sample_rate_model: int,
    sample_rate_teacher: int,
    bird_conf_thr: float,
    species_conf_thr: float,
) -> List[TeacherWindowRow]:
    labeled: List[TeacherWindowRow] = []

    for window in windows:
        teacher = classify_window_with_birdnet_teacher(
            window["wave_16k"],
            target_sci_name=target_sci_name,
            analyzer=analyzer,
            sample_rate_model=sample_rate_model,
            sample_rate_teacher=sample_rate_teacher,
            bird_conf_thr=bird_conf_thr,
            species_conf_thr=species_conf_thr,
        )

        labeled.append(
            {
                **window,
                "teacher_decision": teacher["decision"],
                "teacher_top_sci": teacher["top_sci"],
                "teacher_top_common": teacher["top_common"],
                "teacher_top_conf": teacher["top_conf"],
                "teacher_max_conf": teacher["max_conf"],
            }
        )

    return labeled


def build_selection_lookup_for_windows(
    selected_species: List[TeacherWindowRow],
    selected_nonbird: List[TeacherWindowRow],
) -> Dict[tuple[float, float], str]:
    lookup: Dict[tuple[float, float], str] = {}
    for w in selected_species:
        lookup[(float(w["start_s"]), float(w["end_s"]))] = "species"
    for w in selected_nonbird:
        lookup[(float(w["start_s"]), float(w["end_s"]))] = "non_bird"
    return lookup


def write_species_clip_manifest(
    manifest_dir: Path,
    species_label: str,
    rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "selected" not in df.columns:
        df["selected"] = 0
    if "final_class" not in df.columns:
        df["final_class"] = "drop"
    if "xc_id" not in df.columns:
        df["xc_id"] = ""

    out_csv = manifest_dir / f"{species_label}_clips.csv"
    df.to_csv(out_csv, index=False)
    return df
