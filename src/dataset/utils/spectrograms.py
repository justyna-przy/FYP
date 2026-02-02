# src/dataset/spectrograms.py

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from PIL import Image
import librosa


def compute_logmel_db(
    wav_path: str | Path,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    n_mels: int,
    fmin: float,
    fmax: float,
    power: float,
    center: bool,
    db_ref: str,
) -> np.ndarray:
    """Returns log-mel in dB (float32), shape (n_mels, T)."""
    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
        center=center,
    )

    if db_ref == "max":
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
    elif db_ref == "1.0":
        mel_db = librosa.power_to_db(mel_power, ref=1.0)
    else:
        raise ValueError("db_ref must be 'max' or '1.0'")

    return mel_db.astype(np.float32)


def db_to_uint8(
    mel_db: np.ndarray,
    *,
    db_min: float,
    db_max: float,
    flip_freq_axis: bool,
) -> np.ndarray:
    """Clamp dB then map to uint8 [0,255]."""
    mel_db = np.clip(mel_db, db_min, db_max)

    if flip_freq_axis:
        mel_db = mel_db[::-1, :]  # flip vertically so low freqs appear at bottom

    x = (mel_db - db_min) / (db_max - db_min + 1e-12)
    x = np.clip(x, 0.0, 1.0)
    img_u8 = (x * 255.0).round().astype(np.uint8)
    return img_u8


def save_png(img_u8: np.ndarray, out_path: str | Path, resize_hw: Optional[Tuple[int, int]] = None):
    """Save grayscale PNG. resize_hw is (H, W)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pil = Image.fromarray(img_u8, mode="L")

    if resize_hw is not None:
        out_h, out_w = resize_hw
        pil = pil.resize((out_w, out_h), resample=Image.BILINEAR)

    pil.save(out_path)


def make_both_pngs(
    wav_path: str | Path,
    out_full_png: str | Path,
    out_train_png: str | Path,
    cfg,
):
    """Compute full-res logmel -> save full PNG + downsampled train PNG."""
    mel_db = compute_logmel_db(
        wav_path,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=cfg.window,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=cfg.power,
        center=cfg.center,
        db_ref=cfg.db_ref,
    )

    img_u8 = db_to_uint8(
        mel_db,
        db_min=cfg.db_min,
        db_max=cfg.db_max,
        flip_freq_axis=cfg.flip_freq_axis,
    )

    if cfg.save_full_res_png:
        save_png(img_u8, out_full_png, resize_hw=None)

    if cfg.save_train_png:
        save_png(img_u8, out_train_png, resize_hw=cfg.train_out_shape_hw)


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import random

import pandas as pd
from tqdm import tqdm


def _get_config():
    """
    Import CONFIG lazily to avoid annoying import-order / circular issues.
    """
    from src.config import CONFIG  # type: ignore
    return CONFIG


def _find_manifest_files(manifests_dir: Path) -> List[Path]:
    manifests_dir = Path(manifests_dir)
    if not manifests_dir.exists():
        raise FileNotFoundError(f"Manifests dir not found: {manifests_dir}")
    files = sorted(manifests_dir.glob("*_clips.csv"))
    if not files:
        raise FileNotFoundError(f"No *_clips.csv found in: {manifests_dir}")
    return files


def _pick_wav_path_column(df: pd.DataFrame) -> str:
    # Common columns from your pipeline / typical variants
    for c in ["saved_path", "wav_path", "path", "file_path", "local_path"]:
        if c in df.columns:
            return c

    # Fallback: any column containing "path"
    path_cols = [c for c in df.columns if "path" in c.lower()]
    if path_cols:
        return path_cols[0]

    raise ValueError(f"Could not find a wav path column. Columns={list(df.columns)}")


def _normalize_class_name(s: str) -> str:
    return str(s).strip().lower()


def _resolve_path(p: str | Path) -> Path:
    """
    Resolve a path that might be absolute, relative, or weirdly rooted.
    Keep it simple: try as-is, then CWD-relative, then ../ relative.
    """
    p = Path(str(p))
    if p.exists():
        return p
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    if (Path("..") / p).exists():
        return (Path("..") / p).resolve()
    return p  # may be missing; caller can decide what to do


def build_index_from_manifests(
    manifests_dir: Path,
) -> pd.DataFrame:
    """
    Build a unified index from per-species *_clips.csv manifests.

    Output columns (minimum):
      - wav_path
      - class_name
      - target_species
      - teacher_decision
      - plus metadata columns if present
    """
    files = _find_manifest_files(Path(manifests_dir))

    rows: List[Dict[str, Any]] = []

    for mf in files:
        df = pd.read_csv(mf)
        if df.empty:
            continue

        if "teacher_decision" not in df.columns:
            raise ValueError(f"'teacher_decision' missing in {mf}. Columns={list(df.columns)}")

        wav_col = _pick_wav_path_column(df)
        fallback_species = _normalize_class_name(mf.name.replace("_clips.csv", ""))

        for _, r in df.iterrows():
            decision = str(r.get("teacher_decision", "")).strip()
            if decision == "drop":
                continue

            wav_path = _resolve_path(r.get(wav_col))
            target_species = _normalize_class_name(r.get("target_species", fallback_species))

            if decision == "species":
                class_name = target_species
            elif decision == "non_bird":
                class_name = "non_bird"
            else:
                # ignore unexpected labels
                continue

            rows.append(
                {
                    "manifest_file": mf.as_posix(),
                    "wav_path": str(wav_path),
                    "class_name": class_name,
                    "target_species": target_species,
                    "teacher_decision": decision,
                    # optional metadata if present
                    "xc_id": r.get("xc_id", None),
                    "start_s": r.get("start_s", r.get("start", None)),
                    "end_s": r.get("end_s", r.get("end", None)),
                    "rms_db": r.get("rms_db", None),
                    "teacher_top_species": r.get("teacher_top_species", r.get("teacher_top", None)),
                    "teacher_conf": r.get("teacher_conf", r.get("teacher_confidence", None)),
                }
            )

    idx = pd.DataFrame(rows)
    if idx.empty:
        raise ValueError(f"Index is empty. Check manifests in {manifests_dir}")
    return idx


def cap_non_bird_global(idx: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """
    Deterministically cap the number of non_bird samples globally.
    """
    if cap <= 0:
        # cap<=0 means "no non_bird"
        return idx[idx["class_name"] != "non_bird"].copy()

    non_bird = idx[idx["class_name"] == "non_bird"].copy()
    birds = idx[idx["class_name"] != "non_bird"].copy()

    if len(non_bird) > cap:
        non_bird = non_bird.sample(n=cap, random_state=seed)

    out = pd.concat([birds, non_bird], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _clip_stem(wav_path: str | Path) -> str:
    return Path(str(wav_path)).stem


def add_png_paths(
    idx: pd.DataFrame,
    out_root: Path,
) -> pd.DataFrame:
    """
    Add full_png and train_png paths to the index.
    """
    out_root = Path(out_root)
    full_dir = out_root / "full"
    train_dir = out_root / "train_64x128"

    full_paths = []
    train_paths = []
    for _, r in idx.iterrows():
        cls = r["class_name"]
        stem = _clip_stem(r["wav_path"])
        full_paths.append(str(full_dir / cls / f"{stem}.png"))
        train_paths.append(str(train_dir / cls / f"{stem}.png"))

    idx = idx.copy()
    idx["full_png"] = full_paths
    idx["train_png"] = train_paths
    return idx


def generate_pngs_from_index(
    idx: pd.DataFrame,
    *,
    spectrogram_cfg,
    save_full: bool,
    overwrite: bool = False,
) -> List[Tuple[str, str]]:
    """
    Generate full + train PNGs for every row in idx.
    Returns a list of (wav_path, error_str) for failures.
    """
    errors: List[Tuple[str, str]] = []


    for _, r in tqdm(idx.iterrows(), total=len(idx), desc="Generating spectrograms"):
        wav_path = Path(r["wav_path"])
        full_png = Path(r["full_png"])
        train_png = Path(r["train_png"])

        try:
            if not wav_path.exists():
                raise FileNotFoundError(str(wav_path))

            # skip if already exists
            if not overwrite:
                ok_full = (not save_full) or full_png.exists()
                ok_train = train_png.exists()
                if ok_full and ok_train:
                    continue

            full_png.parent.mkdir(parents=True, exist_ok=True)
            train_png.parent.mkdir(parents=True, exist_ok=True)

            make_both_pngs(
                wav_path=wav_path,
                out_full_png=full_png,
                out_train_png=train_png,
                cfg=spectrogram_cfg,
            )
        except Exception as e:
            errors.append((str(wav_path), str(e)))

    return errors


def build_spectrogram_dataset_v1(
    *,
    non_bird_cap: int = 1000,
    seed: int = 22,
    save_full: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    One-call dataset builder. Uses CONFIG to locate:
      - manifests: bird_data/manifests/clips_v1
      - output:    bird_data/spectrograms_v1
      - spectrogram cfg: CONFIG.spectrogram

    Writes:
      bird_data/spectrograms_v1/index_train_64x128.csv

    Returns the index CSV path.
    """
    CONFIG = _get_config()

    # Paths from config
    data_dir = Path(CONFIG.paths.data_dir)
    manifests_dir = data_dir / CONFIG.paths.manifests_dir

    out_root = data_dir / "spectrograms_v1"

    if not hasattr(CONFIG, "spectrogram"):
        raise AttributeError(
            "CONFIG.spectrogram missing. Add a SpectrogramConfig to src/config.py "
            "and expose it as CONFIG.spectrogram."
        )

    spec_cfg = CONFIG.spectrogram

    # Build index
    idx = build_index_from_manifests(manifests_dir)

    # Cap non_bird globally
    idx = cap_non_bird_global(idx, cap=non_bird_cap, seed=seed)

    # Add png paths
    idx = add_png_paths(idx, out_root=out_root)

    # Generate pngs
    errors = generate_pngs_from_index(
        idx,
        spectrogram_cfg=spec_cfg,
        save_full=save_full,
        overwrite=overwrite,
    )

    # Save index
    out_root.mkdir(parents=True, exist_ok=True)
    index_csv = out_root / "index_train_64x128.csv"
    idx.to_csv(index_csv, index=False)

    if errors:
        # Write a simple error log next to the index
        err_path = out_root / "errors_spectrograms_v1.txt"
        with err_path.open("w", encoding="utf-8") as f:
            for wav, msg in errors:
                f.write(f"{wav}\t{msg}\n")

    return index_csv
