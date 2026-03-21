# src/dataset/spectrograms.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image
from tqdm import tqdm


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


def _get_config():
    # lazy import to avoid circular imports
    from src.config import CONFIG  # type: ignore

    return CONFIG


def _list_wavs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])


def _species_dirs(root: Path) -> List[Path]:
    # returns immediate subdirectories (each species folder)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


@dataclass(frozen=True)
class SpectrogramBuildPaths:
    data_dir: Path
    clips_root: Path
    species_root: Path
    non_bird_root: Path
    out_root: Path
    full_dir: Path
    train_dir: Path


def resolve_spectrogram_build_paths(
    *,
    out_name: str = "spectrograms_v1",
    train_shape_hw: Tuple[int, int] = (64, 128),
) -> SpectrogramBuildPaths:
    """Resolve input/output directories for the spectrogram build."""
    CONFIG = _get_config()
    data_dir = Path(CONFIG.paths.data_dir)
    clips_root = data_dir / CONFIG.paths.clips_dir

    species_root = clips_root / "species"
    non_bird_root = clips_root / "non_bird"

    out_root = data_dir / out_name
    full_dir = out_root / "full"
    train_dir = out_root / f"train_{train_shape_hw[0]}x{train_shape_hw[1]}"

    return SpectrogramBuildPaths(
        data_dir=data_dir,
        clips_root=clips_root,
        species_root=species_root,
        non_bird_root=non_bird_root,
        out_root=out_root,
        full_dir=full_dir,
        train_dir=train_dir,
    )


def collect_species_clip_rows(species_root: Path) -> List[Dict[str, str]]:
    """Collect all species wav files as index rows."""
    species_rows: List[Dict[str, str]] = []
    for sp_dir in _species_dirs(species_root):
        sp_name = sp_dir.name
        for wav in _list_wavs(sp_dir):
            species_rows.append(
                {
                    "wav_path": str(wav),
                    "class_name": sp_name,
                    "source": "species",
                }
            )
    return species_rows


def collect_capped_non_bird_rows(
    non_bird_root: Path,
    *,
    non_bird_cap: int = 1000,
    seed: int = 22,
) -> List[Dict[str, str]]:
    """Collect non-bird wav files and apply a global cap across folders."""
    non_bird_wavs: List[Path] = []
    for sp_dir in _species_dirs(non_bird_root):
        non_bird_wavs.extend(_list_wavs(sp_dir))

    rnd = random.Random(seed)
    if non_bird_cap <= 0:
        non_bird_wavs = []
    elif len(non_bird_wavs) > non_bird_cap:
        non_bird_wavs = rnd.sample(non_bird_wavs, non_bird_cap)

    return [
        {
            "wav_path": str(w),
            "class_name": "non_bird",
            "source": "non_bird",
        }
        for w in non_bird_wavs
    ]


def combine_and_shuffle_clip_rows(
    species_rows: Sequence[Dict[str, str]],
    non_bird_rows: Sequence[Dict[str, str]],
    *,
    seed: int = 22,
) -> List[Dict[str, str]]:
    """Combine species + non-bird rows and deterministically shuffle."""
    rows = [dict(r) for r in species_rows] + [dict(r) for r in non_bird_rows]
    if not rows:
        raise RuntimeError("No wav files found. Check bird_data/clips/ layout.")

    rnd = random.Random(seed)
    rnd.shuffle(rows)
    return rows


def build_spectrogram_index_dataframe(
    rows: Sequence[Dict[str, str]],
    *,
    full_dir: Path,
    train_dir: Path,
) -> pd.DataFrame:
    """Build index dataframe and fill full/train PNG output paths."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise RuntimeError("No rows provided for spectrogram index.")

    df["clip_id"] = df["wav_path"].apply(lambda p: Path(p).stem)
    df["full_png"] = df.apply(
        lambda r: str((full_dir / r["class_name"] / f'{r["clip_id"]}.png')),
        axis=1,
    )
    df["train_png"] = df.apply(
        lambda r: str((train_dir / r["class_name"] / f'{r["clip_id"]}.png')),
        axis=1,
    )
    return df


def generate_spectrogram_pngs_from_index(
    index_df: pd.DataFrame,
    *,
    spec_cfg,
    save_full: bool = True,
    overwrite: bool = False,
) -> List[Tuple[str, str]]:
    """Generate full/train PNGs for every row in the index dataframe."""
    errors: List[Tuple[str, str]] = []

    for _, r in tqdm(index_df.iterrows(), total=len(index_df), desc="Generating spectrograms"):
        wav_path = Path(r["wav_path"])
        out_full = Path(r["full_png"])
        out_train = Path(r["train_png"])

        try:
            if not wav_path.exists():
                raise FileNotFoundError(str(wav_path))

            if not overwrite:
                ok_full = (not save_full) or out_full.exists()
                ok_train = out_train.exists()
                if ok_full and ok_train:
                    continue

            out_full.parent.mkdir(parents=True, exist_ok=True)
            out_train.parent.mkdir(parents=True, exist_ok=True)

            make_both_pngs(
                wav_path=wav_path,
                out_full_png=out_full,
                out_train_png=out_train,
                cfg=spec_cfg,
            )
        except Exception as exc:
            errors.append((str(wav_path), str(exc)))

    return errors


def write_spectrogram_index_and_errors(
    index_df: pd.DataFrame,
    *,
    out_root: Path,
    train_shape_hw: Tuple[int, int],
    errors: Sequence[Tuple[str, str]],
) -> Path:
    """Write index CSV and optional error log."""
    out_root.mkdir(parents=True, exist_ok=True)
    index_csv = out_root / f"index_train_{train_shape_hw[0]}x{train_shape_hw[1]}.csv"
    index_df.to_csv(index_csv, index=False)

    if errors:
        err_path = out_root / "errors_spectrograms_v1.txt"
        with err_path.open("w", encoding="utf-8") as f:
            for wav, msg in errors:
                f.write(f"{wav}\t{msg}\n")

    return index_csv


def compute_class_counts(index_df: pd.DataFrame) -> pd.Series:
    """Return class counts sorted descending."""
    if "class_name" not in index_df.columns:
        raise ValueError("Expected a class_name column in the index dataframe.")
    return index_df["class_name"].value_counts().sort_values(ascending=False)
