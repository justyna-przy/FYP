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


# ---- Simple dataset builder (folder-walk, no manifests) ----

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


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


def _stem(p: Path) -> str:
    return p.stem


def build_spectrogram_dataset_v1(
    *,
    out_name: str = "spectrograms_v1",
    train_shape_hw: Tuple[int, int] = (64, 128),  # (H, W)
    non_bird_cap: int = 1000,
    seed: int = 22,
    save_full: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Walk:
      bird_data/clips/species/<species>/*.wav
      bird_data/clips/non_bird/<species>/*.wav

    Output:
      bird_data/<out_name>/full/<class_name>/*.png
      bird_data/<out_name>/train_64x128/<class_name>/*.png
      bird_data/<out_name>/index_train_64x128.csv

    Classes:
      - each bird species is its own class_name (folder name)
      - all non_bird clips map to class_name = "non_bird"

    Global cap:
      - sample at most `non_bird_cap` non_bird wavs across all species folders
    """
    import random

    CONFIG = _get_config()
    if not hasattr(CONFIG, "spectrogram"):
        raise AttributeError("CONFIG.spectrogram missing in src/config.py")

    spec_cfg = CONFIG.spectrogram
    # Use config as the single source of truth for train output size.
    if train_shape_hw != spec_cfg.train_out_shape_hw:
        raise ValueError(
            f"train_shape_hw={train_shape_hw} does not match CONFIG.spectrogram.train_out_shape_hw="
            f"{spec_cfg.train_out_shape_hw}. Update config.py or pass the matching value."
        )

    data_dir = Path(CONFIG.paths.data_dir)          # "bird_data"
    clips_root = data_dir / CONFIG.paths.clips_dir  # "bird_data/clips"

    species_root = clips_root / "species"
    non_bird_root = clips_root / "non_bird"

    out_root = data_dir / out_name
    full_dir = out_root / "full"
    train_dir = out_root / f"train_{spec_cfg.train_out_shape_hw[0]}x{spec_cfg.train_out_shape_hw[1]}"

    # 1) collect species wavs
    species_rows: List[Dict] = []
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

    # 2) collect non_bird wavs (from all per-species subfolders)
    non_bird_wavs: List[Path] = []
    for sp_dir in _species_dirs(non_bird_root):
        non_bird_wavs.extend(_list_wavs(sp_dir))

    # apply global cap
    rnd = random.Random(seed)
    if non_bird_cap <= 0:
        non_bird_wavs = []
    elif len(non_bird_wavs) > non_bird_cap:
        non_bird_wavs = rnd.sample(non_bird_wavs, non_bird_cap)

    non_bird_rows = [
        {
            "wav_path": str(w),
            "class_name": "non_bird",
            "source": "non_bird",
        }
        for w in non_bird_wavs
    ]

    rows = species_rows + non_bird_rows
    if not rows:
        raise RuntimeError("No wav files found. Check bird_data/clips/ layout.")

    # deterministically shuffle index
    rnd.shuffle(rows)

    # 3) build index df + output paths
    df = pd.DataFrame(rows)
    df["clip_id"] = df["wav_path"].apply(lambda p: Path(p).stem)

    df["full_png"] = df.apply(
        lambda r: str((full_dir / r["class_name"] / f'{r["clip_id"]}.png')),
        axis=1,
    )
    df["train_png"] = df.apply(
        lambda r: str((train_dir / r["class_name"] / f'{r["clip_id"]}.png')),
        axis=1,
    )

    # 4) generate pngs
    errors: List[Tuple[str, str]] = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Generating spectrograms"):
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
        except Exception as e:
            errors.append((str(wav_path), str(e)))

    # 5) write index + error log
    out_root.mkdir(parents=True, exist_ok=True)
    index_csv = out_root / f"index_train_{train_shape_hw[0]}x{train_shape_hw[1]}.csv"
    df.to_csv(index_csv, index=False)

    if errors:
        err_path = out_root / "errors_spectrograms_v1.txt"
        with err_path.open("w", encoding="utf-8") as f:
            for wav, msg in errors:
                f.write(f"{wav}\t{msg}\n")

    return index_csv
