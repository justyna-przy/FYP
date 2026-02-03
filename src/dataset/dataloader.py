# src/dataset/dataloader.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class AugmentConfig:
    # SpecAugment-lite params (applied on tensor after loading)
    enable: bool = True
    time_mask_prob: float = 0.8
    time_mask_max_width: int = 12  # columns (time frames)
    freq_mask_prob: float = 0.8
    freq_mask_max_width: int = 8   # rows (mel bins)
    time_shift_prob: float = 0.5
    time_shift_max: int = 8        # columns
    contrast_jitter_prob: float = 0.2
    contrast_jitter_strength: float = 0.08  # +/- 8%


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


def _load_classes(classes_json: str | Path) -> Dict[str, int]:
    classes_json = Path(classes_json)
    if not classes_json.exists():
        raise FileNotFoundError(f"classes.json not found: {classes_json}")
    return json.loads(classes_json.read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    return pd.read_csv(path)


class SpectrogramPNGs(Dataset):
    """
    Loads grayscale spectrogram PNGs and returns:
      x: float32 tensor, shape (1,H,W), in [0,1]
      y: int64
    """
    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        *,
        png_col: str = "train_png",
        class_col: str = "class_name",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.png_col = png_col
        self.class_col = class_col
        self.transform = transform

        missing = {png_col, class_col} - set(self.df.columns)
        if missing:
            raise ValueError(f"DF missing columns {missing}. Has: {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.df.iloc[idx]
        png_path = Path(str(r[self.png_col]))
        cls = str(r[self.class_col])

        # Load grayscale PNG -> float32 [0,1]
        img = Image.open(png_path).convert("L")
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).float() / 255.0  # (H,W)
        x = x.unsqueeze(0)  # (1,H,W)

        y = torch.tensor(self.class_to_idx[cls], dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y


def _specaugment(x: torch.Tensor, rng: np.random.Generator, cfg: AugmentConfig) -> torch.Tensor:
    """
    x: (1,H,W) float tensor
    Applies simple time/freq masks and time shift.
    """
    _, H, W = x.shape

    # time shift
    if cfg.time_shift_prob > 0 and rng.random() < cfg.time_shift_prob and cfg.time_shift_max > 0:
        shift = int(rng.integers(-cfg.time_shift_max, cfg.time_shift_max + 1))
        x = torch.roll(x, shifts=shift, dims=2)

    # time mask (mask columns)
    if cfg.time_mask_prob > 0 and rng.random() < cfg.time_mask_prob and cfg.time_mask_max_width > 0:
        w = int(rng.integers(1, cfg.time_mask_max_width + 1))
        start = int(rng.integers(0, max(1, W - w)))
        x[:, :, start:start + w] = 0.0

    # freq mask (mask rows)
    if cfg.freq_mask_prob > 0 and rng.random() < cfg.freq_mask_prob and cfg.freq_mask_max_width > 0:
        h = int(rng.integers(1, cfg.freq_mask_max_width + 1))
        start = int(rng.integers(0, max(1, H - h)))
        x[:, start:start + h, :] = 0.0

    # tiny contrast jitter (multiplicative)
    if cfg.contrast_jitter_prob > 0 and rng.random() < cfg.contrast_jitter_prob:
        strength = cfg.contrast_jitter_strength
        scale = float(1.0 + rng.uniform(-strength, strength))
        x = torch.clamp(x * scale, 0.0, 1.0)

    return x


def make_transforms(*, train: bool, seed: int = 22, aug: Optional[AugmentConfig] = None):
    """
    Returns a callable transform(x)->x.
    We keep it simple and torch-only.
    """
    if not train:
        return None

    aug = aug or AugmentConfig()
    if not aug.enable:
        return None

    rng = np.random.default_rng(seed)

    def transform(x: torch.Tensor) -> torch.Tensor:
        return _specaugment(x, rng, aug)

    return transform


def make_dataloaders(
    *,
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: Optional[str | Path],
    classes_json: str | Path,
    loader_cfg: Optional[LoaderConfig] = None,
    aug_cfg: Optional[AugmentConfig] = None,
    seed: int = 22,
) -> Dict[str, DataLoader]:
    """
    Returns dict: {"train": ..., "val": ..., "test": ... (optional)}
    """
    loader_cfg = loader_cfg or LoaderConfig()
    class_to_idx = _load_classes(classes_json)

    train_df = _read_csv(train_csv)
    val_df = _read_csv(val_csv)
    test_df = _read_csv(test_csv) if test_csv is not None else None

    train_tf = make_transforms(train=True, seed=seed, aug=aug_cfg)
    eval_tf = make_transforms(train=False, seed=seed, aug=aug_cfg)

    train_ds = SpectrogramPNGs(train_df, class_to_idx, transform=train_tf)
    val_ds = SpectrogramPNGs(val_df, class_to_idx, transform=eval_tf)
    test_ds = SpectrogramPNGs(test_df, class_to_idx, transform=eval_tf) if test_df is not None else None

    train_loader = DataLoader(
        train_ds,
        batch_size=loader_cfg.batch_size,
        shuffle=True,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        drop_last=False,
    )

    out = {"train": train_loader, "val": val_loader}
    if test_ds is not None:
        out["test"] = DataLoader(
            test_ds,
            batch_size=loader_cfg.batch_size,
            shuffle=False,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=False,
        )
    return out
