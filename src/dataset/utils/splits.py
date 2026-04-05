# src/dataset/splits.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


_XC_RE = re.compile(r"(XC\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 22
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    stratify_col: str = "class_name"
    path_col: str = "train_png"  # we train on train-size PNGs
    group_col: str = "group_id"  # filled by add_group_id()


def _get_config():
    # Lazy import to avoid circular import issues
    from src.config import CONFIG  # type: ignore
    return CONFIG


def load_index(index_csv: str | Path) -> pd.DataFrame:
    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")
    df = pd.read_csv(index_csv)
    required = {"class_name", "train_png"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Index missing columns {missing}. Has: {list(df.columns)}")
    return df


def add_group_id(df: pd.DataFrame, *, wav_col: str = "wav_path", png_col: str = "train_png") -> pd.DataFrame:
    """
    Adds df['group_id'] for group-wise splitting.
    Prefer extracting XC id from wav_path if available, else from train_png stem.
    Falls back to stem if XC id not found.
    """
    df = df.copy()

    def extract_xc(s: str) -> str | None:
        m = _XC_RE.search(str(s))
        return m.group(1).upper() if m else None

    group_ids = []
    for _, r in df.iterrows():
        xc = None
        if wav_col in df.columns:
            xc = extract_xc(r[wav_col])
        if xc is None:
            xc = extract_xc(Path(str(r[png_col])).stem)
        if xc is None:
            # fallback: use stem (still deterministic, but group may be too fine)
            xc = Path(str(r[png_col])).stem
        group_ids.append(xc)

    df["group_id"] = group_ids
    return df


def _normalize_ratios(cfg: SplitConfig) -> Tuple[float, float, float]:
    s = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if s <= 0:
        raise ValueError("Split ratios sum to 0")
    return cfg.train_ratio / s, cfg.val_ratio / s, cfg.test_ratio / s


def _group_stratified_split(
    df: pd.DataFrame,
    *,
    stratify_col: str,
    group_col: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Group-wise stratified split.
    We assign entire groups (recordings) to train/val/test, attempting
    to preserve class proportions based on per-group dominant class.

    Note: For your dataset, each clip belongs to a class already, and group_id
    is recording-based. Most recordings should map to a single class; for non_bird
    you may have multiple per species folder, but that's still fine.
    """
    rng = np.random.default_rng(seed)

    # Map group -> majority class (based on counts)
    grp = df.groupby(group_col)[stratify_col].value_counts().rename("n").reset_index()
    grp_major = grp.sort_values(["n"], ascending=False).drop_duplicates(group_col)
    group_to_class = dict(zip(grp_major[group_col], grp_major[stratify_col]))

    groups = np.array(list(group_to_class.keys()))
    classes = np.array([group_to_class[g] for g in groups])

    # Shuffle groups within each class, then allocate by ratios
    train_groups, val_groups, test_groups = [], [], []
    for cls in np.unique(classes):
        cls_groups = groups[classes == cls]
        rng.shuffle(cls_groups)

        n = len(cls_groups)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # ensure total <= n
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_groups.extend(cls_groups[:n_train].tolist())
        val_groups.extend(cls_groups[n_train:n_train + n_val].tolist())
        test_groups.extend(cls_groups[n_train + n_val:].tolist())

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()

    return train_df, val_df, test_df


def build_splits(
    index_csv: str | Path,
    *,
    out_dir: str | Path,
    cfg: SplitConfig | None = None,
) -> Dict[str, Path]:
    """
    Loads the index CSV, adds group_id, performs group-stratified split, and writes:
      - train.csv, val.csv, test.csv
      - classes.json (class_name -> class_id)
      - split_info.json (ratios, seed, counts)
    Returns dict with paths.
    """
    cfg = cfg or SplitConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_index(index_csv)
    df = add_group_id(df)

    tr, vr, te = _normalize_ratios(cfg)

    train_df, val_df, test_df = _group_stratified_split(
        df,
        stratify_col=cfg.stratify_col,
        group_col=cfg.group_col,
        seed=cfg.seed,
        train_ratio=tr,
        val_ratio=vr,
        test_ratio=te,
    )

    # Create class mapping from ALL (stable), but you can also do from train only.
    classes = sorted(df[cfg.stratify_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Write splits
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    classes_path = out_dir / "classes.json"
    classes_path.write_text(json.dumps(class_to_idx, indent=2), encoding="utf-8")

    info = {
        "seed": cfg.seed,
        "ratios": {"train": tr, "val": vr, "test": te},
        "counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df), "total": len(df)},
        "num_classes": len(classes),
    }
    info_path = out_dir / "split_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "classes": classes_path,
        "info": info_path,
    }


def build_default_splits_for_spectrograms() -> Dict[str, Path]:
    CONFIG = _get_config()
    data_dir_cfg = Path(CONFIG.paths.data_dir)
    project_root = Path(__file__).resolve().parents[3]
    data_dir_candidates = [
        (project_root / data_dir_cfg).resolve(),
        (Path.cwd() / data_dir_cfg).resolve(),
        (project_root / "src" / data_dir_cfg).resolve(),
    ]
    data_dir = next((p for p in data_dir_candidates if p.exists()), data_dir_candidates[0])
    spec_root = data_dir / "spectrograms"
    index_csv = spec_root / "index_train_64x128.csv"
    out_dir = spec_root / "splits"
    return build_splits(index_csv, out_dir=out_dir)
