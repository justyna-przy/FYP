from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv
from collections import Counter


Row = Dict[str, str]


def read_manifest_csv(path: Path | str) -> Tuple[List[Row], List[str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_manifest_csv(path: Path | str, rows: List[Row], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _default_fieldnames(rows: List[Row]) -> List[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def _quality_key(row: Row) -> str:
    return (row.get("quality") or "").strip().upper()


def select_rows_for_species(
    rows: List[Row],
    *,
    target_n: int = 400,
    prefer_countries: Tuple[str, ...] = ("Ireland", "United Kingdom"),
    prefer_qualities: Tuple[str, ...] = ("A", "B", "C"),
    recordist_cap: int = 50,
) -> List[Row]:
    prefer_set = set(prefer_countries)
    quality_order: List[str] = []
    for quality in prefer_qualities:
        quality_key = quality.strip().upper()
        if quality_key and quality_key not in quality_order:
            quality_order.append(quality_key)

    selected: List[Row] = []
    recordist_counts: Counter[str] = Counter()
    capped_target = min(target_n, len(rows))

    # Tiered pass:
    # A+IE/UK, A+other, B+IE/UK, B+other, C+IE/UK, C+other, ...
    for quality in quality_order:
        for prefer_country in (True, False):
            for row in rows:
                if len(selected) >= capped_target:
                    return selected

                row_quality = _quality_key(row)
                if row_quality != quality:
                    continue

                country = (row.get("country") or "").strip()
                country_is_preferred = country in prefer_set
                if country_is_preferred != prefer_country:
                    continue

                recordist = (row.get("recordist") or "").strip()
                if recordist and recordist_counts[recordist] >= recordist_cap:
                    continue

                selected.append(row)
                if recordist:
                    recordist_counts[recordist] += 1

    return selected


def write_selected_manifests(
    manifest_dir: Path | str,
    *,
    target_per_species: int = 300,
    recordist_cap: int = 50,
    prefer_countries: Tuple[str, ...] = ("Ireland", "United Kingdom"),
    prefer_qualities: Tuple[str, ...] = ("A", "B", "C"),
    selected_suffix: str = "selected",
) -> List[Path]:
    manifest_dir = Path(manifest_dir)
    files = sorted(manifest_dir.glob("*.csv"))
    written_paths: List[Path] = []

    for csv_path in files:
        stem = csv_path.stem
        if stem.endswith((f"_{selected_suffix}", "_downloaded", "_clips")):
            continue
        if stem.startswith("species_selection_summary"):
            continue

        rows, fieldnames = read_manifest_csv(csv_path)
        if not rows:
            continue
        if not fieldnames:
            fieldnames = _default_fieldnames(rows)

        selected_rows = select_rows_for_species(
            rows,
            target_n=target_per_species,
            prefer_countries=prefer_countries,
            prefer_qualities=prefer_qualities,
            recordist_cap=recordist_cap,
        )

        out_path = csv_path.with_name(f"{csv_path.stem}_{selected_suffix}.csv")
        write_manifest_csv(out_path, selected_rows, fieldnames)
        written_paths.append(out_path)

    return written_paths
