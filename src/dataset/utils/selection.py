from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any
import csv
from collections import defaultdict, Counter
from datetime import datetime


Row = Dict[str, str]


def _safe_month_from_date(date_str: str) -> str:
    """
    Return month as '01'..'12'.
    If missing/invalid, return '00'.
    """
    if not date_str:
        return "00"
    date_str = date_str.strip()
    try:
        # expected: YYYY-MM-DD
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.month:02d}"
    except Exception:
        return "00"


def _read_manifest_csv(path: Path) -> List[Row]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_manifest_csv(path: Path, rows: List[Row], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _default_fieldnames(rows: List[Row]) -> List[str]:
    # Prefer preserving original column ordering if possible
    if not rows:
        return []
    return list(rows[0].keys())


def _pick_balanced(
    preferred_by_month: Dict[str, List[Row]],
    other_by_month: Dict[str, List[Row]],
    *,
    target_n: int,
    recordist_cap: int,
) -> List[Row]:
    """
    Month-balanced selection with strong preference for preferred_by_month.
    Greedy round-robin: always pick from the month with the lowest selected count so far.
    """
    selected: List[Row] = []
    selected_month_counts: Counter[str] = Counter()
    recordist_counts: Counter[str] = Counter()

    # All months we can possibly draw from (excluding '00' for balancing, we'll use it later if needed)
    months = sorted(set(preferred_by_month.keys()) | set(other_by_month.keys()))
    months_balanced = [m for m in months if m != "00"]
    months_missing = ["00"] if "00" in months else []

    # For deterministic-ish behaviour, keep original order within each month list (already read order)
    # (you can shuffle later if you want)
    def can_take(row: Row) -> bool:
        rec = (row.get("recordist") or "").strip()
        if not rec:
            # no recordist -> treat as its own bucket; allow it
            return True
        return recordist_counts[rec] < recordist_cap

    def take_from_bucket(bucket: List[Row]) -> Optional[Row]:
        # pop first row that passes recordist cap
        for i, row in enumerate(bucket):
            if can_take(row):
                return bucket.pop(i)
        return None

    def choose_month() -> Optional[str]:
        # pick the month with smallest selected count, but only if something is available there
        candidates: List[Tuple[int, str]] = []
        for m in months_balanced:
            if preferred_by_month.get(m) or other_by_month.get(m):
                candidates.append((selected_month_counts[m], m))
        if not candidates:
            return None
        candidates.sort()  # least used month first
        return candidates[0][1]

    # 1) Balanced selection over real months
    while len(selected) < target_n:
        m = choose_month()
        if m is None:
            break

        row = None
        if preferred_by_month.get(m):
            row = take_from_bucket(preferred_by_month[m])
        if row is None and other_by_month.get(m):
            row = take_from_bucket(other_by_month[m])

        # If we couldn't take from that month due to recordist caps, mark it as "blocked" by emptying buckets
        # (so we don't infinite loop)
        if row is None:
            # no selectable rows in that month
            preferred_by_month[m] = []
            other_by_month[m] = []
            continue

        selected.append(row)
        mm = _safe_month_from_date(row.get("date", ""))
        selected_month_counts[mm] += 1
        rec = (row.get("recordist") or "").strip()
        if rec:
            recordist_counts[rec] += 1

    # 2) If still short, fill from anything left (preferred first), including month '00'
    def flatten_remaining(by_month: Dict[str, List[Row]]) -> List[Row]:
        out: List[Row] = []
        for m in sorted(by_month.keys()):
            out.extend(by_month[m])
        return out

    if len(selected) < target_n:
        remaining_pref = flatten_remaining(preferred_by_month)
        remaining_other = flatten_remaining(other_by_month)

        for pool in (remaining_pref, remaining_other):
            if len(selected) >= target_n:
                break
            i = 0
            while i < len(pool) and len(selected) < target_n:
                row = pool[i]
                if can_take(row):
                    selected.append(row)
                    rec = (row.get("recordist") or "").strip()
                    if rec:
                        recordist_counts[rec] += 1
                    pool.pop(i)
                else:
                    i += 1

    return selected


def select_rows_for_species(
    rows: List[Row],
    *,
    target_n: int = 400,
    prefer_countries: Tuple[str, ...] = ("Ireland", "United Kingdom"),
    recordist_cap: int = 50,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "selected_rows": [...],
        "stats": {...}
      }
    """
    prefer_set = set(prefer_countries)

    # Split into preferred vs other, then by month
    preferred_by_month: Dict[str, List[Row]] = defaultdict(list)
    other_by_month: Dict[str, List[Row]] = defaultdict(list)

    for r in rows:
        country = (r.get("country") or "").strip()
        month = _safe_month_from_date(r.get("date", ""))
        if country in prefer_set:
            preferred_by_month[month].append(r)
        else:
            other_by_month[month].append(r)

    selected = _pick_balanced(
        preferred_by_month=dict(preferred_by_month),
        other_by_month=dict(other_by_month),
        target_n=min(target_n, len(rows)),
        recordist_cap=recordist_cap,
    )

    # stats
    sel_countries = Counter((r.get("country") or "").strip() for r in selected)
    sel_months = Counter(_safe_month_from_date(r.get("date", "")) for r in selected)
    sel_recordists = Counter((r.get("recordist") or "").strip() for r in selected if (r.get("recordist") or "").strip())

    pref_count = sum(sel_countries.get(c, 0) for c in prefer_set)
    stats = {
        "total_in": len(rows),
        "selected": len(selected),
        "prefer_countries": list(prefer_countries),
        "preferred_selected": pref_count,
        "preferred_selected_pct": (pref_count / len(selected) * 100.0) if selected else 0.0,
        "selected_country_top": sel_countries.most_common(10),
        "selected_month_counts": dict(sel_months),
        "selected_month_coverage": len([m for m, c in sel_months.items() if c > 0 and m != "00"]),
        "selected_recordist_top": sel_recordists.most_common(10),
        "max_selected_by_one_recordist": max(sel_recordists.values()) if sel_recordists else 0,
    }

    return {"selected_rows": selected, "stats": stats}


def write_selected_manifests(
    manifest_dir: Path | str,
    *,
    target_per_species: int = 300,
    recordist_cap: int = 50,
    prefer_countries: Tuple[str, ...] = ("Ireland", "United Kingdom"),
    selected_suffix: str = "selected",
) -> Dict[str, Any]:
    """
    Reads each raw manifest *.csv in manifest_dir (excluding already-selected files),
    writes *_selected.csv beside it.

    Returns a summary dict you can print in notebook.
    """
    manifest_dir = Path(manifest_dir)
    files = sorted(manifest_dir.glob("*.csv"))

    summary: Dict[str, Any] = {
        "manifest_dir": str(manifest_dir),
        "params": {
            "target_per_species": target_per_species,
            "recordist_cap": recordist_cap,
            "prefer_countries": list(prefer_countries),
            "selected_suffix": selected_suffix,
        },
        "per_file": {},
    }

    for csv_path in files:
        # skip already-selected files
        if csv_path.stem.endswith(f"_{selected_suffix}"):
            continue

        rows = _read_manifest_csv(csv_path)
        if not rows:
            summary["per_file"][csv_path.name] = {"error": "empty_csv", "raw_file": str(csv_path)}
            continue

        fieldnames = _default_fieldnames(rows)

        result = select_rows_for_species(
            rows,
            target_n=target_per_species,
            prefer_countries=prefer_countries,
            recordist_cap=recordist_cap,
        )
        selected_rows: List[Row] = result["selected_rows"]
        stats: Dict[str, Any] = result["stats"]

        out_path = csv_path.with_name(f"{csv_path.stem}_{selected_suffix}.csv")
        _write_manifest_csv(out_path, selected_rows, fieldnames)

        summary["per_file"][csv_path.name] = {
            "raw_file": str(csv_path),
            "selected_file": str(out_path),
            "stats": stats,
        }

    return summary
