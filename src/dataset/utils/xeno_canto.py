from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable
import csv
import json
import os
import time

from dotenv import load_dotenv
import requests

from src.config import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

Recording = Dict[str, Any]
ManifestEntry = Dict[str, Any]

MANIFEST_COLUMNS: List[str] = [
    "xc_id",
    "sci_name",
    "common_name",
    "recordist",
    "country",
    "date",
    "time",
    "length_s",
    "quality",
    "type",
    "method",
    "animal_seen",
    "also",
    "sampling_rate",
    "xc_url",
    "file_url",
    "local_path",
]


def _get_with_retry(url: str, params: Dict[str, Any], max_retries: int = 3, timeout_s: int = 30,) -> Dict[str, Any]:
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Xeno-canto API error: {data['error']}")
            return data
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise last_exception or RuntimeError("Unknown Xeno-canto API error.")


def _get_api_key() -> Optional[str]:
    key = os.environ.get("XENO_CANTO_API_KEY")
    if key is None:
        return None
    key = key.strip()
    return key or None


def fetch_recordings(query: str, per_page: int = 500,) -> List[Recording]:
    """Fetch all recordings matching a Xeno-canto query"""
    
    query = query.replace("+", " ")
    api_key = _get_api_key()

    all_recordings: List[Recording] = []
    page = 1

    while True:
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
        }
        if api_key:
            params["key"] = api_key

        try:
            data = _get_with_retry(CONFIG.xeno_canto.base_url, params)
        except requests.HTTPError as exc:
            # If an API key is invalid, the endpoint can return 401.
            # Retry once without a key so notebook runs still work.
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 401 and "key" in params:
                params.pop("key", None)
                data = _get_with_retry(CONFIG.xeno_canto.base_url, params)
            else:
                raise

        recs = data.get("recordings", []) or []
        all_recordings.extend(recs)

        num_pages = int(data.get("numPages", 1) or 1)
        if page >= num_pages:
            break
        page += 1

    return all_recordings


def slugify_species(sci_name: str) -> str:
    return sci_name.lower().replace(" ", "_")


def manifest_path(manifest_dir: Union[Path, str], sci_name: str, suffix: str = "") -> Path:
    manifest_dir = Path(manifest_dir)
    slug = slugify_species(sci_name)
    return manifest_dir / (f"{slug}_{suffix}.csv" if suffix else f"{slug}.csv")


def _length_to_seconds(length_str: Optional[str]) -> Optional[int]:
    if not length_str or ":" not in length_str:
        return None
    parts = length_str.split(":")
    try:
        parts_int = list(map(int, parts))
    except ValueError:
        return None

    if len(parts_int) == 2:
        minutes, seconds = parts_int
        return minutes * 60 + seconds
    if len(parts_int) == 3:
        hours, minutes, seconds = parts_int
        return hours * 3600 + minutes * 60 + seconds
    return None


def recording_to_manifest_entry(recording: Recording, *, local_path: str = "") -> ManifestEntry:
    """Convert one Xeno Canto recording dict into a manifest row."""
    
    # Creating the scientific name
    gen = recording.get("gen")
    sp = recording.get("sp")
    sci_name = f"{gen} {sp}".strip() if gen and sp else ""

    length_str = recording.get("length")
    length_s = _length_to_seconds(length_str) if isinstance(length_str, str) else None

    xc_id = str(recording.get("id") or "").strip()
    if not xc_id:
        raise ValueError("Recording missing id.")

    file_url = str(recording.get("file") or "").strip()
    if not file_url:
        raise ValueError(f"Recording {xc_id} missing file URL.")

    xc_url = f"https://xeno-canto.org/{xc_id}"

    return {
        "xc_id": xc_id,
        "sci_name": sci_name,
        "common_name": recording.get("en"),
        "recordist": recording.get("rec"),
        "country": recording.get("cnt"),
        "date": recording.get("date"),
        "time": recording.get("time"),
        "length_s": length_s,
        "quality": recording.get("q"),
        "type": recording.get("type"),
        "method": recording.get("method"),
        "animal_seen": recording.get("animal-seen"),
        "also": json.dumps(recording.get("also", [])),
        "sampling_rate": recording.get("smp"),
        "xc_url": xc_url,
        "file_url": file_url,
        "local_path": local_path,
    }


def write_manifest_csv(entries: List[ManifestEntry], path: Union[Path, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(entries)


def read_manifest_csv(path: Union[Path, str]) -> List[ManifestEntry]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]  


def filter_recordings_to_config(recordings: List[Recording]) -> List[Recording]:
    xc = CONFIG.xeno_canto
    allowed_countries = set(xc.countries)

    out: List[Recording] = []
    for r in recordings:
        country = (r.get("cnt") or "").strip()

        if country and country not in allowed_countries:
            continue

        out.append(r)

    return out


def write_raw_manifest_for_species(sci_name: str, query: str, manifest_dir: Union[Path, str], per_page: int = 500,) -> Path:
    """Fetch metadata and write raw manifest file for each species"""
    
    recs = fetch_recordings(query, per_page=per_page)
    recs = filter_recordings_to_config(recs)
    
    entries = [recording_to_manifest_entry(r, local_path="") for r in recs]
    out_path = manifest_path(manifest_dir, sci_name)  # no suffix = raw
    write_manifest_csv(entries, out_path)
    return out_path


def download_manifest_entry(entry: ManifestEntry, output_dir: Union[Path, str], overwrite: bool = True, timeout_s: int = 60,) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xc_id = str(entry.get("xc_id") or "").strip()
    file_url = str(entry.get("file_url") or "").strip()
    dest_path = output_dir / f"XC{xc_id}.mp3"

    if dest_path.exists() and not overwrite:
        entry["local_path"] = str(dest_path)
        return dest_path

    resp = requests.get(file_url, stream=True, timeout=timeout_s)
    resp.raise_for_status()
    with dest_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    entry["local_path"] = str(dest_path)
    return dest_path


def download_from_selected_manifest(sci_name: str, manifest_dir: Union[Path, str], raw_audio_dir: Union[Path, str], overwrite: bool = True) -> Path:
    """Reads manifests/<species>_selected.csv and download into raw/<species>/"""
    
    manifest_dir = Path(manifest_dir)
    raw_audio_dir = Path(raw_audio_dir)

    selected_csv = manifest_path(manifest_dir, sci_name, "selected")
    entries = read_manifest_csv(selected_csv)

    species_out = raw_audio_dir / slugify_species(sci_name)
    for i, e in enumerate(entries, start=1):
        print(i, "/", len(entries))
        download_manifest_entry(e, species_out, overwrite=overwrite)

    downloaded_csv = manifest_path(manifest_dir, sci_name, "downloaded")
    write_manifest_csv(entries, downloaded_csv)
    return downloaded_csv
