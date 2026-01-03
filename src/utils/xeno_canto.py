from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import csv
import json
import os
import time

from dotenv import load_dotenv
import requests

from ..config import CONFIG

load_dotenv()
XENO_CANTO_API_KEY = os.environ.get("XENO_CANTO_API_KEY")

Recording = Dict[str, Any]
ManifestEntry = Dict[str, Any]

MANIFEST_COLUMNS = [
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


def _get_with_retry(
    url: str,
    params: Dict[str, Any],
    max_retries: int = 3,
) -> Dict[str, Any]:
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Xeno-canto API error: {data['error']}")
            return data
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    if last_exception is None:
        raise RuntimeError("Unknown Xeno-canto API error.")
    raise last_exception


def get_recordings_data_for_species(
    query: str,
    per_page: int = 500,
) -> List[Recording]:
    if ":" not in query:
        query = f'sp:"{query}"'
    query = query.replace("+", " ")

    all_recordings: List[Recording] = []
    page = 1

    while True:
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "key": XENO_CANTO_API_KEY,
        }

        data = _get_with_retry(CONFIG.xeno_canto.base_url, params)

        recordings = data.get("recordings", []) or []
        all_recordings.extend(recordings)

        num_pages = int(data.get("numPages", 1) or 1)
        if page >= num_pages:
            break

        page += 1

    return all_recordings


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


def _manifest_entry(recording: Recording, local_path: Path, file_url: str) -> ManifestEntry:
    gen = recording.get("gen")
    sp = recording.get("sp")
    sci_name = f"{gen} {sp}".strip() if gen and sp else None
    length_str = recording.get("length")
    length_s = _length_to_seconds(length_str) if isinstance(length_str, str) else None
    xc_id = str(recording.get("id") or "").strip()
    xc_url = f"https://xeno-canto.org/{xc_id}" if xc_id else ""

    sampling_rate = recording.get("smp")
    if isinstance(sampling_rate, str) and sampling_rate.isdigit():
        sampling_rate = int(sampling_rate)

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
        "sampling_rate": sampling_rate,
        "xc_url": xc_url,
        "file_url": file_url,
        "local_path": str(local_path),
    }


def _write_manifest_csv(entries: List[ManifestEntry], manifest_path: Union[Path, str]) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        if entries:
            writer.writerows(entries)


def download_recording(
    recording: Recording,
    output_dir: Union[Path, str],
    *,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xc_id = str(recording.get("id") or "").strip()
    if not xc_id:
        raise ValueError("Recording missing id.")

    file_url = str(recording.get("file") or "").strip()
    if not file_url:
        raise ValueError("Recording missing file URL.")
    if file_url.startswith("//"):
        file_url = f"https:{file_url}"
    elif file_url.startswith("/"):
        file_url = f"https://xeno-canto.org{file_url}"

    file_name = recording.get("file-name")
    ext = Path(str(file_name)).suffix if file_name else ""
    if not ext:
        ext = ".mp3"

    dest_path = output_dir / f"XC{xc_id}{ext.lower()}"
    if dest_path.exists() and not overwrite:
        return dest_path

    resp = requests.get(file_url, stream=True)
    resp.raise_for_status()
    with dest_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return dest_path


def download_recordings(
    recordings: List[Recording],
    output_dir: Union[Path, str],
    *,
    overwrite: bool = False,
    manifest_csv_path: Optional[Union[Path, str]] = None,
) -> List[ManifestEntry]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: List[ManifestEntry] = []
    for recording in recordings:
        local_path = download_recording(recording, output_dir, overwrite=overwrite)
        file_url = str(recording.get("file") or "").strip()
        if file_url.startswith("//"):
            file_url = f"https:{file_url}"
        elif file_url.startswith("/"):
            file_url = f"https://xeno-canto.org{file_url}"
        entries.append(_manifest_entry(recording, local_path, file_url))

    if manifest_csv_path:
        _write_manifest_csv(entries, manifest_csv_path)

    return entries
