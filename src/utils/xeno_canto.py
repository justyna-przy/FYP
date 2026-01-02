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


def _manifest_entry(recording: Recording, local_path: Path, file_url: str) -> ManifestEntry:
    gen = recording.get("gen")
    sp = recording.get("sp")
    sci_name = f"{gen} {sp}".strip() if gen and sp else None

    return {
        "xc_id": str(recording.get("id") or ""),
        "sci_name": sci_name,
        "common_name": recording.get("en"),
        "country": recording.get("cnt"),
        "quality": recording.get("q"),
        "length": recording.get("length"),
        "recording_type": recording.get("type"),
        "license_url": recording.get("lic"),
        "file_url": file_url,
        "local_path": str(local_path),
    }


def _write_manifest_json(entries: List[ManifestEntry], manifest_path: Union[Path, str]) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _write_manifest_csv(entries: List[ManifestEntry], manifest_path: Union[Path, str]) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        manifest_path.write_text("", encoding="utf-8")
        return
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        writer.writeheader()
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
    manifest_json_path: Optional[Union[Path, str]] = None,
    manifest_csv_path: Optional[Union[Path, str]] = None,
) -> List[ManifestEntry]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: List[ManifestEntry] = []
    for recording in recordings:
        local_path = download_recording(recording, output_dir, overwrite=overwrite)
        file_url = str(recording.get("file") or "")
        if file_url.startswith("//"):
            file_url = f"https:{file_url}"
        elif file_url.startswith("/"):
            file_url = f"https://xeno-canto.org{file_url}"
        entries.append(_manifest_entry(recording, local_path, file_url))

    if manifest_json_path:
        _write_manifest_json(entries, manifest_json_path)
    if manifest_csv_path:
        _write_manifest_csv(entries, manifest_csv_path)

    return entries
