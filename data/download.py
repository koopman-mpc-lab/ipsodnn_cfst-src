from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from src.constants import MENDELEY_DATASET_ID, MENDELEY_REFERER

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def _download_url(url: str, destination: Path, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Referer": MENDELEY_REFERER},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        destination.write_bytes(response.read())


def download_cabrera_dataset(
    manifest_path: Path,
    output_dir: Path,
    skip_existing: bool = True,
) -> list[Path]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    downloaded: list[Path] = []
    for entry in manifest["files"]:
        destination = output_dir / entry["name"]
        if skip_existing and destination.exists() and destination.stat().st_size > 0:
            downloaded.append(destination)
            continue
        url = (
            f"https://data.mendeley.com/public-files/datasets/"
            f"{MENDELEY_DATASET_ID}/files/{entry['file_id']}/file_downloaded"
        )
        try:
            _download_url(url, destination)
            downloaded.append(destination)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Failed to download {entry['name']}: {exc}") from exc
    return downloaded


def download_cabrera_zip(output_path: Path) -> Path:
    url = f"https://data.mendeley.com/public-api/zip/{MENDELEY_DATASET_ID}/download/1"
    _download_url(url, output_path, timeout=600)
    return output_path
