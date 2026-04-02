#!/usr/bin/env python3
"""Download dataset from source URLs."""
import requests
from pathlib import Path

def download_file(url: str, dest: Path):
    """Download a file from URL to destination."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {dest}")

if __name__ == "__main__":
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_dir.mkdir(exist_ok=True)
    print("Download complete!")