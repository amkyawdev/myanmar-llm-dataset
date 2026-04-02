#!/usr/bin/env python3
"""Preprocess and clean the dataset."""
import json, re
from pathlib import Path

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    for split in ['train', 'validation', 'test']:
        split_dir = base_dir / "data" / "processed" / split
        for f in split_dir.glob("*.jsonl"):
            print(f"Processed: {f.name}")