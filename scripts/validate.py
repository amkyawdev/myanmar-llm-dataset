#!/usr/bin/env python3
"""Validate dataset format and content."""
import json, sys
from pathlib import Path

def validate_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                json.loads(line)
    return True

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    for split in ['train', 'validation', 'test']:
        for f in (base_dir / "data" / "processed" / split).glob("*.jsonl"):
            if validate_jsonl(f):
                print(f"OK: {f.name}")
    sys.exit(0)