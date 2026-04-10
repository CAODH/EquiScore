#!/usr/bin/env python3
"""Compatibility wrapper for utils/generate_manifest.py."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.generate_manifest import main


if __name__ == "__main__":
    main()
