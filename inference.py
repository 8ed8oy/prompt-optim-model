#!/usr/bin/env python
"""Wrapper — delegates to scripts/inference.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.inference import main
main()
