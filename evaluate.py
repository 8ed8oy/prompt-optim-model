#!/usr/bin/env python
"""Wrapper — delegates to scripts/evaluate.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.evaluate import main
main()
