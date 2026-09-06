"""Pytest fixtures for mobile-bench script tests.

Adds the scripts dir (scripts/analyze/mobilebench) to sys.path so tests can
import the vendored mobile-bench scripts directly.
"""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
