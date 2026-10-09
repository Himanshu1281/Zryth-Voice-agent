"""Shared pytest setup.

Unit tests (tests/unit) never call an API: CI=1 stops tools.py from creating the
Gemini embedding client at import time, so they run without any keys.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CI", "1")
