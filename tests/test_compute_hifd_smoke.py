"""Smoke test: verify compute_hifd.py --help exits cleanly."""

import subprocess
import sys
from pathlib import Path


def test_compute_hifd_help_runs():
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "compute_hifd.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "predictions-dir" in r.stdout
