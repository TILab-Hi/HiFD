"""End-to-end smoke test: run compute_hifd against synthetic fixtures and
verify the CSV is written with sensible numeric values.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixtures(repo: Path) -> Path:
    return repo / "tests" / "fixtures"


def test_compute_hifd_synthetic_endtoend(tmp_path: Path, repo: Path,
                                         fixtures: Path) -> None:
    pred_dir = fixtures / "predictions"
    results_dir = tmp_path / "results"

    # Trim methods.yaml to MockMethod for this run
    methods = tmp_path / "methods.yaml"
    methods.write_text(
        "original:\n  lmdb: balanced.lmdb\n"
        "methods:\n  - {name: MockMethod, family: test, lmdb: MockMethod.lmdb}\n"
    )

    r = subprocess.run([
        sys.executable, str(repo / "scripts" / "compute_hifd.py"),
        "--predictions-dir", str(pred_dir),
        "--results-dir", str(results_dir),
        "--methods-config", str(methods),
        "--profiles-config", str(repo / "configs" / "profiles.yaml"),
        "--constants-config", str(repo / "configs" / "constants.yaml"),
    ], capture_output=True, text=True)

    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    csv_path = results_dir / "scores.csv"
    assert csv_path.exists()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "MockMethod"
    for col in ("privacy", "U1", "HiFD_Balanced"):
        v = float(row[col]) if row[col] else None
        assert v is not None and 0.0 <= v <= 1.0
