"""AutoDock poses are ranked by binding energy, not by LGA run order."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docking"))

from docking.pose_ranking import rank_by_energy  # noqa: E402


def test_parser_orders_poses_by_energy_and_keeps_run_order_for_ties():
    mols, energies = rank_by_energy(
        ["run1", "run2", "run3", "run4"], [-7.9, -8.2, -7.9, -9.0]
    )
    assert energies == [-9.0, -8.2, -7.9, -7.9]
    assert mols == ["run4", "run2", "run1", "run3"]


def test_energy_ranking_import_does_not_require_docking_tools():
    script = (
        "import sys\n"
        "for name in ('biotite', 'openbabel', 'pymol'):\n"
        "    sys.modules[name] = None\n"
        "from docking.pose_ranking import rank_by_energy\n"
        "assert rank_by_energy(['first', 'best'], [-1.0, -2.0])[0] == ['best', 'first']\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
