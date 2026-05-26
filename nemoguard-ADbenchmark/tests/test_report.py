import json
import tempfile
from pathlib import Path

from report import load_results, generate_report


def _write_summary(path: Path, condition: str, attack: str, utility: dict, security: dict):
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "condition": condition,
        "attack": attack,
        "elapsed_seconds": 10.0,
        "utility_results": utility,
        "security_results": security,
    }
    with open(path / "summary.json", "w") as f:
        json.dump(summary, f)


def test_load_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        _write_summary(
            results_dir / "baseline" / "direct",
            "baseline", "direct",
            {"t0_i0": True, "t0_i1": False},
            {"t0_i0": False, "t0_i1": False},
        )

        data = load_results(results_dir)
        assert "baseline" in data
        assert "direct" in data["baseline"]
        assert data["baseline"]["direct"]["utility_results"]["t0_i0"] is True


def test_generate_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        for condition in ["baseline", "execution_rails"]:
            for attack in ["direct", "dos"]:
                _write_summary(
                    results_dir / condition / attack,
                    condition, attack,
                    {"t0_i0": True, "t0_i1": False, "t1_i0": True, "t1_i1": True},
                    {"t0_i0": condition != "baseline", "t0_i1": False,
                     "t1_i0": condition != "baseline", "t1_i1": True},
                )

        report = generate_report(results_dir)

        assert "baseline" in report
        assert "execution_rails" in report
        assert "Utility" in report or "utility" in report.lower()
        assert "Security" in report or "security" in report.lower()
