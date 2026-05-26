import json
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """Load all summary.json files from the results directory."""
    data = {}
    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir() or condition_dir.name.startswith("."):
            continue
        condition_name = condition_dir.name
        data[condition_name] = {}
        for attack_dir in sorted(condition_dir.iterdir()):
            if not attack_dir.is_dir():
                continue
            summary_path = attack_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    data[condition_name][attack_dir.name] = json.load(f)
    return data


def _compute_rates(results: dict) -> tuple[float, float]:
    """Compute utility and security pass rates from a summary dict."""
    utility = results.get("utility_results", {})
    security = results.get("security_results", {})

    n_utility = sum(1 for v in utility.values() if v)
    n_security = sum(1 for v in security.values() if v)
    total = max(len(utility), 1)

    return n_utility / total, n_security / total


def generate_report(results_dir: Path) -> str:
    """Generate a markdown comparison report from benchmark results."""
    data = load_results(results_dir)

    if not data:
        return "No results found."

    conditions = sorted(data.keys())
    all_attacks = set()
    for cond in conditions:
        all_attacks.update(data[cond].keys())
    attacks = sorted(all_attacks)

    lines = []
    lines.append("# NeMoGuard-ADBenchmark Comparison Report\n")

    # Summary table
    lines.append("## Summary\n")
    header = "| Metric |"
    separator = "|--------|"
    for cond in conditions:
        header += f" {cond} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    # Compute overall averages
    avg_utility = {}
    avg_security = {}
    for cond in conditions:
        utilities = []
        securities = []
        for attack in attacks:
            if attack in data[cond]:
                u, s = _compute_rates(data[cond][attack])
                utilities.append(u)
                securities.append(s)
        avg_utility[cond] = sum(utilities) / max(len(utilities), 1)
        avg_security[cond] = sum(securities) / max(len(securities), 1)

    row_u = "| Utility (avg) |"
    row_s = "| Security (avg) |"
    for cond in conditions:
        row_u += f" {avg_utility[cond]:.0%} |"
        row_s += f" {avg_security[cond]:.0%} |"
    lines.append(row_u)
    lines.append(row_s)

    # Delta from baseline
    if "baseline" in conditions:
        lines.append("")
        baseline_u = avg_utility.get("baseline", 0)
        baseline_s = avg_security.get("baseline", 0)
        row_du = "| Utility delta |"
        row_ds = "| Security delta |"
        for cond in conditions:
            if cond == "baseline":
                row_du += " -- |"
                row_ds += " -- |"
            else:
                du = avg_utility[cond] - baseline_u
                ds = avg_security[cond] - baseline_s
                row_du += f" {du:+.0%} |"
                row_ds += f" {ds:+.0%} |"
        lines.append(row_du)
        lines.append(row_ds)

    # Per-attack breakdown
    lines.append("\n## By Attack Type\n")
    header = "| Attack |"
    separator = "|--------|"
    for cond in conditions:
        header += f" {cond} (U/S) |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    for attack in attacks:
        row = f"| {attack} |"
        for cond in conditions:
            if attack in data[cond]:
                u, s = _compute_rates(data[cond][attack])
                row += f" {u:.0%} / {s:.0%} |"
            else:
                row += " -- |"
        lines.append(row)

    return "\n".join(lines)


def main():
    results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print("No results directory found. Run runner.py first.")
        return

    report = generate_report(results_dir)
    output_path = results_dir / "comparison_report.md"
    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
