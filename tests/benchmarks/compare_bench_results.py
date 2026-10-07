"""Compare two pytest-benchmark JSON files and write the table of a PR comment.

usage: python compare_bench_results.py BASE.json NEW.json [OUT.md]

A benchmark is called slower or faster when the difference of the medians is
larger than ``SIGNIFICANCE`` times the combined interquartile ranges (the
rule of DESC's comparison); otherwise it is marked as unchanged. CI timings are
noisy, so the table is a hint for the reviewer, not a gate.
"""

import json
import sys

import numpy as np

SIGNIFICANCE = 3.0


def load(path):
    """``{name: (median, iqr)}`` in seconds from a pytest-benchmark JSON file."""
    with open(path) as f:
        data = json.load(f)
    return {
        b["name"]: (b["stats"]["median"], b["stats"]["iqr"]) for b in data["benchmarks"]
    }


def table(base, new):
    """The markdown (diff-highlighted) comparison of the benchmarks both ran."""
    names = [n for n in new if n in base]
    head = (
        f"| {'benchmark':<38} | {'dt (%)':^18} | {'new (s)':^10} | {'base (s)':^10} |"
    )
    lines = ["```diff", head, f"| {'-' * 38} | {'-' * 18} | {'-' * 10} | {'-' * 10} |"]
    for name in names:
        (t0, s0), (t1, s1) = base[name], new[name]
        dt, ds = t1 - t0, np.hypot(s0, s1)
        mark = " "
        if abs(dt) > SIGNIFICANCE * ds:
            mark = "-" if dt > 0 else "+"  # red: slower, green: faster
        pct = 100 * dt / t0
        change = f"{pct:+8.2f} +/- {100 * ds / t0:5.2f}"
        lines.append(f"{mark}{name:<39}| {change} | {t1:10.3e} | {t0:10.3e} |")
    only = sorted(set(new) ^ set(base))
    lines.append("```")
    if only:
        lines.append(f"\nNot in both runs: {', '.join(only)}.")
    lines.append(
        "\nGitHub runners are noisy; read a change with that in mind. "
        "Times are medians of 3 rounds after one warm-up."
    )
    return "\n".join(lines)


def main(argv):
    """Print the table and write it where the workflow reads it."""
    text = table(load(argv[1]), load(argv[2]))
    print(text)
    with open(argv[3] if len(argv) > 3 else "commit_msg.txt", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main(sys.argv)
