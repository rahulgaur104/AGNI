#!/usr/bin/env python3
"""Fold the entries in ``changelog.d/`` into ``CHANGELOG.md``.

A pull request does not edit ``CHANGELOG.md``: every PR adding a line at the
top of the same section conflicts with every other one as soon as one of them
merges. It adds one file instead, ``changelog.d/<PR number>.<kind>.md``, with
``<kind>`` one of ``feature``, ``breaking`` or ``bugfix``, holding the entry
text without the leading ``-`` and without the PR link. New files never
conflict.

This script puts each entry at the top of its section above the latest
version, newest PR first, adds the PR link, and deletes the file.

Usage
-----
    python tools/changelog.py            # fold and delete the files
    python tools/changelog.py --dry-run  # print the new CHANGELOG.md only
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = {
    "feature": "New Features",
    "breaking": "Breaking Changes and Deprecations",
    "bugfix": "Bug Fixes",
}
URL = "https://github.com/rahulgaur104/AGNI/pull/{}"
NAME = re.compile(r"^(\d+)\.(" + "|".join(SECTIONS) + r")\.md$")


def fragments(folder):
    """``(pr, kind, text, path)`` of every entry file, newest PR first."""
    out = []
    for path in sorted(folder.glob("*.md")):
        match = NAME.match(path.name)
        if match is None:
            continue  # README.md
        text = " ".join(path.read_text().split()).removeprefix("- ").rstrip(".")
        out.append((int(match[1]), match[2], text, path))
    return sorted(out, key=lambda f: -f[0])


def fold(lines, entries):
    """``lines`` of CHANGELOG.md with ``entries`` at the top of their sections."""
    end = next(
        (i for i, line in enumerate(lines) if re.match(r"^v\d", line)), len(lines)
    )
    head, tail = lines[:end], lines[end:]
    order = list(SECTIONS.values())
    for kind, section in SECTIONS.items():
        new = [f"- {t} [#{n}]({URL.format(n)})." for n, k, t, _ in entries if k == kind]
        if not new:
            continue
        if section not in head:  # before the next section present, else last
            later = [s for s in order[order.index(section) + 1 :] if s in head]
            if later:
                at = head.index(later[0])
                head[at:at] = [section, "", ""]
            else:
                at = max(i for i, line in enumerate(head) if line) + 1
                head[at:at] = ["", section, ""]
        at = head.index(section) + 2
        head[at:at] = new
    return head + tail


def main():
    """Fold ``changelog.d/`` into ``CHANGELOG.md``, or print it with --dry-run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    entries = fragments(ROOT / "changelog.d")
    if not entries:
        sys.exit("changelog.d/ has no entries")
    changelog = ROOT / "CHANGELOG.md"
    text = "\n".join(fold(changelog.read_text().splitlines(), entries)) + "\n"
    if args.dry_run:
        print(text, end="")
        return
    changelog.write_text(text)
    for *_, path in entries:
        path.unlink()
    print(f"folded {len(entries)} entries into CHANGELOG.md")


if __name__ == "__main__":
    main()
