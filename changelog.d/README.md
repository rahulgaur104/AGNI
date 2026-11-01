# Changelog entries

A pull request adds its `CHANGELOG.md` entry here as its own file, not in
`CHANGELOG.md`: PRs that all add a line to the same section of one file conflict
with each other every time one of them merges, and new files never do.

Name the file `<PR number>.<kind>.md`, with `<kind>` one of

| kind | section |
|---|---|
| `feature` | New Features |
| `breaking` | Breaking Changes and Deprecations |
| `bugfix` | Bug Fixes |

and write the entry as it should read, without the leading `-` and without the
PR link, e.g. `changelog.d/124.bugfix.md`:

```
``EquilibriumData.validate`` rejects a mirror (``iota`` zero everywhere), which
used to pass and give an all-NaN ``A``.
```

`python tools/changelog.py` moves every entry into `CHANGELOG.md`, at the top of
its section above the latest version with the PR link, and deletes the files
(`--dry-run` prints the result instead).
