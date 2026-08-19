#!/usr/bin/env bash
# agnimhd must never import DESC -- not in the package, not in a test, not
# lazily inside a function. The dependency runs the other way: DESC installs
# agnimhd. A lazy `import desc` inside a helper would satisfy a casual reading
# of the source while making the package unusable without DESC present, so this
# greps rather than trusting the import graph.
#
# tools/ and examples/ are NOT checked. They are allowed to import DESC: the
# fixture exporter and the reference adapter both need it, and neither is part
# of the package nor imported by it.
#
# Run by pre-commit; also runnable by hand:  bash tools/check_no_desc.sh
set -uo pipefail

hits=$(grep -rnE '^[[:space:]]*(import|from)[[:space:]]+desc([.[:space:]]|$)' src tests || true)

if [ -n "$hits" ]; then
    echo "agnimhd (or its test suite) imports DESC:"
    echo "$hits"
    echo
    echo "The dependency direction is the point of this package: DESC installs"
    echo "agnimhd, never the reverse. Adapters and export scripts belong in"
    echo "tools/ or examples/, which are excluded from this check."
    exit 1
fi
exit 0
