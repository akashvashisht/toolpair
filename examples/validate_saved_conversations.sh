#!/usr/bin/env sh
# Example 3: guard a directory of persisted conversations in CI or a cron job.
# Exits non-zero if any file has pairing issues; writes repaired copies.
set -u
status=0
for f in "${1:-examples}"/*.json; do
  if ! toolpair check "$f" > /dev/null; then
    echo "BROKEN: $f"
    toolpair check "$f" | sed 's/^/    /'
    toolpair repair "$f" -o "${f%.json}.repaired.json" 2>/dev/null
    status=1
  fi
done
exit $status
