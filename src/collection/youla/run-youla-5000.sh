#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p out
output=out/youla-5000.jsonl
count() { if [[ -f "$output" ]]; then wc -l < "$output"; else echo 0; fi; }
while (( $(count) < 5000 )); do
  date -u
  echo "Rows before pass: $(count)"
  docker run --rm --name igh-avito-youla-5000 --init --shm-size=768m \
    -e MIN_LENGTH=300 -e TARGET_ROWS=5000 -e MAX_VISITS=60 -e MAX_SCROLLS=20 \
    -e SEED_FILES=/output/youla-300.jsonl,/output/youla-300.jsonl.cache \
    -v "$PWD/src/collection/youla:/collector:ro" -v "$PWD/out:/output" \
    mcr.microsoft.com/playwright:v1.56.1-noble bash -lc \
    'cd /tmp && npm install --no-audit --no-fund playwright@1.56.1 >&2 && NODE_PATH=/tmp/node_modules node /collector/collect-youla.cjs "$@"' \
    -- /output/youla-5000.jsonl moskva sankt-peterburg kazan krasnodar novosibirsk ekaterinburg samara ufa chelyabinsk tyumen perm voronezh rostov-na-donu volgograd omsk krasnoyarsk irkutsk saratov
  echo "Rows after pass: $(count)"
  if (( $(count) < 5000 )); then sleep 1800; fi
done
echo "Reached $(count) rows"
