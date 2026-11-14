#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
target=${1:-500}
shift "$(( $# > 0 ))"
output=out/yandex/profiles.jsonl
checkpoint=out/yandex/profiles.pages.json
while true; do
  before=
  if [[ -f "$checkpoint" ]]; then before=$(cat "$checkpoint"); fi
  python3 src/collection/yandex/collect.py --output "$output" --pages 3 --target "$target" "$@"
  count=$(wc -l < "$output")
  if (( count >= target )); then exit 0; fi
  if [[ -f "$checkpoint" ]] && [[ $(cat "$checkpoint") != "$before" ]]; then continue; fi
  printf 'Catalogs exhausted. Collected %s of %s profiles.\n' "$count" "$target" >&2
  exit 1
done
