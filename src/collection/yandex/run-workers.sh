#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
target=${1:-500}
[[ "$target" =~ ^[1-9][0-9]*$ ]] || exit 2
shift "$(( $# > 0 ))"
if (( $# == 0 )); then
  mapfile -t regions < <(cd src/collection/yandex && python3 -c 'from collect import REGIONS; print(*REGIONS, sep=chr(10))')
  set -- "${regions[@]}"
fi
root=out/yandex
mkdir -p "$root/workers"
exec flock -n "$root/workers.lock" bash -s -- "$target" "$root" "$@" <<'RUN'
set -euo pipefail
target=$1
root=$2
shift 2
collector=src/collection/yandex/collect.py
while true; do
  before=$(find "$root/workers" -name '*.pages.json' -exec sha256sum {} + | sort)
  pids=()
  failed=0
  for region in "$@"; do
    [[ "$region" =~ ^[0-9]+-[a-z-]+$ ]] || exit 2
    base="https://uslugi.yandex.ru/$region/category"
    python3 "$collector" --output "$root/workers/$region.jsonl" --pages 3 --target "$target" \
      "$base/remont-i-stroitelstvo--1344" \
      "$base/remont-i-stroitelstvo/remont-kvartir-i-domov--1816" \
      "$base/remont-i-stroitelstvo/elektromontazhnyie-rabotyi--2007" \
      "$base/remont-i-stroitelstvo/santehnicheskie-rabotyi-i-otoplenie--1844" \
      >>"$root/workers/$region.log" 2>&1 &
    pids+=("$!")
    if (( ${#pids[@]} == 4 )); then
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  count=$(python3 - "$root" "$target" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = int(sys.argv[2])
output = root / "profiles.jsonl"
temporary = output.with_suffix(".tmp")
seen_ids = set()
seen_text = set()
with temporary.open("w", encoding="utf-8") as stream:
    for path in [output, *sorted((root / "workers").glob("*.cache.jsonl"))]:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row["platform"] != "yandex":
                    raise ValueError(f"Unexpected platform in {path}")
                description = row["description"]
                key = " ".join(description.casefold().split())
                provider = row["provider_id"]
                if len(seen_ids) >= target or len(description) < 300 or provider in seen_ids or key in seen_text:
                    continue
                seen_ids.add(provider)
                seen_text.add(key)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
temporary.replace(output)
print(len(seen_ids))
PY
)
  echo "Saved $count of $target profiles"
  if (( count >= target )); then exit 0; fi
  if (( failed )); then echo 'A worker failed. See workers/*.log.' >&2; exit 1; fi
  after=$(find "$root/workers" -name '*.pages.json' -exec sha256sum {} + | sort)
  if [[ "$before" == "$after" ]]; then echo 'Catalogs exhausted.' >&2; exit 1; fi
done
RUN
