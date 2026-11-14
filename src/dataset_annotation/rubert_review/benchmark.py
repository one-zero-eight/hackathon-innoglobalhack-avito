"""Время применения сохранённых train-review решений и шесть примеров разметки."""

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_annotation.rubert_review.run import load_reviewed_rows

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "out/rubert-thresholds/train.jsonl"
DECISIONS = HERE / "decisions.json"
REPORT = HERE / "results.json"
EXAMPLE_INDICES = (23, 2215, 107, 2485, 2287, 2747)
logger = logging.getLogger(__name__)


def main() -> None:
    """Проверить происхождение данных и измерить применение review, без обучения."""
    raw = SOURCE.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    rows = [
        {
            "description": row["description"],
            "targets": {item["mcId"] for item in row["found_microcategories"]},
        }
        for row in records
    ]
    start = perf_counter()
    reviewed, weights, summary = load_reviewed_rows(rows, DECISIONS)
    elapsed = perf_counter() - start
    decisions = {entry["train_index"]: entry for entry in json.loads(DECISIONS.read_text())["entries"]}
    examples = []
    for index in EXAMPLE_INDICES:
        entry = decisions[index]
        original = sorted(rows[index]["targets"])
        output = sorted(reviewed[index]["targets"])
        if original != entry["original_targets"] or output != entry["reviewed_targets"]:
            raise ValueError(f"Решение для строки {index} не совпадает с сохранённым review")
        examples.append({
            "train_index": index,
            "input": {"description": rows[index]["description"], "mcId": 101},
            "output": {"mcIds": output},
            "decision": entry["decision"],
            "sample_weight": float(weights[index]),
            "description_sha256": hashlib.sha256(rows[index]["description"].encode()).hexdigest(),
        })
    result = {
        "status": "complete", "date_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "review_sha256": hashlib.sha256(DECISIONS.read_bytes()).hexdigest(),
        "operation": "load_reviewed_rows: apply existing decisions to 3196 train rows",
        "timing": {"calls": 1, "seconds": elapsed, "seconds_per_row": elapsed / len(rows)},
        "summary": summary, "examples": examples,
        "note": "Existing provisional labels, not fresh annotation or independent gold.",
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Review: %d строк, %.6f с, %d примеров", len(rows), elapsed, len(examples))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
