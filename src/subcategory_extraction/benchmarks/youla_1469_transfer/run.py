"""Обучить модели на разметке Youla и оценить на InnoGlobalHack."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from main import MICROCATEGORIES

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAIN = Path(__file__).resolve().parents[1] / "data/youla-1469-decision-1.jsonl"
DEFAULT_REPORT = Path(__file__).with_name("results.json")


def main() -> None:
    """Проверить разметку и запустить общий бенчмарк переноса."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    raw = args.train.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 1469:
        parser.error(f"Ожидалось 1469 записей, получено {len(rows)}")
    if len({row["source"]["url"] for row in rows}) != len(rows):
        parser.error("Повторяются URL объявлений")
    configurations = {json.dumps(row["configuration"], sort_keys=True) for row in rows}
    if len(configurations) != 1:
        parser.error("Разметка сделана разными конфигурациями")
    usable = []
    for row in rows:
        ids = row["found_microcategories"]
        if ids is None and row.get("error") == "prohibited_content":
            continue
        if (not isinstance(ids, list) or any(type(cid) is not int or cid not in MICROCATEGORIES for cid in ids)
                or not isinstance(row["source"]["description"], str) or not row["source"]["description"].strip()):
            parser.error("Недопустимая запись разметки или пустое описание")
        usable.append(row)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        train = Path(directory) / "train.jsonl"
        train.write_text("".join(json.dumps({
            "description": row["source"]["description"],
            "target_mc_ids": [cid for cid in row["found_microcategories"] if cid != 101],
        }, ensure_ascii=False) + "\n" for row in usable), encoding="utf-8")
        report = Path(directory) / "report.json"
        artifact = Path(directory) / "model.json"
        start = perf_counter()
        subprocess.run([sys.executable, "-m", "subcategory_extraction.benchmarks.run_youla_keywords",
                        "--train", str(train), "--report", str(report), "--artifact", str(artifact),
                        "--ensemble"],
                       cwd=ROOT, check=True)
        duration = perf_counter() - start
        result = json.loads(report.read_text(encoding="utf-8"))
    result.update({"source": str(args.train), "source_sha256": hashlib.sha256(raw).hexdigest(),
                   "label_configuration": json.loads(configurations.pop()),
                   "organizer_labels": "original public-train-annotated.jsonl",
                   "excluded_failed_annotations": len(rows) - len(usable),
                   "training_and_evaluation_seconds": duration})
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
