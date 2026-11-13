"""Сравнение промптов Gemini Decision 1 на объединённом train.csv."""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from random import Random
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from main import MICROCATEGORIES, PipelineInput
from subcategory_extraction.benchmarks.metrics import score
from subcategory_extraction.google_decision.client import DEFAULT_MODEL, TokenBudget
from subcategory_extraction.google_decision.pipeline import (
    CATEGORY_EXAMPLES, PROMPT_DIR, SubcategoryExtractionPipeline,
)

DATASET = ROOT / "local-notes/joined-datasets/train.csv"
MODEL = DEFAULT_MODEL
THRESHOLD = 0.5
CATEGORIES = {cid: (name, CATEGORY_EXAMPLES[cid]) for cid, name in MICROCATEGORIES.items()}
logger = logging.getLogger(__name__)


def sample_rows(dataset: Path, size: int, seed: int) -> list[dict[str, str]]:
    """Выбрать строки с разными описаниями по group_id."""
    with dataset.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    Random(seed).shuffle(rows)
    groups: dict[str, dict[str, str]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], row)
    if not 1 <= size <= len(groups):
        raise ValueError(f"Размер выборки должен быть от 1 до {len(groups)}")
    return list(groups.values())[:size]


def predict(api_key: str, prompt: str, description: str, source: int,
             budget: TokenBudget, categories: set[int] | None = None) -> dict[int, float]:
    """Совместимость исторических прогонов с модульной реализацией."""
    pipeline = SubcategoryExtractionPipeline()
    pipeline.prompt = prompt
    pipeline.client.api_key = api_key
    pipeline.client.budget = budget
    return pipeline.probabilities(PipelineInput(description=description, mcId=source), categories)


def main() -> None:
    """Оценить оба промпта на одной выборке."""
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, default=HERE / "out/decision-1.json")
    args = parser.parse_args()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        parser.error("Укажите GOOGLE_API_KEY в окружении или .env")

    rows = sample_rows(args.dataset, args.sample_size, args.seed)
    sources = [int(row["source_mc_id"]) for row in rows]
    expected = [set(json.loads(row["target_mc_ids"])) for row in rows]
    for source, targets in zip(sources, expected, strict=True):
        if source not in CATEGORIES or source in targets or targets - CATEGORIES.keys():
            raise ValueError("Некорректные категории в train.csv")

    report = {
        "model": MODEL,
        "seed": args.seed,
        "threshold": THRESHOLD,
        "sample": [
            {"group_id": row["group_id"], "source_mc_id": source, "expected": sorted(targets)}
            for row, source, targets in zip(rows, sources, expected, strict=True)
        ],
        "results": {},
    }
    budget = TokenBudget()
    for variant in ("baseline", "guided"):
        prompt = (PROMPT_DIR / f"google_decision_{variant}.md").read_text(encoding="utf-8")
        probabilities = []
        for index, (row, source) in enumerate(zip(rows, sources, strict=True), start=1):
            try:
                probabilities.append(predict(api_key, prompt, row["description"], source, budget))
            except HTTPError as exc:
                raise SystemExit(f"{variant} {index}/{len(rows)}: HTTP {exc.code}") from None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"{variant} {index}/{len(rows)}: {type(exc).__name__}") from None
        predicted = [{mcid for mcid, p in scores.items() if p >= THRESHOLD} for scores in probabilities]
        metrics = score(expected, predicted)
        logger.info("%s micro-F1 %.4f, TP %d, FP %d, FN %d",
                    variant, metrics["micro_f1"], metrics["tp"], metrics["fp"], metrics["fn"])
        report["results"][variant] = {
            **metrics, "predicted": [sorted(ids) for ids in predicted], "probabilities": probabilities
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
