"""Разметить внешние коллекции для просмотра примеров."""

import argparse
import fcntl
import json
import logging
import random
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

from subcategory_extraction.google_decision.client import DEFAULT_MODEL, TokenBudget
from subcategory_extraction.google_decision.pipeline import SubcategoryExtractionPipeline

ROOT = Path(__file__).resolve().parents[3]
INPUTS = (ROOT / "out/youla-1469.jsonl", ROOT / "out/yandex/profiles-1492.jsonl")


class CollectionBudget(TokenBudget):
    """Ограничить число запросов и входных токенов за минуту."""

    limit = 48_000
    request_limit = 200

    def reserve(self, estimate: int) -> None:
        """Учитывать каждую попытку, включая повторы клиента."""
        if len(self.requests) >= self.request_limit:
            sleep(max(0, 60 - (monotonic() - self.requests[0][0])))
        super().reserve(estimate)


def score(pipeline: SubcategoryExtractionPipeline, description: str) -> dict[int, float]:
    """Повторить запрос при временной ошибке, сохраняя общий бюджет."""
    for attempt in range(5):
        try:
            return pipeline.score_description(description)
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise
            delay = min(120, 5 * 2 ** attempt) + random.uniform(0, 2)
            if isinstance(exc, HTTPError):
                retry_after = exc.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            logging.warning("Временная ошибка %s. Повтор через %.1f с", type(exc).__name__, delay)
            sleep(delay)
    raise AssertionError("Исчерпаны попытки")


def annotate(path: Path, pipeline: SubcategoryExtractionPipeline, limit: int | None) -> None:
    """Добавить оценки в JSONL с проверкой уже сохранённого префикса."""
    output = path.with_name(f"{path.stem}-decision-1.jsonl")
    settings = configuration(pipeline)
    with path.open(encoding="utf-8") as source, output.open("a+b") as target:
        target.seek(0)
        for line in target:
            if not line.endswith(b"\n"):
                target.truncate(target.tell() - len(line))
                break
            saved = json.loads(line)
            if saved["source"] != json.loads(next(source, "null")):
                raise ValueError(f"Исходные записи изменились: {path}")
            if saved["configuration"] != settings:
                raise ValueError(f"Настройки разметки изменились: {output}")

        for index, line in enumerate(source):
            if limit is not None and index == limit:
                break
            row = json.loads(line)
            probabilities = score(pipeline, row["description"])
            result = {
                "source": row,
                "configuration": settings,
                "probabilities": probabilities,
                "found_microcategories": sorted(
                    cid for cid, value in probabilities.items() if value >= pipeline.threshold
                ),
            }
            target.write((json.dumps(result, ensure_ascii=False) + "\n").encode())
            target.flush()
            if (index + 1) % 25 == 0:
                logging.info("%s: добавлено %s записей", output.name, index + 1)
    logging.info("Результат: %s", output)


def configuration(pipeline: SubcategoryExtractionPipeline) -> dict:
    """Сохранить настройки вместе с каждой оценкой."""
    return {
        "model": pipeline.client.model_id,
        "prompt_sha256": sha256(pipeline.prompt.encode()).hexdigest(),
        "threshold": pipeline.threshold,
    }


def main() -> None:
    """Запустить одну общую очередь для двух коллекций."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Сколько новых записей обработать в каждой коллекции")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть положительным")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")
    pipeline = SubcategoryExtractionPipeline()
    pipeline.client.model_id = DEFAULT_MODEL
    pipeline.client.budget = CollectionBudget()
    with (ROOT / "out/.decision-annotation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in INPUTS:
            annotate(path, pipeline, args.limit)


if __name__ == "__main__":
    main()
