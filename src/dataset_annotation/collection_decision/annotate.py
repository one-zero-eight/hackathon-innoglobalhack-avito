"""Разметить внешние коллекции для просмотра примеров."""

import argparse
import fcntl
import json
import logging
import os
import random
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

from subcategory_extraction.decisions.google_decision.client import DEFAULT_MODEL, TokenBudget
from dataset_annotation.collection_decision.pipeline import CollectionAnnotationPipeline

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


class BlockedContent(Exception):
    """Google отклонил описание по правилам обработки контента."""


def score(pipeline: CollectionAnnotationPipeline, description: str) -> dict[int, float]:
    """Повторить запрос при временной ошибке, сохраняя общий бюджет."""
    for attempt in range(5):
        try:
            return pipeline.score_description(description)
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            if isinstance(exc, HTTPError) and exc.code == 400:
                try:
                    error = json.load(exc)
                except (ValueError, UnicodeError):
                    error = {}
                if isinstance(error, dict) and isinstance(error.get("error"), dict):
                    if error["error"].get("code") == "prohibited_content":
                        raise BlockedContent from exc
            # Decision иногда возвращает 400 для запроса, который проходит при повторе.
            if isinstance(exc, HTTPError) and exc.code not in (400, 408, 429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise
            delay = min(120, 5 * 2 ** attempt) + random.uniform(0, 2)
            if isinstance(exc, HTTPError):
                retry_after = exc.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            logging.warning("Временная ошибка %s. Повтор через %.1f с", f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__, delay)
            sleep(delay)
    raise AssertionError("Исчерпаны попытки")


def annotate(path: Path, pipeline: CollectionAnnotationPipeline, limit: int | None) -> None:
    """Добавить оценки в JSONL с проверкой уже сохранённого префикса."""
    output = path.with_name(f"{path.stem}-decision-1.jsonl")
    settings = configuration(pipeline)
    with path.open(encoding="utf-8") as source, output.open("a+b") as target:
        saved_count = 0
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

            saved_count += 1

        logging.info("%s: сохранено до запуска %d записей", output.name, saved_count)
        for index, line in enumerate(source):
            if limit is not None and index == limit:
                break
            row = json.loads(line)
            try:
                probabilities = score(pipeline, row["description"])
            except BlockedContent:
                probabilities = None
                logging.warning("%s: строка %d заблокирована Google", path.name, saved_count + index + 1)
            result = {
                "source": row,
                "configuration": settings,
                "probabilities": probabilities,
                "found_microcategories": sorted(
                    cid for cid, value in probabilities.items() if value >= pipeline.threshold
                ) if probabilities is not None else None,
            }
            if probabilities is None:
                result["error"] = "prohibited_content"
            target.write((json.dumps(result, ensure_ascii=False) + "\n").encode())
            target.flush()
            os.fsync(target.fileno())
            if (saved_count + index + 1) % 25 == 0:
                logging.info("%s: всего сохранено %s записей", output.name, saved_count + index + 1)
    logging.info("Результат: %s", output)


def configuration(pipeline: CollectionAnnotationPipeline) -> dict:
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
    parser.add_argument("--inputs", nargs="+", type=Path, default=INPUTS)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть положительным")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")
    pipeline = CollectionAnnotationPipeline()
    if not pipeline.client.api_key:
        parser.error("Укажите LLM_API_KEY или GOOGLE_API_KEY в .env")
    pipeline.client.model_id = DEFAULT_MODEL
    pipeline.client.budget = CollectionBudget()
    (ROOT / "out").mkdir(parents=True, exist_ok=True)
    with (ROOT / "out/.decision-annotation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in args.inputs:
            annotate(path, pipeline, args.limit)


if __name__ == "__main__":
    main()
