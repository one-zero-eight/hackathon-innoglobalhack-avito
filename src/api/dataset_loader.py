"""Лоадер датасета кейса InnoGlobalHack × Avito в PostgreSQL (таблица case_ads).

Читает официальные файлы и загружает их в БД идемпотентно: повторный запуск
не создаёт дубликатов (UNIQUE ``row_key`` + ``ON CONFLICT DO NOTHING``).

Файлы (по умолчанию в ``data/innoglobalhack_avito/innoglobalhack_avito``):
    public-train-annotated.jsonl  train: 10 000 объявлений; массив
        ``found_microcategories`` превращаем в JSONB-массив mcId (101–106),
        ``item_id`` храним в колонке ``item_id``.
    public-test-dataset.parquet   test: 208 объявлений; список названий
        категорий храним в JSONB-колонке ``categories``, ``item_id`` пустой.

Исходная микрокатегория всех официальных объявлений — 101 («Ремонт под ключ»).

Запуск из корня репозитория (пакет ``api`` лежит в ``src``):
    uv run python src/api/dataset_loader.py
    PYTHONPATH=src uv run python -m api.dataset_loader
Параметры: ``--data-dir``, ``--database-url`` (по умолчанию DATABASE_URL
из окружения или корневой ``.env``).
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Прямой запуск скриптом (python src/api/dataset_loader.py): добавляем src/,
# чтобы импорты пакета api и модуля main работали без PYTHONPATH.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pyarrow.parquet as pq

from api.config import PROJECT_ROOT, Settings
from api.db import close_pool, init_pool
from main import MICROCATEGORIES, VALID_MCIDS

logger = logging.getLogger(__name__)

TRAIN_FILENAME = "public-train-annotated.jsonl"
TEST_FILENAME = "public-test-dataset.parquet"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "innoglobalhack_avito" / "innoglobalhack_avito"

# Исходная категория объявлений официального датасета кейса (см. AGENTS.md).
SOURCE_MC_ID = 101
BATCH_SIZE = 1000
# Длина хэша текста в ключе test-строк (см. build_test_row_key).
_HASH_PREFIX = 16

# DDL совпадает с deploy/db/init.sql: CREATE TABLE IF NOT EXISTS, чтобы
# загрузка работала и на свежей базе (init.sql), и на существующей базе,
# где init.sql уже не выполняется.
CASE_ADS_DDL = """
CREATE TABLE IF NOT EXISTS case_ads (
    id            BIGSERIAL PRIMARY KEY,
    split         TEXT NOT NULL CHECK (split IN ('train', 'test')),
    item_id       BIGINT,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    source_mc_id  SMALLINT NOT NULL DEFAULT 101,
    found_mc_ids  JSONB,
    categories    JSONB,
    row_key       TEXT NOT NULL UNIQUE,
    CHECK (found_mc_ids IS NULL OR jsonb_typeof(found_mc_ids) = 'array'),
    CHECK (categories  IS NULL OR jsonb_typeof(categories)  = 'array')
);

CREATE INDEX IF NOT EXISTS case_ads_split_idx ON case_ads (split, id);
"""

# (split, item_id, title, description, source_mc_id, found_mc_ids, categories, row_key)
INSERT_SQL = """
    INSERT INTO case_ads
        (split, item_id, title, description, source_mc_id, found_mc_ids, categories, row_key)
    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8)
    ON CONFLICT (row_key) DO NOTHING
"""

Row = tuple[Any, ...]


def build_train_row_key(item_id: int) -> str:
    """Уникальный ключ train-строки: item_id в датасете не повторяется."""
    return f"train:{item_id}"


def build_test_row_key(index: int, title: str, description: str) -> str:
    """Ключ test-строки: ``test:{номер строки}:{хэш текста}``.

    В public-test-dataset.parquet из 208 строк только 140 уникальны по
    title+description, поэтому порядковый номер обязателен, иначе идемпотентный
    ключ схлопнул бы легальные дубликаты. Хэш текста страхует смещение
    номеров при случайном сдвиге файла.
    """
    digest = hashlib.sha256(f"{title}\n{description}".encode()).hexdigest()[:_HASH_PREFIX]
    return f"test:{index}:{digest}"


def _extract_mc_ids(found: object, position: str) -> list[int]:
    """Список mcId из found_microcategories; за пределами таблицы 101–106 — ошибка."""
    if found is None:
        return []
    if not isinstance(found, list):
        raise ValueError(f"{position}: found_microcategories — не список: {type(found).__name__}")
    mc_ids: list[int] = []
    for entry in found:
        mc_id = entry.get("mcId") if isinstance(entry, dict) else None
        if not isinstance(mc_id, int) or mc_id not in VALID_MCIDS:
            raise ValueError(
                f"{position}: недопустимый mcId {mc_id!r}, ожидаются {sorted(VALID_MCIDS)}"
            )
        if mc_id not in mc_ids:
            mc_ids.append(mc_id)
    return sorted(mc_ids)


def load_train_rows(path: Path) -> list[Row]:
    """Разобрать public-train-annotated.jsonl в кортежи для вставки в case_ads."""
    rows: list[Row] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            position = f"train, строка {line_no}"
            record = json.loads(line)
            item_id = record.get("item_id")
            title = record.get("title")
            description = record.get("description")
            if not isinstance(item_id, int):
                raise ValueError(f"{position}: item_id отсутствует или не целое")
            if not isinstance(title, str) or not isinstance(description, str) or not description:
                raise ValueError(f"{position}: title/description должны быть строками, текст непустой")
            mc_ids = _extract_mc_ids(record.get("found_microcategories"), position)
            rows.append(
                (
                    "train",
                    item_id,
                    title,
                    description,
                    SOURCE_MC_ID,
                    json.dumps(mc_ids),
                    None,
                    build_train_row_key(item_id),
                )
            )
    return rows


def load_test_rows(path: Path) -> list[Row]:
    """Разобрать public-test-dataset.parquet в кортежи для вставки в case_ads."""
    table = pq.read_table(path)
    missing = {"title", "description", "categories"} - set(table.column_names)
    if missing:
        raise ValueError(f"в {path.name} нет колонок: {sorted(missing)}")
    rows: list[Row] = []
    for index, record in enumerate(table.to_pylist()):
        title = record["title"]
        description = record["description"]
        categories = record["categories"]
        position = f"test, строка {index}"
        if not isinstance(title, str) or not isinstance(description, str) or not description:
            raise ValueError(f"{position}: title/description должны быть строками, текст непустой")
        if categories is None:
            categories = []
        if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
            raise ValueError(f"{position}: categories должен быть списком строк")
        rows.append(
            (
                "test",
                None,
                title,
                description,
                SOURCE_MC_ID,
                None,
                json.dumps(list(categories)),
                build_test_row_key(index, title, description),
            )
        )
    return rows


async def insert_rows(pool: Any, rows: list[Row]) -> int:
    """Вставить строки батчами executemany; вернуть число реально добавленных строк.

    Идемпотентность даёт ON CONFLICT (row_key) DO NOTHING: повторный запуск
    лоадера не дублирует объявления.
    """
    async with pool.acquire() as conn:
        before = await conn.fetchval("SELECT count(*) FROM case_ads")
        for start in range(0, len(rows), BATCH_SIZE):
            await conn.executemany(INSERT_SQL, rows[start : start + BATCH_SIZE])
        after = await conn.fetchval("SELECT count(*) FROM case_ads")
    return after - before


async def fetch_split_counts(pool: Any) -> dict[str, int]:
    """Количество строк в case_ads по каждому сплиту."""
    async with pool.acquire() as conn:
        records = await conn.fetch("SELECT split, count(*) AS n FROM case_ads GROUP BY split")
    return {record["split"]: record["n"] for record in records}


async def load_dataset(data_dir: Path, database_url: str) -> dict[str, int]:
    """Загрузить оба файла датасета в case_ads и вернуть итоговые счётчики по сплитам."""
    train_path = data_dir / TRAIN_FILENAME
    test_path = data_dir / TEST_FILENAME
    for path in (train_path, test_path):
        if not path.is_file():
            raise SystemExit(f"нет файла датасета: {path}")

    train_rows = load_train_rows(train_path)
    test_rows = load_test_rows(test_path)
    logger.info("Разобрано из %s: train=%d, test=%d", data_dir, len(train_rows), len(test_rows))

    pool = await init_pool(database_url)
    try:
        async with pool.acquire() as conn:
            await conn.execute(CASE_ADS_DDL)
        inserted = await insert_rows(pool, train_rows) + await insert_rows(pool, test_rows)
        counts = await fetch_split_counts(pool)
    finally:
        await close_pool(pool)

    skipped = len(train_rows) + len(test_rows) - inserted
    logger.info("Добавлено %d строк, %d уже было в базе (идемпотентный повтор)", inserted, skipped)
    logger.info(
        "case_ads по сплитам: train=%d, test=%d (ожидаем train=%d, test=%d)",
        counts.get("train", 0),
        counts.get("test", 0),
        len(train_rows),
        len(test_rows),
    )
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Аргументы командной строки лоадера."""
    parser = argparse.ArgumentParser(
        description="Загрузка датасета кейса (train.jsonl + test.parquet) в таблицу case_ads."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"каталог с файлами датасета (по умолчанию {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="DSN PostgreSQL; по умолчанию DATABASE_URL из окружения или .env",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> None:
    """Разобрать аргументы и выполнить загрузку датасета."""
    args = parse_args(argv)
    database_url = args.database_url or Settings().database_url
    await load_dataset(args.data_dir, database_url)


def main(argv: list[str] | None = None) -> None:
    """Точка входа CLI: логирование и запуск asyncio-загрузки."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Микрокатегории кейса: %s", sorted(MICROCATEGORIES))
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
