"""Доступ к PostgreSQL: пул asyncpg, кэш справочника и запросы к датасету кейса.

Приложение не должно падать на старте при недоступной базе:
пул создаётся с ``min_size=0``, первое подключение откладывается на запрос.
"""

import asyncio
import json
import logging
import time
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

CATALOG_SQL = "SELECT mc_id, title FROM microcategories ORDER BY mc_id"
POOL_MAX_SIZE = 5

# Простой кэш справочника на уровне модуля: {(monotonic-время, данные)}.
_catalog_cache: dict[int, str] | None = None
_catalog_cached_at: float = 0.0
_catalog_lock = asyncio.Lock()


async def init_pool(dsn: str) -> asyncpg.Pool:
    """Создать пул соединений; к базе на этом шаге не подключаемся."""
    pool = await asyncpg.create_pool(dsn=dsn, min_size=0, max_size=POOL_MAX_SIZE)
    logger.info("Пул соединений PostgreSQL создан")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    """Закрыть все соединения пула."""
    await pool.close()
    logger.info("Пул соединений PostgreSQL закрыт")


async def get_catalog(pool: asyncpg.Pool, ttl_s: int = 60) -> dict[int, str]:
    """Справочник {mcId: название} из БД с TTL-кэшем на ``ttl_s`` секунд.

    Ошибки БД не глотаются: они доходят до обработчиков приложения (HTTP 503).
    """
    global _catalog_cache, _catalog_cached_at
    async with _catalog_lock:
        now = time.monotonic()
        if _catalog_cache is not None and now - _catalog_cached_at < ttl_s:
            return dict(_catalog_cache)
        async with pool.acquire() as conn:
            rows = await conn.fetch(CATALOG_SQL)
        _catalog_cache = {row["mc_id"]: row["title"] for row in rows}
        _catalog_cached_at = now
        logger.debug("Кэш микрокатегорий обновлён: %d шт.", len(_catalog_cache))
        return dict(_catalog_cache)


# --- Датасет кейса (таблица case_ads, см. api.dataset_loader) ---

DATASET_COLUMNS = (
    "id, split, item_id, title, description, source_mc_id, found_mc_ids, categories"
)


def _decode_dataset_row(row: asyncpg.Record) -> dict[str, Any]:
    """asyncpg отдаёт JSONB текстом: раскодируем found_mc_ids и categories в списки."""
    data: dict[str, Any] = dict(row)
    for column in ("found_mc_ids", "categories"):
        if data.get(column) is not None:
            data[column] = json.loads(data[column])
    return data


async def fetch_dataset_samples(
    pool: asyncpg.Pool, split: str, limit: int, offset: int
) -> list[dict[str, Any]]:
    """Страница объявлений сплита: по возрастанию id, не более ``limit`` со сдвигом ``offset``."""
    sql = f"SELECT {DATASET_COLUMNS} FROM case_ads WHERE split = $1 ORDER BY id LIMIT $2 OFFSET $3"
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, split, limit, offset)
    return [_decode_dataset_row(row) for row in rows]


async def fetch_random_dataset_sample(pool: asyncpg.Pool, split: str) -> dict[str, Any] | None:
    """Один случайный сэмпл сплита; на датасете кейса ORDER BY random() допустим."""
    sql = f"SELECT {DATASET_COLUMNS} FROM case_ads WHERE split = $1 ORDER BY random() LIMIT 1"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, split)
    return _decode_dataset_row(row) if row is not None else None


async def fetch_dataset_stats(pool: asyncpg.Pool) -> dict[str, int]:
    """Число объявлений по сплитам: {'train': N, 'test': M} (пустой сплит — 0)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT split, count(*) AS n FROM case_ads GROUP BY split")
    counts = {"train": 0, "test": 0}
    for row in rows:
        counts[row["split"]] = row["n"]
    return counts
