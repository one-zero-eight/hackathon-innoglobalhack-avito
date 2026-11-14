"""Доступ к PostgreSQL: пул asyncpg и TTL-кэш справочника микрокатегорий.

Приложение не должно падать на старте при недоступной базе:
пул создаётся с ``min_size=0``, первое подключение откладывается на запрос.
"""

import asyncio
import logging
import time

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
