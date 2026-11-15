"""Наполнить PostgreSQL-таблицы проекта из локальных файлов.

Все таблицы создаются в едином формате (см. src/database.py) идемпотентно:
повторный запуск очищает таблицу и вставляет строки заново.

Примеры:
    uv run python src/database_load.py              # наполнить все таблицы
    uv run python src/database_load.py --tables case_train case_test
    uv run python src/database_load.py --tables igh_deduped_no_conflict
    uv run python src/database_load.py --tables private_test

Локальный db_export/igh_deduped_no_conflict.jsonl импортируется только при
явном выборе его таблицы; запуск без --tables заполняет прежние пять таблиц.
private_test объединяет db_export/private-test-dataset.csv и data/responce.csv
по testcase; source mc_id=NULL, метки включают 101 и загружаются только явно.
"""

from __future__ import annotations

import argparse
import logging

from database import DATABASE_TABLES, DatabaseError, create_schema, load_all, table_sha256


def main() -> int:
    """Создать схему и наполнить выбранные таблицы."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables", nargs="+", choices=DATABASE_TABLES,
        help="таблицы для наполнения (по умолчанию — пять основных, без локальных диагностических)",
    )
    args = parser.parse_args()
    try:
        create_schema()
        counts = load_all(args.tables)
    except DatabaseError as exc:
        logging.error("%s", exc)
        return 1
    for table, count in counts.items():
        logging.info("%s: %d строк, sha256=%s", table, count, table_sha256(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
