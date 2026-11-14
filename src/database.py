"""Единый слой PostgreSQL для датасетов проекта.

Все таблицы хранятся в одном формате: колонки
``row_id`` (номер строки в источнике), ``description``, ``title``,
``mc_id`` (исходная микрокатегория объявления), ``mc_ids`` (дополнительные
целевые категории как массив int), ``source`` (платформа/датасет) и
``extra`` (все остальные исходные поля строки как JSON).

Назначение таблиц:

- ``case_train`` — официальный InnoGlobalHack train
  (docs/innoglobalhack_avito/public-train-annotated.jsonl).
- ``case_test`` — официальный InnoGlobalHack test
  (docs/innoglobalhack_avito/public-test-dataset.parquet).
- ``manual_train``, ``manual_test`` — журнал ручной смысловой разметки
  (docs/manual_review/inputs.jsonl + reviews/*.jsonl), разбитый по
  исходному сплиту ``sources[].split``.
- ``yandex_yoola`` — внешние снимки Яндекса и Юлы
  (docs/provider_collections/*), объединённые в один формат.

Чтение из БД возвращает строки в виде, совместимом с существующими
функциями ``run.prepare_rows``/``read_records``: каждая строка —
dict с ключами ``description``, ``mcId`` и ``found_microcategories``
(список ``{"mcId": int, "label": str}``). Так код бенчмарков не зависит
от того, лежат ли данные в файлах или в PostgreSQL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from main import MICROCATEGORIES, mcid_name

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Все таблицы проекта в едином формате.
# Порядок списка — порядок наполнения (yandex_yoola может ссылаться на общее).
DATABASE_TABLES: tuple[str, ...] = (
    "case_train",
    "case_test",
    "manual_train",
    "manual_test",
    "yandex_yoola",
)

CONNECT_ATTEMPTS = 16
CONNECT_RETRY_DELAY_SECONDS = 0.4

TARGET_CATEGORIES: tuple[int, ...] = tuple(
    cid for cid in sorted(MICROCATEGORIES) if cid != 101
)


class DatabaseError(RuntimeError):
    """Ошибка соединения или целостности данных PostgreSQL."""


def env(name: str, default: str = "") -> str:
    """Вернуть переменную окружения после гарантированной загрузки .env."""
    load_dotenv(ROOT / ".env")
    value = Path(default)
    import os

    return os.getenv(name, default)


def dsn() -> str:
    """Собрать DSN из переменных окружения (шаблон в .env.example)."""
    host = env("DB_HOST")
    port = env("DB_PORT", "5433")
    name = env("DB_NAME", "avito")
    user = env("DB_USER", "avito")
    password = env("DB_PASSWORD")
    sslmode = env("DB_SSLMODE", "prefer")
    if not host or not password:
        raise DatabaseError(
            "Задайте DB_HOST и DB_PASSWORD в .env (шаблон — в .env.example)"
        )
    return (
        f"host={host} port={port} dbname={name} user={user} "
        f"password={password} sslmode={sslmode} connect_timeout=5"
    )


def _connect() -> psycopg.Connection:
    """Установить соединение с повторами: балансировщик иногда закрывает порт."""
    last_error: Exception | None = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return psycopg.connect(dsn())
        except psycopg.OperationalError as exc:
            last_error = exc
            logger.warning("PostgreSQL: попытка %d/%d не удалась (%s)",
                           attempt, CONNECT_ATTEMPTS, exc)
            if attempt < CONNECT_ATTEMPTS:
                time.sleep(CONNECT_RETRY_DELAY_SECONDS)
    raise DatabaseError(f"Не удалось подключиться к PostgreSQL: {last_error}") from last_error


def _rows_to_records(rows: Iterable[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Преобразовать строки БД в формат, понятный prepare_rows/read_records."""
    records: list[dict[str, Any]] = []
    for row_id, description, title, mc_id, mc_ids, source, source_sha256, extra in rows:
        labels = [{"mcId": cid, "label": mcid_name(cid)} for cid in sorted(mc_ids)]
        record: dict[str, Any] = {
            "description": description,
            "mcId": mc_id,
            "found_microcategories": labels,
        }
        if title is not None:
            record["title"] = title
        record["_row_id"] = row_id
        record["_source"] = source
        record["_source_sha256"] = source_sha256
        if isinstance(extra, dict):
            record["_extra"] = extra
        records.append(record)
    return records


def read_records(table: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Прочитать таблицу БД в порядке ``row_id``.

    Возвращает dict-строки в формате ``run.prepare_rows``. ``limit``
    ограничивает первые строки таблицы (для быстрых проверок).
    """
    if table not in DATABASE_TABLES:
        raise DatabaseError(f"Неизвестная таблица: {table}")
    query = (
        "SELECT row_id, description, title, mc_id, mc_ids, source, source_sha256, extra "
        f"FROM {table} ORDER BY row_id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with _connect() as connection:
        cursor = connection.execute(query)
        return _rows_to_records(cursor.fetchall())


def read_raw(table: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Прочитать строки таблицы без адаптации под prepare_rows (сырой вид)."""
    if table not in DATABASE_TABLES:
        raise DatabaseError(f"Неизвестная таблица: {table}")
    query = (
        "SELECT row_id, description, title, mc_id, mc_ids, source, source_sha256, extra "
        f"FROM {table} ORDER BY row_id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with _connect() as connection:
        cursor = connection.execute(query)
        return [
            {
                "row_id": row_id,
                "description": description,
                "title": title,
                "mc_id": mc_id,
                "mc_ids": list(mc_ids),
                "source": source,
                "source_sha256": source_sha256,
                "extra": extra,
            }
            for row_id, description, title, mc_id, mc_ids, source, source_sha256, extra in cursor.fetchall()
        ]


def table_sha256(table: str) -> str:
    """Хеш содержимого таблицы для сравнения протоколов прогонов.

    Считается по каждой строке в порядке ``row_id``: описание, целевые
    категории и имя источника. Воспроизводим для одинакового содержимого
    независимо от порядка вставки.
    """
    if table not in DATABASE_TABLES:
        raise DatabaseError(f"Неизвестная таблица: {table}")
    query = (
        "SELECT row_id, description, mc_id, mc_ids, source "
        f"FROM {table} ORDER BY row_id"
    )
    digest = hashlib.sha256()
    with _connect() as connection:
        for row_id, description, mc_id, mc_ids, source in connection.execute(query).fetchall():
            payload = {
                "row_id": row_id,
                "description": description,
                "mc_id": mc_id,
                "mc_ids": sorted(mc_ids),
                "source": source,
            }
            digest.update((json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def read_training_rows(table: str) -> tuple[list[dict[str, Any]], str]:
    """Прочитать обучающую таблицу в формате trainer'ов.

    Возвращает пару ``(rows, sha256)``, где каждая строка —
    ``{"description": str, "targets": set[int]}``, а хеш считается по
    содержимому таблицы в порядке ``row_id`` (совместимо с
    ``semantic_encoder.train.load_training_rows`` по смыслу).
    """
    records = read_records(table)
    rows = []
    for record in records:
        rows.append({
            "description": record["description"],
            "targets": {item["mcId"] for item in record["found_microcategories"]},
        })
    if not rows:
        raise DatabaseError(f"Обучающая таблица пуста: {table}")
    return rows, table_sha256(table)


# --- Наполнение таблиц --------------------------------------------------


def normalize_mc_ids(values: Iterable[Any] | None) -> list[int]:
    """Оставить только валидные целевые категории 102–106, убрать повторы."""
    if values is None:
        return []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("mcId", value.get("mc_id"))
        try:
            cid = int(value)
        except (TypeError, ValueError):
            continue
        if cid in TARGET_CATEGORIES:
            seen.add(cid)
    return sorted(seen)


def categories_names_to_ids(names: Iterable[Any]) -> list[int]:
    """Перевести названия категорий (как в parquet case_test) в mc_ids."""
    name_to_id = {name: cid for cid, name in MICROCATEGORIES.items()}
    ids: set[int] = set()
    for name in names:
        cid = name_to_id.get(name)
        if cid in TARGET_CATEGORIES:
            ids.add(cid)
    return sorted(ids)


def _prepare_row(
    row_id: int,
    description: str,
    mc_ids: Iterable[int],
    *,
    source: str,
    mc_id: int = 101,
    title: str | None = None,
    source_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[int, str, str | None, int, list[int], str, str | None, dict[str, Any]]:
    """Собрать строку таблицы единого формата без обращения к БД."""
    return (
        int(row_id),
        str(description),
        title,
        int(mc_id),
        normalize_mc_ids(mc_ids),
        source,
        source_sha256,
        extra if isinstance(extra, dict) else {},
    )


def _source_digest(files: Iterable[Path]) -> str:
    """Хэш исходных файлов, из которых строится таблица.

    Для одного файла — его собственный SHA-256 (совпадает с прежним
    ``dataset_sha256`` в recorded-results). Для нескольких файлов — хэш
    по байтам каждого в порядке аргументов. Хранится в колонке
    ``source_sha256`` и используется для сверки протоколов бенчмарков.
    """
    files = list(files)
    if len(files) == 1:
        return hashlib.sha256(files[0].read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _insert_many(table: str, rows: list[tuple[int, str, str | None, int, list[int], str, str | None, dict[str, Any]]]) -> None:
    """Очистить таблицу и вставить строки (идемпотентно, устойчиво к обрывам).

    Балансировщик иногда рвёт соединение посреди вставки, поэтому каждая
    пачка коммитится отдельно (autocommit), а INSERT использует
    ON CONFLICT DO NOTHING: повторный прогон той же пачки безопасен.
    При обрыве соединение поднимается заново и вставка продолжается.
    """
    if table not in DATABASE_TABLES:
        raise DatabaseError(f"Неизвестная таблица: {table}")
    insert = (
        f"INSERT INTO {table} "
        "(row_id, description, title, mc_id, mc_ids, source, source_sha256, extra) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (row_id) DO NOTHING"
    )
    prepared = []
    for row in rows:
        _, description, _, _, mc_ids, _, _, extra = row
        if not isinstance(description, str) or not description.strip():
            raise DatabaseError("description не может быть пустым")
        if not isinstance(mc_ids, list) or not all(type(cid) is int for cid in mc_ids):
            raise DatabaseError("mc_ids должны быть списком int")
        prepared.append((*row[:7], Jsonb(extra)))
    with _connect() as connection:
        connection.execute(f"TRUNCATE {table}")
        connection.commit()
    connection = None
    batch_rows = 500
    index = 0
    while index < len(prepared):
        batch = prepared[index:index + batch_rows]
        try:
            if connection is None or connection.closed:
                connection = _connect()
                connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.executemany(insert, batch)
            index += batch_rows
        except psycopg.OperationalError as exc:
            logger.warning("%s: обрыв соединения на строке %d, повтор пачки (%s)",
                           table, index, exc)
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            connection = None
        except Exception:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            raise
    if connection is not None:
        connection.close()
    logger.info("Таблица %s: вставлено %d строк", table, len(rows))


def load_case_train() -> int:
    """Наполнить case_train из официального train-файла."""
    source = ROOT / "docs/innoglobalhack_avito/public-train-annotated.jsonl"
    source_sha256 = _source_digest([source])
    rows = []
    for row_id, line in enumerate(_read_json_lines(source), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(_prepare_row(
            row_id,
            row["description"],
            [item["mcId"] for item in row["found_microcategories"]],
            source="avito_case_train",
            title=row.get("title"),
            mc_id=101,
            source_sha256=source_sha256,
            extra={"item_id": row.get("item_id")},
        ))
    _insert_many("case_train", rows)
    return len(rows)


def load_case_test() -> int:
    """Наполнить case_test из официального parquet test."""
    import pyarrow.parquet as pq

    source = ROOT / "docs/innoglobalhack_avito/public-test-dataset.parquet"
    source_sha256 = _source_digest([source])
    records = pq.read_table(source).to_pylist()
    rows = [
        _prepare_row(
            row_id,
            record["description"],
            categories_names_to_ids(record.get("categories", [])),
            source="avito_case_test",
            title=record.get("title"),
            mc_id=101,
            source_sha256=source_sha256,
            extra={"categories_names": record.get("categories")},
        )
        for row_id, record in enumerate(records, start=1)
    ]
    _insert_many("case_test", rows)
    return len(rows)


def _load_manual(table: str, split: str) -> int:
    """Наполнить manual_train/manual_test из журнала ручной разметки."""
    inputs_path = ROOT / "docs/manual_review/inputs.jsonl"
    reviews_dir = ROOT / "docs/manual_review/reviews"
    inputs = [
        json.loads(line) for line in _read_json_lines(inputs_path) if line.strip()
    ]
    review_by_group: dict[str, dict[str, Any]] = {}
    for path in sorted(reviews_dir.glob("*.jsonl")):
        for line in _read_json_lines(path):
            if not line.strip():
                continue
            review = json.loads(line)
            review_by_group.setdefault(review["group_id"], review)
    source_sha256 = _source_digest([inputs_path, *sorted(reviews_dir.glob("*.jsonl"))])
    rows = []
    for index, record in enumerate(inputs, start=1):
        splits = {source.get("split") for source in record.get("sources", [])}
        if split not in splits:
            continue
        review = review_by_group.get(record["group_id"])
        labels = review["labels"] if review and isinstance(review.get("labels"), list) else []
        rows.append(_prepare_row(
            index,
            record["description"],
            labels,
            source=f"manual_review_{split}",
            mc_id=101,
            source_sha256=source_sha256,
            extra={
                "group_id": record["group_id"],
                "sources": record.get("sources"),
                "index_in_journal": record.get("index"),
                "review": {
                    "status": review.get("status") if review else None,
                    "evidence": review.get("evidence") if review else None,
                    "reason": review.get("reason") if review else None,
                    "question": review.get("question") if review else None,
                    "policy_revision": review.get("policy_revision") if review else None,
                } if review else None,
            },
        ))
    if not rows:
        raise DatabaseError(f"Нет строк для сплита {split} в журнале manual_review")
    _insert_many(table, rows)
    return len(rows)


def load_manual_train() -> int:
    """Наполнить manual_train из журнала ручной разметки (сплит train)."""
    return _load_manual("manual_train", "train")


def load_manual_test() -> int:
    """Наполнить manual_test из журнала ручной разметки (сплит test)."""
    return _load_manual("manual_test", "test")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Прочитать JSONL-файл, пропуская пустые строки.

    Разбиение только по ``\\n``: ``splitlines()`` ломает строки, внутри
    которых встречаются юникод-разделители (например, ``\\u2028``).
    """
    text = path.read_text(encoding="utf-8")
    return [
        json.loads(line)
        for line in text.split("\n")
        if line.strip()
    ]


def _read_json_lines(path: Path) -> list[str]:
    """Прочитать строки файла с разбиением только по ``\\n``."""
    return path.read_text(encoding="utf-8").split("\n")


def _merge_decision(
    key: str,
    maps: dict[str, dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    *,
    annotated_rows: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Объединить raw-снимки и разметки Decision 1 по ключу платформы.

    Единый формат: description/title берутся из raw (или из вложенного
    ``source`` в decision-1, если raw-строки нет), метки — из
    ``target_mc_ids`` (annotated) или ``found_microcategories`` (decision-1).
    Все исходные поля сохраняются в ``extra``.
    """
    combined: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        key_value = raw.get(key)
        if key_value is None:
            continue
        combined[str(key_value)] = dict(raw)
    for decision in decision_rows:
        found = decision.get("found_microcategories")
        key_value = None
        if isinstance(decision.get("source"), dict):
            key_value = decision["source"].get(key)
        if key_value is None:
            continue
        key_value = str(key_value)
        record = combined.get(key_value) or {
            **decision.get("source", {}),
            "probabilities": decision.get("probabilities"),
            "found_microcategories_raw": found,
        }
        record["probabilities"] = decision.get("probabilities", record.get("probabilities"))
        record["found_microcategories"] = found if found is not None else record.get("found_microcategories")
        record["decision_configuration"] = decision.get("configuration", record.get("decision_configuration"))
        combined[key_value] = record
    for annotated in annotated_rows or []:
        key_value = annotated.get(key)
        if key_value is None:
            continue
        key_value = str(key_value)
        record = combined.get(key_value) or dict(annotated)
        record.update(annotated)
        record["target_mc_ids"] = annotated.get("target_mc_ids")
        record["probabilities"] = annotated.get("probabilities", record.get("probabilities"))
        record["input_tokens"] = annotated.get("input_tokens", record.get("input_tokens"))
        combined[key_value] = record
    return combined


def load_yandex_yoola() -> int:
    """Наполнить yandex_yoola: объединить снимки Яндекса и Юлы.

    Юла: youla-1469-raw + youla-1469-decision-1 + youla-859-annotated.
    Яндекс: yandex-1492-raw + yandex-1492-decision-1.
    Ключ слияния для Юлы — ``url``, для Яндекса — ``provider_id``.
    """
    provider_dir = ROOT / "docs/provider_collections"
    source_files = [
        provider_dir / "youla-1469-raw.jsonl",
        provider_dir / "youla-1469-decision-1.jsonl",
        provider_dir / "youla-859-annotated.jsonl",
        provider_dir / "yandex-1492-raw.jsonl",
        provider_dir / "yandex-1492-decision-1.jsonl",
    ]
    youla_raw = _load_jsonl(source_files[0])
    youla_decision = _load_jsonl(source_files[1])
    youla_annotated = _load_jsonl(source_files[2])
    yandex_raw = _load_jsonl(source_files[3])
    yandex_decision = _load_jsonl(source_files[4])
    source_sha256 = _source_digest(source_files)

    youla = _merge_decision("url", {}, youla_raw, youla_decision, annotated_rows=youla_annotated)
    yandex = _merge_decision("provider_id", {}, yandex_raw, yandex_decision)

    rows = []
    row_id = 0
    for platform, key, merged in (
        ("youla", "url", youla),
        ("yandex", "provider_id", yandex),
    ):
        for key_value, record in sorted(merged.items()):
            row_id += 1
            if isinstance(record.get("description"), str) and record["description"].strip():
                description = record["description"]
            elif isinstance(record.get("source"), dict) and isinstance(record["source"].get("description"), str):
                description = record["source"]["description"]
            else:
                logger.warning("yandex_yoola: строка без description пропущена (%s=%s)", platform, key_value)
                continue
            title = record.get("title") if isinstance(record.get("title"), str) else None
            mc_ids = normalize_mc_ids(record.get("target_mc_ids") or record.get("found_microcategories"))
            extra = {field: record.get(field) for field in (
                "url", "provider_id", "catalog_url", "source_type", "sample_mc_id",
                "probabilities", "input_tokens", "decision_configuration",
                "found_microcategories", "target_mc_ids",
            ) if record.get(field) is not None}
            rows.append(_prepare_row(
                row_id,
                description,
                mc_ids,
                source=f"{platform}_provider",
                mc_id=int(record.get("sample_mc_id") or 101),
                title=title,
                source_sha256=source_sha256,
                extra=extra,
            ))
    if not rows:
        raise DatabaseError("yandex_yoola пуст: нет строк из снимков провайдеров")
    _insert_many("yandex_yoola", rows)
    return len(rows)


LOADERS: dict[str, Any] = {
    "case_train": load_case_train,
    "case_test": load_case_test,
    "manual_train": load_manual_train,
    "manual_test": load_manual_test,
    "yandex_yoola": load_yandex_yoola,
}


def create_schema() -> None:
    """Создать все таблицы единого формата (идемпотентно).

    Для существующих таблиц добавляет недостающие колонки без сброса данных.
    """
    create = """
    CREATE TABLE IF NOT EXISTS {table} (
        row_id BIGINT PRIMARY KEY,
        description TEXT NOT NULL,
        title TEXT,
        mc_id INTEGER NOT NULL,
        mc_ids INTEGER[] NOT NULL,
        source TEXT NOT NULL,
        source_sha256 TEXT,
        extra JSONB NOT NULL DEFAULT '{{}}'
    )
    """
    with _connect() as connection:
        for table in DATABASE_TABLES:
            connection.execute(create.format(table=table))
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS source_sha256 TEXT"
            )
    logger.info("Схема БД готова: %s", ", ".join(DATABASE_TABLES))


def load_all(tables: Iterable[str] | None = None) -> dict[str, int]:
    """Наполнить выбранные таблицы; по умолчанию — все."""
    create_schema()
    selected = list(tables) if tables is not None else list(DATABASE_TABLES)
    unknown = [table for table in selected if table not in DATABASE_TABLES]
    if unknown:
        raise DatabaseError(f"Неизвестные таблицы: {', '.join(unknown)}")
    counts: dict[str, int] = {}
    for table in selected:
        counts[table] = int(LOADERS[table]())
    return counts
