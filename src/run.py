"""Оценка полного решения: micro-F1 по mcId из итоговых черновиков.

Источники датасетов — таблицы PostgreSQL (единый формат из src/database.py):
``dataset`` и ``training_dataset`` в run.config.json задаются именами таблиц
(case_test/case_train и т.п.), а не путями к файлам.
"""

import argparse
import json
import logging
import time
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from database import (
    DATABASE_TABLES,
    DatabaseError,
    read_records as database_read_records,
    table_sha256,
)
from main import (
    LLMCostLedger,
    MICROCATEGORIES,
    McId,
    PipelineInput,
    load_config,
    run_pipeline,
    track_llm_costs,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG = ROOT / "run.config.json"
CLASSIFICATION_POINTS = 60
logger = logging.getLogger(__name__)


class RunConfig(BaseModel):
    """Параметры оценки; пути к файлам считаются от файла конфигурации,
    а ``dataset``/``training_dataset`` — имена таблиц PostgreSQL."""

    model_config = ConfigDict(extra="forbid", strict=True)

    pipeline_config: str
    dataset: str
    training_dataset: str | None = None
    report: str
    source_mc_id: McId
    limit: int | None = Field(default=None, gt=0)
    log_every: int = Field(default=10, gt=0)

    @property
    def dataset_table(self) -> str:
        """Имя таблицы БД, заданное в dataset (без нормализации пути)."""
        return str(self.dataset).strip()

    @property
    def training_table(self) -> str | None:
        """Имя таблицы БД для обучения, если задано training_dataset."""
        return str(self.training_dataset).strip() if self.training_dataset else None


class EvaluationRow(BaseModel):
    """Проверенная строка датасета с эталонным набором категорий."""

    inp: PipelineInput
    expected: set[McId]


def resolve_path(base: Path, value: str) -> Path:
    """Разрешить путь из конфига независимо от текущей рабочей папки."""
    path = Path(value).expanduser()
    return (base / path).resolve()


def read_records(table: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Прочитать строки датасета из таблицы PostgreSQL в едином формате.

    Возвращает dict со строкой, совместимый с ``prepare_rows``: ключи
    ``description``, ``mcId`` и ``found_microcategories`` (список
    ``{"mcId": int, "label": str}``).
    """
    try:
        return database_read_records(table, limit=limit)
    except DatabaseError as exc:
        raise ValueError(f"Не удалось прочитать таблицу {table}: {exc}") from exc


def prepare_rows(records: list[dict[str, Any]], source_mc_id: int) -> list[EvaluationRow]:
    """Сопоставить названия официальных категорий с общей таблицей mcId."""
    names = {name: mc_id for mc_id, name in MICROCATEGORIES.items()}
    rows = []
    for index, record in enumerate(records, start=1):
        try:
            inp = PipelineInput.model_validate({
                "description": record["description"],
                "mcId": record.get("mcId", source_mc_id),
            })
            if "categories" in record and "found_microcategories" not in record:
                labels = record["categories"]
                if not isinstance(labels, list) or any(
                    not isinstance(label, str) for label in labels
                ):
                    raise ValueError("categories должен быть списком названий категорий")
                expected = {names[label] for label in labels}
            else:
                labels = record["found_microcategories"]
                if not isinstance(labels, list):
                    raise ValueError("found_microcategories должен быть списком объектов")
                expected = {label["mcId"] for label in labels}
            row = EvaluationRow(inp=inp, expected=expected)
            row.expected.discard(inp.mc_id)
            rows.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Некорректная строка датасета {index}: {exc}") from exc
    if not rows:
        raise ValueError("Датасет пуст")
    return rows


def normalize_description(text: str) -> str:
    """Нормализовать описания для проверки пересечения с обучающими данными."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def audit_overlap(rows: list[EvaluationRow], training_table: str | None) -> dict[str, Any]:
    """Проверить точное нормализованное пересечение, не удаляя строки оценки."""
    if training_table is None:
        return {"checked": False}
    if training_table not in DATABASE_TABLES:
        raise ValueError(f"Неизвестная таблица обучения: {training_table}")
    descriptions = {
        normalize_description(record["description"])
        for record in read_records(training_table)
    }
    indices = [
        index
        for index, row in enumerate(rows, start=1)
        if normalize_description(row.inp.description) in descriptions
    ]
    if indices:
        logger.warning(
            "Пересечение с train: %d/%d строк. Метрика может быть завышена; "
            "строки не исключены. Проверены только нормализованные совпадения.",
            len(indices), len(rows),
        )
    return {
        "checked": True,
        "method": "normalized_exact_description",
        "training_dataset": training_table,
        "overlapping_rows": len(indices),
        "row_numbers": indices,
    }


def calculate_metrics(
    expected: list[set[int]], predicted: list[set[int]]
) -> dict[str, int | float]:
    """Посчитать micro-F1 через sklearn и баллы по правилам хакатона."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("Нужны непустые списки эталонов и предсказаний одинаковой длины")
    if any(ids - MICROCATEGORIES.keys() for ids in expected + predicted):
        raise ValueError("Неизвестный mcId при расчёте метрики")
    binarizer = MultiLabelBinarizer(classes=sorted(MICROCATEGORIES))
    truth = binarizer.fit_transform(expected)
    output = binarizer.transform(predicted)
    micro_f1 = float(f1_score(truth, output, average="micro", zero_division=0))
    tp = sum(len(truth_ids & output_ids) for truth_ids, output_ids in zip(expected, predicted))
    fp = sum(len(output_ids - truth_ids) for truth_ids, output_ids in zip(expected, predicted))
    fn = sum(len(truth_ids - output_ids) for truth_ids, output_ids in zip(expected, predicted))
    return {
        "micro_f1": micro_f1,
        "classification_points": CLASSIFICATION_POINTS * micro_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate(config_path: Path) -> dict[str, Any]:
    """Прогнать оба шага решения и сохранить отчёт с каждым ответом."""
    config_path = config_path.resolve()
    config = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    pipeline_path = resolve_path(base, config.pipeline_config)
    pipeline_config = load_config(pipeline_path)
    # Проверяем выбор всех шагов до чтения данных и потенциально платных вызовов.
    for service, module in pipeline_config.model_dump().items():
        if module is None:
            raise ValueError(
                f"Для раздела {service} не выбран модуль в {pipeline_path}; "
                "оценка полного решения требует оба шага, без заглушек"
            )
    dataset = config.dataset_table
    if dataset not in DATABASE_TABLES:
        raise ValueError(f"Неизвестная таблица датасета: {dataset}")
    records = read_records(dataset)
    if config.limit is not None:
        records = records[:config.limit]
    rows = prepare_rows(records, config.source_mc_id)
    training_table = config.training_table
    if training_table is not None and training_table not in DATABASE_TABLES:
        raise ValueError(f"Неизвестная таблица обучения: {training_table}")
    report_path = resolve_path(base, config.report)
    if report_path in {pipeline_path, config_path}:
        raise ValueError("Путь отчёта не должен совпадать с входными файлами")
    cost_path = Path(str(report_path) + ".costs.json")
    if cost_path in {pipeline_path, config_path}:
        raise ValueError("Отчёт стоимости не должен совпадать с входными файлами")
    costs = LLMCostLedger()
    overlap = audit_overlap(rows, training_table)
    predictions: list[set[int]] = []
    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        try:
            with track_llm_costs(costs):
                output = run_pipeline(row.inp, pipeline_config)
        except Exception:
            cost_path.parent.mkdir(parents=True, exist_ok=True)
            cost_path.write_text(json.dumps(costs.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
            logger.exception("Ошибка решения на строке %d; итоговая метрика не рассчитана", index)
            raise
        # Дубли не дают TP; исходная категория, ошибочно выданная решением, даёт FP.
        predicted = {draft.mc_id for draft in output.drafts}
        predictions.append(predicted)
        results.append({
            "row_number": index,
            "expected": sorted(row.expected),
            "predicted": sorted(predicted),
            "output": output.model_dump(by_alias=True),
        })
        if index % config.log_every == 0 or index == len(rows):
            logger.info("Обработано %d/%d", index, len(rows))
    elapsed = time.perf_counter() - started
    metrics = calculate_metrics([row.expected for row in rows], predictions)
    report = {
        "dataset": dataset,
        "dataset_sha256": table_sha256(dataset),
        "training_dataset": training_table,
        "run_config": config.model_dump(),
        "pipeline_config": pipeline_config.model_dump(),
        "rows": len(rows),
        "limited": config.limit is not None,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "seconds_per_row": elapsed / len(rows),
        "overlap": overlap,
        "costs": costs.summary(),
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "micro-F1: %.6f; баллы: %.4f / %d; TP=%d FP=%d FN=%d; отчёт: %s",
        metrics["micro_f1"], metrics["classification_points"], CLASSIFICATION_POINTS,
        metrics["tp"], metrics["fp"], metrics["fn"], report_path,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Запустить оценку с параметрами из JSON-конфига."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG)
    args = parser.parse_args(argv)
    try:
        evaluate(args.config)
    except (ValueError, OSError, ImportError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
