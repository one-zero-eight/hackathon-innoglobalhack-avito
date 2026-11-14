"""Общие данные, замеры времени и отчёты бенчмарков без расчёта стоимости."""

from collections import Counter
import hashlib
import json
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv

from database import read_records as database_read_records, table_sha256

ROOT = Path(__file__).resolve().parent.parent
# Официальный train живёт в PostgreSQL, а не в файле.
TRAIN_TABLE = "case_train"


def metadata() -> dict[str, Any]:
    """Зафиксировать дату, commit и отпечаток исполняемого кода с промптами."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix not in (".py", ".md") or "dataset_annotation" in path.parts:
            continue
        if path.suffix == ".md" and "prompts" not in path.parts:
            continue
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return {
        "date": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": digest.hexdigest(),
        "working_tree": True,
    }


def timing(seconds: list[float]) -> dict[str, int | float]:
    """Посчитать время только завершённых вызовов; повторы входят в задержку."""
    if not seconds:
        return {"calls": 0}
    return {
        "calls": len(seconds), "total_seconds": sum(seconds),
        "mean_seconds": mean(seconds), "median_seconds": median(seconds),
        "p95_seconds": float(np.percentile(seconds, 95)),
        "min_seconds": min(seconds), "max_seconds": max(seconds),
    }


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Обновить строки решений и атомарно записать отчёт без расчёта цены."""
    if path.is_file() and isinstance(report.get("results"), dict):
        previous = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(previous.get("results"), dict):
            for key in ("scenario", "dataset_sha256", "sample_seed", "sample_size"):
                if key in previous and key in report and previous[key] != report[key]:
                    raise ValueError(f"Нельзя объединить разные протоколы: {key}; выберите другой --report")
            before = previous.get("protocol", {}).get("dataset_sha256", previous.get("dataset_sha256"))
            after = report.get("protocol", {}).get("dataset_sha256", report.get("dataset_sha256"))
            if before is not None and after is not None and before != after:
                raise ValueError("Нельзя объединить результаты разных датасетов; выберите другой --report")
            retained = {}
            for name, result in previous["results"].items():
                if name in report["results"]:
                    continue
                result = dict(result)
                result.setdefault("measurement_metadata", {
                    key: previous[key] for key in (
                        "date", "commit", "code_sha256", "working_tree", "mode", "protocol",
                        "timing_scope", "sample_seed", "sample_size", "machine", "environment",
                    )
                    if key in previous
                })
                retained[name] = result
            report["results"] = {**retained, **report["results"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validation_sample(size: int, seed: int = 43, nonempty: bool = False) -> list[dict]:
    """Взять фиксированные группы из validation того же разбиения, что TF-IDF."""
    from random import Random
    from sklearn.model_selection import train_test_split
    from main import PipelineInput

    groups: dict[str, dict] = {}
    for row in database_read_records(TRAIN_TABLE):
        inp = PipelineInput(description=row["description"], mcId=101)
        targets = {target["mcId"] for target in row["found_microcategories"]}
        key = " ".join(unicodedata.normalize("NFKC", inp.description).casefold().replace("ё", "е").split())
        if key in groups and groups[key]["targets"] != targets:
            raise ValueError("Конфликт меток одинаковых описаний")
        groups.setdefault(key, {
            "inp": inp, "targets": targets,
            "group_id": hashlib.sha256(key.encode()).hexdigest(),
        })
    rows = list(groups.values())
    _, remaining = train_test_split(list(range(len(rows))), test_size=0.3, random_state=42)
    validation, _ = train_test_split(remaining, test_size=0.5, random_state=43)
    selected = [rows[index] for index in validation if not nonempty or rows[index]["targets"]]
    if not 1 <= size <= len(selected):
        raise ValueError(f"Размер выборки должен быть от 1 до {len(selected)}")
    return Random(seed).sample(selected, size)


def setup() -> None:
    """Загрузить окружение проекта без вывода секретов."""
    load_dotenv(ROOT / ".env")


def replay_recorded_report(directory: Path, scenario: str, report_path: Path) -> None:
    """Пересчитать сохранённые предсказания, не выдавая replay за новый inference."""
    from main import MICROCATEGORIES
    from run import calculate_metrics, prepare_rows, read_records

    source = directory / "recorded-results.json"
    if source.resolve() == report_path.resolve():
        raise ValueError("Нельзя перезаписать исходную запись")
    recorded = json.loads(source.read_text(encoding="utf-8"))
    if scenario not in ("classifier_methods", "improved_dataset"):
        raise ValueError(f"Неизвестный F1-сценарий: {scenario}")
    # Датасеты живут в PostgreSQL: classifier_methods — case_test,
    # improved_dataset — ручная разметка manual_test.
    dataset_table = "case_test" if scenario == "classifier_methods" else "manual_test"
    records = read_records(dataset_table)
    source_hashes = {
        record.get("_source_sha256") for record in records if record.get("_source_sha256")
    }
    if len(source_hashes) != 1:
        raise ValueError(f"Таблица {dataset_table}: ожидается один source_sha256, получено {len(source_hashes)}")
    source_hash = next(iter(source_hashes))
    expected_hash = recorded["protocol"]["dataset_sha256"]
    if source_hash != expected_hash:
        raise ValueError(f"Таблица {dataset_table}: исходный файл изменился относительно сохранённого отчёта")
    actual_rows = prepare_rows(records, 101)
    expected_rows = Counter(
        (hashlib.sha256(row.inp.description.encode()).hexdigest(), tuple(sorted(row.expected)))
        for row in actual_rows
    )
    saved_rows = Counter(
        (row["description_sha256"], tuple(sorted(row["expected"]))) for row in recorded["rows"]
    )
    if saved_rows != expected_rows:
        raise ValueError("Сохранённые строки не покрывают текущий датасет с учётом дубликатов")
    report = {
        **metadata(), "status": "running", "scenario": scenario,
        "mode": "offline_recalculation_of_recorded_predictions",
        "storage": "postgresql",
        "record_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "recorded_at": recorded.get("created_at_utc", recorded.get("measured_at_utc")),
        "dataset_table": dataset_table,
        "dataset_sha256": source_hash,
        "table_sha256": table_sha256(dataset_table),
        "fresh_model_calls": 0, "fresh_api_requests": 0,
        "model_response_time": None,
        "timing_scope": "no fresh inference; historical timing remains in recorded-results.json",
        "results": {},
    }

    def check(name: str, rows: list[dict], predictions: list, saved: dict) -> None:
        """Проверить происхождение эталонов и точное совпадение пересчитанной метрики."""
        if len(rows) != len(predictions) or not rows:
            raise ValueError(f"{name}: неполное число предсказаний")
        for row, prediction in zip(rows, predictions, strict=True):
            if any(type(cid) is not int or cid not in MICROCATEGORIES or cid == 101
                   for cid in prediction) or len(prediction) != len(set(prediction)):
                raise ValueError(f"{name}: неверные категории")
        metric = calculate_metrics(
            [set(row["expected"]) for row in rows],
            [set(prediction) for prediction in predictions],
        )
        if any(abs(metric[key] - saved[key]) > 1e-12 for key in ("micro_f1", "tp", "fp", "fn")):
            raise ValueError(f"{name}: метрика не совпала с исходным отчётом")
        report["results"][name] = {
            "status": "complete", "rows": len(rows), "metrics": metric,
            "recorded_metadata": recorded["results"][name].get("metadata"),
        }

    try:
        for name, result in recorded["results"].items():
            if result["status"] != "completed":
                report["results"][name] = {"status": "unavailable", "reason": result.get("error")}
                continue
            check(name, recorded["rows"], result["predicted"], result["metrics"])
        report["status"] = (
            "complete" if all(item["status"] == "complete" for item in report["results"].values())
            else "failed"
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        report.pop("results", None)
        save_report(report_path, report)
        raise
    save_report(report_path, report)
