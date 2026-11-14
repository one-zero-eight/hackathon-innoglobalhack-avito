"""Общие данные, замеры времени и отчёты бенчмарков без расчёта стоимости."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "docs/innoglobalhack_avito/public-train-annotated.jsonl"


def metadata() -> dict[str, Any]:
    """Зафиксировать дату, commit и отпечаток исполняемого кода с промптами."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix not in (".py", ".md", ".json") or "dataset_annotation" in path.parts:
            continue
        if path.suffix in (".md", ".json") and "prompts" not in path.parts:
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
    """Записать отчёт атомарно, без вычисления или подстановки цены."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validation_sample(size: int, seed: int = 43, nonempty: bool = False) -> list[dict]:
    """Взять фиксированные группы из validation того же разбиения, что TF-IDF."""
    from random import Random
    from sklearn.model_selection import train_test_split
    from subcategory_extraction.benchmarks.tfidf_quality.run import load_rows

    rows = load_rows(TRAIN)
    _, remaining = train_test_split(list(range(len(rows))), test_size=0.3, random_state=42)
    validation, _ = train_test_split(remaining, test_size=0.5, random_state=43)
    selected = [rows[index] for index in validation if not nonempty or rows[index]["targets"]]
    if not 1 <= size <= len(selected):
        raise ValueError(f"Размер выборки должен быть от 1 до {len(selected)}")
    return Random(seed).sample(selected, size)


def setup() -> None:
    """Загрузить окружение проекта без вывода секретов."""
    load_dotenv(ROOT / ".env")
