"""Общие данные, замеры времени и отчёты бенчмарков без расчёта стоимости."""

import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
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
    """Взять фиксированные validation-группы, сохранив прежнюю выборку."""
    from random import Random
    from sklearn.model_selection import train_test_split

    rows = load_training_groups(TRAIN)
    _, remaining = train_test_split(list(range(len(rows))), test_size=0.3, random_state=42)
    validation, _ = train_test_split(remaining, test_size=0.5, random_state=43)
    selected = [rows[index] for index in validation if not nonempty or rows[index]["targets"]]
    if not 1 <= size <= len(selected):
        raise ValueError(f"Размер выборки должен быть от 1 до {len(selected)}")
    return Random(seed).sample(selected, size)


def load_training_groups(path: Path) -> list[dict[str, Any]]:
    """Прочитать официальный train и объединить точные нормализованные дубликаты."""
    from main import MICROCATEGORIES, PipelineInput
    from run import normalize_description

    groups: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        inp = PipelineInput(description=row["description"], mcId=101)
        targets = {target["mcId"] for target in row["found_microcategories"]}
        if targets - MICROCATEGORIES.keys() or inp.mc_id in targets:
            raise ValueError("Некорректные целевые категории в официальном train")
        key = normalize_description(inp.description)
        if key in groups and groups[key]["targets"] != targets:
            raise ValueError("Конфликт разметки одинаковых описаний")
        groups.setdefault(key, {
            "inp": inp, "targets": targets,
            "group_id": hashlib.sha256(key.encode()).hexdigest(),
        })
    return list(groups.values())


def setup() -> None:
    """Загрузить окружение проекта без вывода секретов."""
    load_dotenv(ROOT / ".env")


def compare_contract_revisions(script: Path, paths: list[str], module_path: str) -> list[dict[str, Any]]:
    """Проверить один контракт на Git-версиях в изолированных процессах."""
    revisions = subprocess.check_output(["git", "log", "--format=%H", "--", *paths], cwd=ROOT, text=True).splitlines()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    results = []
    for revision in dict.fromkeys([head, *revisions]):
        present = subprocess.run(["git", "cat-file", "-e", f"{revision}:{module_path}"], cwd=ROOT, capture_output=True)
        if present.returncode:
            results.append({"commit": revision, "status": "unavailable", "reason": "Выбранный модуль отсутствует"})
            continue
        with TemporaryDirectory(prefix="pipeline-contract-") as folder:
            root = Path(folder)
            entries = subprocess.check_output([
                "git", "ls-tree", "--name-only", revision, "--", "src", "config.json", "data/input.json",
            ], cwd=ROOT, text=True).splitlines()
            archive = subprocess.check_output(["git", "archive", revision, *entries], cwd=ROOT)
            with tarfile.open(fileobj=BytesIO(archive)) as tar:
                tar.extractall(root, filter="data")
            report_path = root / "contract.json"
            process = subprocess.run([
                sys.executable, str(script.resolve()), "--source-root", str(root / "src"),
                "--report", str(report_path),
            ], cwd=ROOT, capture_output=True, text=True)
            if report_path.exists():
                result = json.loads(report_path.read_text())["current"]
            else:
                result = {"status": "failed", "reason": f"Процесс бенчмарка не создал отчёт; код {process.returncode}"}
            results.append({"commit": revision, **result})
    return results
