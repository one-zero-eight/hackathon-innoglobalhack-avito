"""Офлайн-контракт выделения: тот же сценарий на текущем и прежнем коде."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
import logging
import subprocess
import tarfile
from io import BytesIO
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable
from unittest.mock import patch

from benchmark_support import ROOT, metadata, save_report


def require(condition: bool, message: str) -> None:
    """Остановить контрольный случай при нарушении контракта."""
    if not condition:
        raise AssertionError(message)


def check_contract(source_root: Path) -> dict[str, Any]:
    """Проверить сервис через публичный loader с подменой только ответа API."""
    sys.path.insert(0, str(source_root))
    try:
        from main import MICROCATEGORIES, PipelineConfig, PipelineInput, load_pipeline
        config = PipelineConfig(subcategory_extraction="google_decision", draft_generation=None)
        with patch.dict("os.environ", {
            "EXTRACTION_PROMPT": "guided", "EXTRACTION_THRESHOLD": "0.5",
            "GOOGLE_DECISION_PROMPT": "guided", "GOOGLE_DECISION_THRESHOLD": "0.5",
        }):
            pipeline = load_pipeline("subcategory_extraction", config)
    except (ImportError, AttributeError, ValueError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    cases: list[dict[str, Any]] = []
    inp = PipelineInput(description="Устанавливаю раковины и меняю смесители.", mcId=101)

    def case(name: str, operation: Callable[[], None]) -> None:
        """Сохранить результат каждого случая, включая прежние нарушения."""
        try:
            operation()
        except Exception as exc:
            cases.append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        else:
            cases.append({"name": name, "status": "passed"})

    def response(source: int = 101, **values: Any) -> dict[str, Any]:
        """Построить typed answers для всех категорий, кроме исходной."""
        return {"answers": {
            f"mc{cid}": {"type": "noul", "noul": values.get(str(cid), 0.0), "confidence": 0.5}
            for cid in MICROCATEGORIES if cid != source
        }, "usage": {"total_input_tokens": 42}}

    def output(raw: Any, expected: list[int]) -> None:
        """Проверить выход сервиса и ровно один вызов API."""
        with patch.object(pipeline.client, "complete", return_value=json.dumps(raw)) as complete:
            actual = pipeline.run(inp)
        require(actual == expected, f"Ожидались категории {expected}, получены {actual}")
        require(complete.call_count == 1, "Ожидался один запрос для всех категорий")

    case("no_targets", lambda: output(response(), []))
    case("multiple_targets_and_threshold_boundary", lambda: output(response(**{"102": 0.5, "104": 0.99, "105": 0.499}), [102, 104]))
    reversed_response = response(**{"102": 0.8, "104": 0.8, "106": 0.8})
    reversed_response["answers"] = dict(reversed(list(reversed_response["answers"].items())))
    case("deterministic_order", lambda: output(reversed_response, [102, 104, 106]))

    def check_source(source: int) -> None:
        """Проверить поддержку каждой исходной категории и запрет её запроса."""
        value = PipelineInput(description=inp.description, mcId=source)
        raw = response(source)
        for answer in raw["answers"].values():
            answer["noul"] = 0.9
        with patch.object(pipeline.client, "complete", return_value=json.dumps(raw)) as complete:
            actual = pipeline.run(value)
        expected = sorted(set(MICROCATEGORIES) - {source})
        require(actual == expected, "Потеряна категория или возвращена исходная")
        require(complete.call_count == 1, "Ожидался один запрос")
        questions = complete.call_args.kwargs["questions"]
        require(set(questions) == {f"mc{cid}" for cid in expected}, "Запрошен неверный набор категорий")
        require(complete.call_args.args[0] == inp.description, "Описание изменено перед API")
        require(all(q["type"] == "noul" and MICROCATEGORIES[int(key[2:])] in q["instructions"] for key, q in questions.items()), "Вопросы потеряли тип или категорию")

    for source in MICROCATEGORIES:
        case(f"source_{source}", lambda source=source: check_source(source))

    def empty(description: str) -> None:
        """Пустое описание не содержит предложений услуг и не требует API."""
        with patch.object(pipeline.client, "complete", return_value=json.dumps(response())) as complete:
            require(pipeline.run(PipelineInput(description=description, mcId=101)) == [], "Пустое описание породило категории")
        require(not complete.called, "Пустое описание вызвало API")

    case("empty_description", lambda: empty(""))
    case("whitespace_description", lambda: empty(" \n\t "))

    def rejects(raw: Any, literal: bool = False) -> None:
        """Проверить отказ на неверный ответ без возврата пустого предсказания."""
        with patch.object(pipeline.client, "complete", return_value=raw if literal else json.dumps(raw)):
            try:
                pipeline.run(inp)
            except (ValueError, TypeError):
                return
        raise AssertionError("Неверный ответ принят")

    for name, raw in (("not_json", "not JSON"), ("html", "<!doctype html>")):
        case(name, lambda raw=raw: rejects(raw, literal=True))
    for name, raw in (("array", []), ("null", None), ("missing_answers", {}), ("answers_array", {"answers": []}), ("empty_answers", {"answers": {}})):
        case(name, lambda raw=raw: rejects(raw))
    for name, value in (("boolean", True), ("string", "0.9"), ("negative", -0.1), ("too_large", 1.1), ("nan", float("nan")), ("infinity", float("inf")), ("null_probability", None)):
        raw = response()
        raw["answers"]["mc104"]["noul"] = value
        case(name, lambda raw=raw: rejects(raw))
    for name, answer in (("missing_type", {"noul": 0.9}), ("wrong_type", {"type": "choice", "noul": 0.9}), ("missing_probability", {"type": "noul"}), ("null_answer", None), ("string_confidence", {"type": "noul", "noul": 0.9, "confidence": "0.8"}), ("invalid_confidence", {"type": "noul", "noul": 0.9, "confidence": 2})):
        raw = response()
        raw["answers"]["mc104"] = answer
        case(name, lambda raw=raw: rejects(raw))
    raw = response()
    del raw["answers"]["mc104"]
    case("missing_category", lambda: rejects(raw))
    for category in ("mc101", "mc999"):
        raw = response()
        raw["answers"][category] = {"type": "noul", "noul": 0.9}
        case(f"extra_{category}", lambda raw=raw: rejects(raw))

    def subset(categories: set[int], expected: set[str]) -> None:
        """Проверить подмножество вопросов, включая исходную категорию."""
        raw = {"answers": {key: {"type": "noul", "noul": 0.8} for key in expected}}
        with patch.object(pipeline.client, "complete", return_value=json.dumps(raw)) as complete:
            actual = pipeline.probabilities(inp, categories)
        require(set(actual) == {int(key[2:]) for key in expected}, "Подмножество изменено")
        require(complete.called == bool(expected), "Лишний запрос при пустом подмножестве")
        if expected:
            require(set(complete.call_args.kwargs["questions"]) == expected, "Лишние вопросы")

    case("empty_subset", lambda: subset(set(), set()))
    case("source_only_subset", lambda: subset({101}, set()))
    case("selected_subset", lambda: subset({101, 104}, {"mc104"}))

    def bad_subset(categories: set[int]) -> None:
        """Не отправлять заведомо неверные категории в API."""
        with patch.object(pipeline.client, "complete") as complete:
            try:
                pipeline.probabilities(inp, categories)
            except ValueError:
                require(not complete.called, "Неверное подмножество вызвало API")
                return
        raise AssertionError("Неверное подмножество принято")

    case("unknown_subset", lambda: bad_subset({999}))
    case("boolean_subset", lambda: bad_subset({True}))

    def bad_setting(key: str, value: str) -> None:
        """Проверить настройки до обращения к API у обеих версий интерфейса."""
        legacy = "GOOGLE_DECISION_THRESHOLD" if key == "EXTRACTION_THRESHOLD" else "GOOGLE_DECISION_PROMPT"
        with patch.dict("os.environ", {key: value, legacy: value}):
            try:
                load_pipeline("subcategory_extraction", config)
            except ValueError:
                return
        raise AssertionError("Неверная настройка принята")

    for value in ("-0.1", "1.1", "nan", "inf", "bad"):
        case(f"threshold_{value}", lambda value=value: bad_setting("EXTRACTION_THRESHOLD", value))
    case("unknown_prompt", lambda: bad_setting("EXTRACTION_PROMPT", "../unknown"))

    def baseline_prompt_case() -> None:
        """Убедиться, что второй выбираемый промпт формирует рабочие вопросы."""
        with patch.dict("os.environ", {
            "EXTRACTION_PROMPT": "baseline", "EXTRACTION_THRESHOLD": "0.5",
            "GOOGLE_DECISION_PROMPT": "baseline", "GOOGLE_DECISION_THRESHOLD": "0.5",
        }):
            alternative = load_pipeline("subcategory_extraction", config)
        with patch.object(alternative.client, "complete", return_value=json.dumps(response(**{"104": 0.9}))) as complete:
            require(alternative.run(inp) == [104], "Промпт baseline не сформировал ожидаемый запрос")
        require(complete.call_count == 1, "baseline выполнил лишние запросы")

    case("baseline_prompt", baseline_prompt_case)

    def metric_case() -> None:
        """Проверить официальный вес классификации и известные TP/FP/FN."""
        from subcategory_extraction.benchmarks.metrics import score
        result = score([{102, 105}], [{102, 104}])
        require(result["micro_f1"] == 0.5 and result["classification_points"] == 30, "Неверная формула или вес классификации")
        require((result["tp"], result["fp"], result["fn"]) == (1, 1, 1), "Неверные TP/FP/FN")
        require(result["by_category"][102]["precision"] == 1 and result["by_category"][105]["recall"] == 0, "Неверные метрики категории")

    def empty_metric_case() -> None:
        """Пустые наборы меток не должны давать идеальный F1."""
        from subcategory_extraction.benchmarks.metrics import score
        require(score([set()], [set()])["micro_f1"] == 0, "Пустым ответам присвоен ненулевой F1")
        require(score([{102}], [{101, 102}])["fp"] == 1, "Исходная категория не учтена как FP")

    def no_rows_metric_case() -> None:
        """Не оценивать отсутствующий датасет как завершённый прогон."""
        from subcategory_extraction.benchmarks.metrics import score
        try:
            score([], [])
        except ValueError:
            return
        raise AssertionError("Пустой датасет принят")

    case("official_classification_metrics", metric_case)
    case("empty_labels_and_source_false_positive", empty_metric_case)
    case("empty_dataset_rejected", no_rows_metric_case)

    files = [source_root / "main.py", *sorted((source_root / "subcategory_extraction/google_decision").rglob("*"))]
    digest = hashlib.sha256()
    for path in files:
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            digest.update(str(path.relative_to(source_root)).encode())
            digest.update(path.read_bytes())
    passed = sum(row["status"] == "passed" for row in cases)
    return {"status": "complete" if passed == len(cases) else "failed", "passed": passed,
            "total": len(cases), "implementation_sha256": digest.hexdigest(), "cases": cases}


def main() -> int:
    """Сравнить контракт на версиях без запуска настоящих моделей."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-revisions", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT / "src")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    started = perf_counter()
    current = check_contract(args.source_root)
    report = {**metadata(), "scenario": "offline_extraction_contract", "real_api_calls": 0,
              "current": current, "versions": [], "status": current["status"]}
    if args.all_revisions:
        revisions = subprocess.check_output([
            "git", "log", "--format=%H", "--", "src/main.py", "src/subcategory_extraction/google_decision",
        ], cwd=ROOT, text=True).splitlines()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        for revision in dict.fromkeys([head, *revisions]):
            present = subprocess.run(["git", "cat-file", "-e", f"{revision}:src/subcategory_extraction/google_decision/pipeline.py"], cwd=ROOT, capture_output=True)
            if present.returncode:
                report["versions"].append({"commit": revision, "status": "unavailable", "reason": "Модуль google_decision отсутствует"})
                continue
            with TemporaryDirectory(prefix="extraction-contract-") as folder:
                root = Path(folder)
                archive = subprocess.check_output(["git", "archive", revision, "src"], cwd=ROOT)
                with tarfile.open(fileobj=BytesIO(archive)) as tar:
                    tar.extractall(root, filter="data")
                child_report = root / "contract.json"
                process = subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), "--source-root", str(root / "src"),
                    "--report", str(child_report),
                ], cwd=ROOT, capture_output=True, text=True)
                if not child_report.exists():
                    raise RuntimeError(f"Не создан отчёт версии {revision}: код {process.returncode}")
                row = json.loads(child_report.read_text())["current"]
                report["versions"].append({"commit": revision, **row})
    report["offline_elapsed_seconds"] = perf_counter() - started
    save_report(args.report, report)
    logging.info("Текущий контракт: %s/%s, %s", current.get("passed", 0), current.get("total", 0), current["status"])
    for row in report["versions"]:
        logging.info("%s: %s, %s/%s", row["commit"][:12], row["status"], row.get("passed", 0), row.get("total", 0))
    return 0 if current["status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
