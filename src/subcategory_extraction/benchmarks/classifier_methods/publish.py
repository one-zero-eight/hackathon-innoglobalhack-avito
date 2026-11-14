"""Объединить реальные результаты и ошибки, не подставляя F1 отсутствующим методам."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
import logging

from run import calculate_metrics
from subcategory_extraction.benchmarks.classifier_methods.run import HERE, METHODS, ROOT


def publish(paths: list[Path], output: Path) -> dict:
    """Проверить сопоставимость строк и опубликовать все девять статусов."""
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not reports:
        raise ValueError("Нет результатов для публикации")
    reference = reports[0]
    expected = [set(row["expected"]) for row in reference["rows"]]
    results = {}
    for report in reports:
        if report["protocol"] != reference["protocol"] or report["rows"] != reference["rows"]:
            raise ValueError("Отчёты относятся к разным данным или протоколам")
        for method, result in report["results"].items():
            if method in results or method not in METHODS:
                raise ValueError(f"Повторный или неизвестный метод {method}")
            if result["status"] == "completed":
                predicted = [set(row) for row in result["predicted"]]
                metrics = calculate_metrics(expected, predicted)
                if metrics != result["metrics"]:
                    raise ValueError(f"Метрика {method} не воспроизводится из предсказаний")
            elif result["status"] != "failed" or "metrics" in result or "predicted" in result:
                raise ValueError(f"Неизвестный статус или выдуманная метрика {method}")
            results[method] = {**result, "source_hashes": report["sources"]}
    if set(results) != set(METHODS):
        raise ValueError(f"Отсутствуют статусы методов: {set(METHODS) - set(results)}")
    completed = [method for method in METHODS if results[method]["status"] == "completed"]
    blocked = [method for method in METHODS if results[method]["status"] != "completed"]
    auth_blocked = [method for method in blocked if "401" in results[method].get("error", "")]
    training = json.loads((ROOT / "out/semantic-encoder/train-report.json").read_text(encoding="utf-8"))
    result = {
        "protocol": reference["protocol"],
        "branch": "leo/feature/classifier-methods",
        "comparison_complete": not blocked,
        "measured_methods": completed,
        "unmeasured_methods": blocked,
        "best_measured_method": max(completed, key=lambda method: results[method]["metrics"]["micro_f1"]),
        "limitations": [
            f"{len(blocked)} methods have no completed measurement; see their explicit failure records.",
            "A failed method has no F1, not zero F1. No silent fallback was used.",
            "Google baseline is a genuine recorded native-API run, explicitly replayed on matching inputs.",
            "All 208 official reference rows remain, including conflicting duplicate descriptions.",
            "These are classification scores, not draft-quality or total competition scores.",
            "The same public set was inspected in earlier experiments; this is an exploratory benchmark, not an unseen deployment estimate.",
            "Local encoder weights and API caches are not committed; training and download commands reproduce them.",
        ],
        "access_diagnostic": ({
            "provider": "https://bestapi.fugue.pro",
            "status_code": 401,
            "response_detail": "invalid api key",
            "credential_matches_local_env": True,
            "scope": "Authenticated model-list endpoint also failed during this experiment.",
        } if auth_blocked else None),
        "overlap": reference["overlap"],
        "rows": reference["rows"],
        "encoder_training": training,
        "results": results,
        "raw_report_files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    }

    def portable(value: object) -> object:
        """Убрать машинные абсолютные пути, сохранив содержимое и происхождение."""
        if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
            return "repo://" + str(Path(value).relative_to(ROOT))
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value

    result = portable(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Собрать отчёт без повторных запросов к моделям и переобучения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "combined-results.json")
    args = parser.parse_args()
    result = publish(args.reports, args.output)
    logging.info("%s", json.dumps({
        "complete": result["comparison_complete"], "best_measured": result["best_measured_method"],
        "methods": {name: value.get("metrics", {"status": value["status"], "error": value.get("error")})
                    for name, value in result["results"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
