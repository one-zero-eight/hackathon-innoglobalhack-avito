"""Сравнение Jev, TF-IDF и фиксированного гибрида без validation и подбора на test."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from main import MICROCATEGORIES, PipelineConfig, load_pipeline
from run import audit_overlap, calculate_metrics, prepare_rows, read_records
from subcategory_extraction.benchmarks.train_tfidf import ROOT
from subcategory_extraction.jev.pipeline import DEFAULT_PROMPT_PATH
from subcategory_extraction.tfidf.pipeline import DEFAULT_ARTIFACT

logger = logging.getLogger(__name__)
DATASET = ROOT / "docs/innoglobalhack_avito/public-test-dataset.parquet"
MODULES = ("jev", "tfidf", "jev_tfidf")
FIXED_THRESHOLD = 0.5
FIXED_JEV_WEIGHT = 0.5


def compare(dataset: Path, artifact: Path, training_dataset: Path, report_path: Path) -> dict:
    """Оценить реальные модули на одинаковых строках, сохраняя все дубликаты."""
    load_dotenv(ROOT / ".env")
    os.environ["TFIDF_ARTIFACT_PATH"] = str(artifact.resolve())
    os.environ["JEV_THRESHOLD"] = str(FIXED_THRESHOLD)
    os.environ["JEV_TFIDF_THRESHOLD"] = str(FIXED_THRESHOLD)
    os.environ["JEV_TFIDF_WEIGHT"] = str(FIXED_JEV_WEIGHT)
    if report_path.resolve() in {dataset.resolve(), artifact.resolve(), training_dataset.resolve()}:
        raise ValueError("Отчёт не должен перезаписывать входные файлы")
    pipelines = {
        name: load_pipeline("subcategory_extraction", PipelineConfig(
            subcategory_extraction=name, draft_generation=None,
        ))
        for name in MODULES
    }
    if pipelines["tfidf"].threshold != FIXED_THRESHOLD:
        raise ValueError("Нужен артефакт с заранее фиксированным порогом 0.5")
    rows = prepare_rows(read_records(dataset), 101)
    overlap = audit_overlap(rows, training_dataset)
    expected = [row.expected for row in rows]
    predictions: dict[str, list[set[int]]] = {name: [] for name in MODULES}
    durations = {name: 0.0 for name in MODULES}
    api_requests = {name: 0 for name in MODULES}
    cache_hits = {name: 0 for name in MODULES}
    results = []
    started = perf_counter()
    for index, row in enumerate(rows, start=1):
        outputs = {}
        for name, pipeline in pipelines.items():
            before = perf_counter()
            output = pipeline.run(row.inp)
            durations[name] += perf_counter() - before
            if not isinstance(output, list) or any(
                type(cid) is not int or cid not in MICROCATEGORIES or cid == row.inp.mc_id
                for cid in output
            ):
                raise ValueError(f"Модуль {name} вернул неверные категории в строке {index}")
            predicted = set(output)
            predictions[name].append(predicted)
            outputs[name] = sorted(predicted)
            client = pipeline.client if name == "jev" else pipeline.jev.client if name == "jev_tfidf" else None
            if client is not None:
                cache_hits[name] += int(client.last_cache_hit)
                api_requests[name] += int(not client.last_cache_hit)
        results.append({"row_number": index, "expected": sorted(row.expected), "predicted": outputs})
        if index % 10 == 0 or index == len(rows):
            logger.info("Обработано %d/%d", index, len(rows))
    prompt_path = Path(os.getenv("JEV_PROMPT_PATH", str(DEFAULT_PROMPT_PATH))).expanduser()
    report = {
        "scope": "subcategory_extraction_only; no draft-generation or full-competition score",
        "base_revision": "cd0f310",
        "rows": len(rows),
        "source_mc_id_assumption": 101,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "prompt": str(prompt_path.resolve()),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "jev_model": pipelines["jev"].client.model_id,
        "jev_endpoint": pipelines["jev"].client.endpoint,
        "fixed_threshold": FIXED_THRESHOLD,
        "fixed_jev_weight": FIXED_JEV_WEIGHT,
        "validation_used": False,
        "test_parameter_search_used": False,
        "elapsed_seconds": perf_counter() - started,
        "training_overlap": overlap,
        "scores": {
            name: {
                **calculate_metrics(expected, predictions[name]),
                "elapsed_seconds": durations[name],
                "fresh_api_requests": api_requests[name],
                "cache_hits": cache_hits[name],
            }
            for name in MODULES
        },
        "cache_note": "Hybrid reuses exactly the same genuine Jev responses as the baseline; cached timings are not cold-inference latency.",
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, metrics in report["scores"].items():
        logger.info("%s: micro-F1 %.6f, баллы %.3f/60, TP=%d FP=%d FN=%d", name,
                    metrics["micro_f1"], metrics["classification_points"],
                    metrics["tp"], metrics["fp"], metrics["fn"])
    return report


def main() -> None:
    """Запустить один итоговый прогон трёх заранее фиксированных вариантов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--training-dataset", type=Path, default=ROOT / "out/hybrid-training.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "out/jev-hybrid-comparison.json")
    args = parser.parse_args()
    compare(args.dataset, args.artifact, args.training_dataset, args.report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
