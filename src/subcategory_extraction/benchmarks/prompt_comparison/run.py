"""Оценить выбранный модуль и его промпты на общей validation-выборке."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import logging
from time import perf_counter
from typing import Any, Callable

from benchmark_support import TRAIN, metadata, save_report, setup, timing, validation_sample
from main import MICROCATEGORIES, PipelineConfig, load_pipeline
from subcategory_extraction.benchmarks.metrics import score


def evaluate(
    module: str, prompt: str | None, rows: list[dict[str, Any]],
    checkpoint: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Вызвать настоящий run(); незавершённому сценарию не присваивать F1."""
    result: dict[str, Any] = {"module": module, "prompt": prompt, "status": "running", "rows": []}
    durations: list[float] = []
    checkpoint(result)
    try:
        pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
            subcategory_extraction=module, draft_generation=None,
        ))
        client = getattr(pipeline, "client", None)
        result["model"] = client.model_id if client else module
        if prompt is not None:
            prompt_path = Path(__file__).resolve().parents[2] / module / "prompts" / f"google_decision_{prompt}.md"
            pipeline.prompt = prompt_path.read_text(encoding="utf-8")
            result["prompt_path"] = str(prompt_path.relative_to(TRAIN.parents[2]))
            result["prompt_sha256"] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            guidance = prompt_path.with_name("category_guidance.json")
            if guidance.exists():
                result["guidance_sha256"] = hashlib.sha256(guidance.read_bytes()).hexdigest()
        result["threshold"] = pipeline.threshold
        if client is not None:
            settings = getattr(client, "settings", None)
            if settings is not None:
                result["api_settings"] = settings.model_dump()
            if not client.api_key:
                raise ValueError("Не задан API-ключ; настоящие вызовы не выполнялись")
        for index, row in enumerate(rows, 1):
            started = perf_counter()
            try:
                predicted = pipeline.run(row["inp"])
                if not isinstance(predicted, list) or any(type(cid) is not int or cid not in MICROCATEGORIES for cid in predicted):
                    raise ValueError("Модуль вернул неверный контракт категорий")
            except Exception:
                result["failed_row"] = index
                result["failed_seconds"] = perf_counter() - started
                raise
            seconds = perf_counter() - started
            durations.append(seconds)
            result["rows"].append({
                "group_id": row["group_id"], "input": row["inp"].model_dump(by_alias=True),
                "expected": sorted(row["targets"]), "predicted": predicted, "seconds": seconds,
            })
            result["response_time"] = timing(durations)
            checkpoint(result)
            logging.info("%s/%s: %d/%d, %.3f с", module, prompt or "—", index, len(rows), seconds)
        result["metrics"] = score([row["targets"] for row in rows], [set(row["predicted"]) for row in result["rows"]])
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logging.error("%s/%s: %s", module, prompt or "—", result["error"])
    except KeyboardInterrupt:
        result.update(status="failed", interrupted=True, error="Прогон прерван; итоговый F1 не рассчитан")
    result["response_time"] = timing(durations)
    return result


def main() -> int:
    """Записать новый прогон отдельно от сохранённых исторических результатов."""
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="google_decision")
    parser.add_argument("--prompts", choices=("baseline", "guided"), nargs="+", default=["baseline", "guided"])
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("rebuild-results.json"))
    args = parser.parse_args()
    rows = validation_sample(args.size)
    report = {
        **metadata(), "scenario": "validation_quality", "module": args.module,
        "dataset": str(TRAIN.relative_to(TRAIN.parents[2])),
        "dataset_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "split": "validation", "sample_seed": 43, "sample_size": len(rows),
        "sample_groups": [row["group_id"] for row in rows],
        "timing_includes": "run(), API retries and quota waits; excludes instance construction and data loading",
        "results": {},
    }
    prompts = list(dict.fromkeys(args.prompts)) if args.module == "google_decision" else [None]
    for prompt in prompts:
        def checkpoint(result: dict[str, Any]) -> None:
            """Сохранить текущие ответы сразу после очередного run()."""
            report["results"][prompt or "no_prompt"] = result
            save_report(args.report, report)

        result = evaluate(args.module, prompt, rows, checkpoint)
        report["results"][prompt or "no_prompt"] = result
        save_report(args.report, report)
        if result.get("interrupted"):
            break
    complete = all(row["status"] == "complete" for row in report["results"].values())
    report["status"] = "complete" if complete else "failed"
    report["best_prompt"] = max(
        report["results"], key=lambda name: report["results"][name]["metrics"]["micro_f1"],
    ) if complete and args.module == "google_decision" else None
    save_report(args.report, report)
    return 0 if complete else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
