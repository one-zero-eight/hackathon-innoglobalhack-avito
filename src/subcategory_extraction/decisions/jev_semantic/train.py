"""Выбрать семантический промпт/пороги только на независимых группах manual_train."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import sys
from threading import local

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from benchmark_support import metadata, save_report, setup
from database import read_records, table_sha256
from main import PipelineInput
from run import calculate_metrics, prepare_rows
from subcategory_extraction.decisions.jev_semantic.pipeline import CALIBRATION, VARIANTS, SubcategoryExtractionPipeline, prompt_sha256
from subcategory_extraction.decisions.jev_calibrated.pipeline import SubcategoryExtractionPipeline as Baseline
from subcategory_extraction.semantic.semantic_encoder_f1.robust_thresholds import select_robust_thresholds

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "src/subcategory_extraction/benchmarks/classifier_methods/jev-semantic-split.json"
REPORT = PROTOCOL.with_name("jev-semantic-training-results.json")
CATEGORIES = [102, 103, 104, 105, 106]


def main() -> None:
    """Сохранить реальные вероятности, выбор crossfit и нетронутый control."""
    setup()
    split = json.loads(PROTOCOL.read_text())
    if table_sha256("manual_train") != split["table_sha256"]["manual_train"]:
        raise ValueError("Training table changed")
    indexed = {row["_row_id"]: row for row in read_records("manual_train")}
    parts = {name: prepare_rows([indexed[item["row_id"]] for item in split["training"][name]], 101)
             for name in ("fit", "control")}
    public_texts = {hashlib.sha256(row["description"].encode()).hexdigest() for row in read_records("case_train")}
    selected_hashes = [hashlib.sha256(row.inp.description.encode()).hexdigest() for rows in parts.values() for row in rows]
    if any(digest not in public_texts for digest in selected_hashes):
        raise ValueError("Разрешены только побайтно совпавшие публичные case_train описания; private/manual labels остаются локально")
    report = {**metadata(), "scenario": "semantic_train_only", "training_table": "manual_train",
              "public_text_transfer_audit": {"selected": len(selected_hashes), "byte_identical_public_descriptions": len(selected_hashes),
                                             "public_source_table": "case_train", "manual_labels_transmitted": False},
              "split_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "private_labels_used_for_selection": False,
              "results": {}, "status": "running"}

    def collect(variant: str, part: str) -> tuple[np.ndarray, dict]:
        """Дать каждому потоку собственный клиент, сохранив порядок строк."""
        state = local()

        def predict(row):
            if not hasattr(state, "pipeline"):
                state.pipeline = SubcategoryExtractionPipeline(variant=variant) if variant != "baseline" else Baseline()
            probabilities = state.pipeline.probabilities(row.inp)
            return {"description_sha256": hashlib.sha256(row.inp.description.encode()).hexdigest(),
                    "expected": sorted(row.expected), "probabilities": {str(k): v for k, v in probabilities.items()},
                    "usage": state.pipeline.client.last_usage, "cache_hit": state.pipeline.client.last_cache_hit}

        records = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for record in pool.map(predict, parts[part]):
                records.append(record)
                if len(records) % 24 == 0:
                    logging.info("%s/%s: %d/%d", variant, part, len(records), len(parts[part]))
        return np.asarray([[row["probabilities"][str(cid)] for cid in CATEGORIES] for row in records]), {"responses": records}

    truth = np.asarray([[cid in row.expected for cid in CATEGORIES] for row in parts["fit"]])
    for variant in VARIANTS:
        probabilities, candidate = collect(variant, "fit")
        candidate["tuning"] = select_robust_thresholds(probabilities, truth, CATEGORIES,
                                                     np.asarray([item["group"] for item in split["training"]["fit"]]))
        report["results"][variant] = candidate
        save_report(REPORT, report)
    selected = max(VARIANTS, key=lambda name: report["results"][name]["tuning"]["selection_metric"]["micro_f1"])
    report["selected_before_control"] = selected
    thresholds = report["results"][selected]["tuning"]["thresholds"]
    probabilities, candidate = collect(selected, "control")
    predicted = [{cid for cid, score in zip(CATEGORIES, row) if score >= thresholds[str(cid)]} for row in probabilities]
    control = calculate_metrics([row.expected for row in parts["control"]], predicted)
    baseline_probs, baseline = collect("baseline", "control")
    original = Baseline()
    baseline_predictions = [{cid for cid, score in zip(CATEGORIES, row) if score >= original.thresholds[cid]} for row in baseline_probs]
    baseline_metrics = calculate_metrics([row.expected for row in parts["control"]], baseline_predictions)
    report.update(status="complete", control={"selected": control, "baseline": baseline_metrics,
                                               "selected_responses": candidate["responses"], "baseline_responses": baseline["responses"]})
    pipeline = SubcategoryExtractionPipeline(variant=selected)
    calibration = {**metadata(), "provider": pipeline.client.provider, "model": pipeline.client.model_id,
                   "endpoint": pipeline.client.endpoint, "prompt": selected, "prompt_sha256": prompt_sha256(selected),
                   "thresholds": thresholds, "training_table": "manual_train", "fit_rows": 96, "control_rows": 48,
                   "split_sha256": report["split_sha256"], "selection": report["results"][selected]["tuning"]["selected_method"],
                   "private_labels_used_for_selection": False, "category_101": "fixed0.5; no positive train labels available",
                   "control_confirmed": control["micro_f1"] >= baseline_metrics["micro_f1"], "control_metrics": control,
                   "control_baseline_metrics": baseline_metrics}
    CALIBRATION.parent.mkdir(exist_ok=True)
    CALIBRATION.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n")
    save_report(REPORT, report)
    logging.info("Selected %s; control %.6f vs baseline %.6f; thresholds %s", selected, control["micro_f1"],
                 baseline_metrics["micro_f1"], thresholds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
