"""Сравнить фиксированные способы применения veto на групповых сплитах."""

import argparse
import gzip
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from main import PipelineInput
from ..order_format_veto.pipeline import veto
from ..snippet_logistic_regression.pipeline import ad_features
from ..snippet_logistic_regression.train import metrics
from .pipeline import SubcategoryExtractionPipeline
from .train import ARTIFACT_ENV, ARTIFACTS, CATEGORIES, SOURCE_MC_ID, read_sources, resolve, sha256, train

PHASES = ("logreg", "logreg_veto", "joint", "soft_veto")
logger = logging.getLogger(__name__)


def load(path: Path) -> SubcategoryExtractionPipeline:
    """Прочитать веса для одного прогона тем же загрузчиком, что в рантайме."""
    saved = os.environ.get(ARTIFACT_ENV)
    os.environ[ARTIFACT_ENV] = str(path)
    try:
        return SubcategoryExtractionPipeline()
    finally:
        if saved is None:
            os.environ.pop(ARTIFACT_ENV, None)
        else:
            os.environ[ARTIFACT_ENV] = saved


def predict(model: SubcategoryExtractionPipeline, text: str, margin_limit: float) -> dict:
    """Подать одним и тем же вариантам одинаковые оценки и очищенный текст."""
    values, hit = ad_features(text, model.anchors, model.ranked, model.width, model.mode)
    row = model.vectorizer.transform([values])
    margins = (row @ model.weights.T)[0] - model.cutoff
    logreg = [cid for cid, margin in zip(model.categories, margins, strict=True) if hit and margin >= 0]
    candidates = model.candidates(PipelineInput(description=text, mcId=SOURCE_MC_ID))
    soft_candidates = [
        cid for cid in logreg
        if cid in candidates or margins[model.categories.index(cid)] > margin_limit
    ]
    direct, direct_evidence = veto(text, logreg)
    current, current_evidence = veto(text, candidates)
    soft, _ = veto(text, soft_candidates)
    assert set(direct) <= set(logreg) and set(current) <= set(candidates)
    assert set(soft) <= set(soft_candidates) <= set(logreg)
    return {
        "logreg": logreg,
        "logreg_veto": direct,
        "candidates": candidates,
        "joint": current,
        "soft_candidates": soft_candidates,
        "soft_veto": soft,
        "direct_evidence": direct_evidence,
        "joint_evidence": current_evidence,
    }


def evaluate(path: Path, data: tuple[list[str], np.ndarray], refs: list[dict], margin: float) -> list[dict]:
    """Сохранить парные построчные прогнозы и группы для сравнения."""
    model = load(path)
    texts, labels = data
    return [
        {
            **ref,
            "expected": [cid for cid, present in zip(CATEGORIES, y, strict=True) if present],
            **predict(model, text, margin),
        }
        for text, y, ref in zip(texts, labels, refs, strict=True)
    ]


def fit_fold(spec: tuple[str, dict, Path, str, float]) -> dict:
    """Переобучить словарь, веса и пороги внутри одного внешнего фолда."""
    name, split, source, digest, margin = spec
    parts = [split[k] for k in ("fit", "tune", "evaluation")]
    fit_groups, tune_groups, evaluation_groups = ({r["group"] for r in refs} for refs in parts)
    if fit_groups & tune_groups or (fit_groups | tune_groups) & evaluation_groups:
        raise ValueError(f"Пересечение групп в {name}")
    records = read_sources({"dedup": source}, {"dedup": digest})
    fit, tune = (resolve(refs, records) for refs in parts[:2])
    artifact, calibration = train(fit, tune)
    with TemporaryDirectory(prefix="order-veto-ablation-") as directory:
        path = Path(directory) / "model.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(artifact, stream, ensure_ascii=False, allow_nan=False)
        rows = evaluate(path, resolve(parts[2], records), parts[2], margin)
    return {
        "name": name, "fit_n": len(fit[0]), "tune_n": len(tune[0]),
        "evaluation_n": len(rows), "calibration": calibration, "rows": rows,
    }


def summarize(rows: list[dict]) -> dict:
    """Сравнить micro-F1 и изменения категорий относительно чистого логрега."""
    target = np.array([[cid in row["expected"] for cid in CATEGORIES] for row in rows])
    result = {}
    for phase in PHASES:
        estimates = np.array([[cid in row[phase] for cid in CATEGORIES] for row in rows])
        by_category = {}
        for cid in CATEGORIES:
            changes = {"correct_removed": 0, "incorrect_removed": 0, "correct_added": 0, "incorrect_added": 0}
            for r in rows:
                before, after, present = cid in r["logreg"], cid in r[phase], cid in r["expected"]
                if before and not after:
                    changes["incorrect_removed" if present else "correct_removed"] += 1
                elif after and not before:
                    changes["correct_added" if present else "incorrect_added"] += 1
            by_category[cid] = changes
        result[phase] = {
            **metrics(target, estimates, CATEGORIES),
            "exact": sum(row["expected"] == row[phase] for row in rows),
            "changed_ads": sum(row["logreg"] != row[phase] for row in rows),
            "changes_by_category": by_category,
        }
    return result


def main() -> None:
    """Зафиксировать выбор на OOF до прогона отложенной части."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("split", "original_split", "dedup", "report"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    original = json.loads(args.original_split.read_text(encoding="utf-8"))
    protocol = split["protocol"]
    digest = split["source_sha256"]["dedup"]
    if digest != original["source_sha256"]["dedup"]:
        raise ValueError("Сравнение и регрессия должны использовать один снимок дедупа")
    margin = protocol["soft_max_logit_margin"]
    heldout = {r["group"] for r in split["confirmation"]["evaluation"]}
    regression = {r["group"] for r in original["test"]}
    if any(ref["group"] in regression for key in ("fit", "tune", "evaluation")
           for ref in split["confirmation"][key]):
        raise ValueError("Группа регрессии попала в подтверждающий сплит")
    for fold in split["folds"]:
        used = {r["group"] for key in ("fit", "tune", "evaluation") for r in fold[key]}
        if used & (heldout | regression):
            raise ValueError("Отложенная группа попала во внешний фолд")
    specs = [
        (f"fold-{i}", fold, args.dedup, digest, margin)
        for i, fold in enumerate(split["folds"], 1)
    ]
    with ProcessPoolExecutor(max_workers=2) as pool:
        folds = []
        for result in pool.map(fit_fold, specs):
            logger.info("Завершён %s: %s строк", result["name"], result["evaluation_n"])
            folds.append(result)
    oof_rows = [row for fold in folds for row in fold["rows"]]
    oof_scores = summarize(oof_rows)
    selected = max(PHASES, key=lambda phase: oof_scores[phase]["micro_f1"])
    logger.info("Выбор на OOF: %s", selected)
    report = {
        "split_sha256": sha256(args.split), "protocol": protocol, "selected_on_oof": selected,
        "oof_n": len(oof_rows), "oof_scores": oof_scores, "folds": folds,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    confirmation = fit_fold(("confirmation", split["confirmation"], args.dedup, digest, margin))
    report["confirmation"] = {**confirmation, "scores": summarize(confirmation["rows"])}
    records = read_sources({"dedup": args.dedup}, {"dedup": digest})
    test = resolve(original["test"], records)
    report["regression_only"] = {}
    for name in ("dedup", "yandex_youla"):
        path = ARTIFACTS / f"{name}.json.gz"
        rows = evaluate(path, test, original["test"], margin)
        report["regression_only"][name] = {"scores": summarize(rows), "rows": rows}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
