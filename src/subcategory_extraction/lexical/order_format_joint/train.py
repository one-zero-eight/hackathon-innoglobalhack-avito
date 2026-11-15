"""Сравнить источники обучения логрега на зафиксированном тесте дедупа."""

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from main import MICROCATEGORIES, PipelineInput
from run import normalize_description
from ..order_format_veto.pipeline import clean_description, veto
from ..snippet_logistic_regression.pipeline import ad_features, identity
from ..snippet_logistic_regression.train import SOURCE_MC_ID, learn_anchors, metrics
from .pipeline import SubcategoryExtractionPipeline

ARTIFACTS = Path(__file__).with_name("artifacts")
CATEGORIES = sorted(cid for cid in MICROCATEGORIES if cid != SOURCE_MC_ID)
WIDTH = 7
LABELS = "found_microcategories"
ARTIFACT_ENV = "SNIPPET_LOGISTIC_ARTIFACT"
logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    """Вернуть отпечаток файла."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def description_hash(text: str) -> str:
    """Повторить нормализацию, использованную при фиксации сплита."""
    normalized = normalize_description(text)
    normalized = " ".join(re.findall(r"[^\W_]+", normalized)) or normalized
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_sources(paths: dict[str, Path], hashes: dict[str, str]) -> dict[str, dict]:
    """Прочитать только записи из заранее зафиксированных снимков."""
    sources = {}
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"Отпечаток источника {name} изменился")
        records = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row[LABELS] is None:
                    continue
                src = row.get("source", row)
                source_id = src.get("item_id") or src.get("url") or src.get("provider_id")
                key = (source_id, description_hash(src["description"]))
                if key in records:
                    raise ValueError(f"Ненадёжный ID строки {source_id} в {name}")
                records[key] = (src["description"], {
                    label["mcId"] if isinstance(label, dict) else label for label in row[LABELS]
                })
        sources[name] = records
    return sources


def resolve(refs: list[dict], sources: dict[str, dict]) -> tuple[list[str], np.ndarray]:
    """Проверить соответствие строк сплиту и подготовить тексты одинаково."""
    texts = []
    labels = []
    for ref in refs:
        key = (ref["source_id"], ref["description_sha256"])
        text, targets = sources[ref["source"]][key]
        if targets - MICROCATEGORIES.keys():
            raise ValueError(f"Неизвестные категории в строке {ref['source_id']}")
        texts.append(clean_description(text))
        labels.append([cid in targets for cid in CATEGORIES])
    return texts, np.asarray(labels, dtype=bool)


def train(
    fit: tuple[list[str], np.ndarray], tune: tuple[list[str], np.ndarray],
) -> tuple[dict, dict]:
    """Обучить признаки и логрег на fit, подобрать пороги только на tune."""
    texts, labels = fit
    tune_texts, truth = tune
    ranked = learn_anchors(texts, labels, CATEGORIES)
    anchors = {term for terms in ranked.values() for term in terms}
    train_rows = [ad_features(t, anchors, ranked, WIDTH, "pairs")[0] for t in texts]
    tune_rows, hits = zip(*(ad_features(t, anchors, ranked, WIDTH, "pairs") for t in tune_texts), strict=True)
    vectorizer = CountVectorizer(analyzer=identity, binary=True, min_df=2)
    x_fit = vectorizer.fit_transform(train_rows)
    x_tune = vectorizer.transform(tune_rows)

    coefficients = []
    cutoffs = []
    thresholds = []
    predictions = []
    grid = np.arange(0.1, 0.91, 0.025)
    for i, cid in enumerate(CATEGORIES):
        if len(set(labels[:, i])) != 2 or not truth[:, i].any():
            raise ValueError(f"В fit или tune недостаточно разметки категории {cid}")
        model = LogisticRegression(solver="liblinear", max_iter=500, random_state=44)
        model.fit(x_fit, labels[:, i])
        scores = model.predict_proba(x_tune)[:, 1]
        options = (scores[:, None] >= grid) & np.asarray(hits)[:, None]
        target = truth[:, i]
        f1 = 2 * (target[:, None] & options).sum(axis=0) / np.maximum(1, target.sum() + options.sum(axis=0))
        best = int(f1.argmax())
        predictions.append(options[:, best])
        thresholds.append(float(grid[best]))
        coefficients.append(model.coef_[0].tolist())
        cutoffs.append(float(logit(grid[best]) - model.intercept_[0]))

    artifact = {
        "policy": ["pairs", WIDTH, 0],
        "categories": CATEGORIES, "anchors": sorted(anchors), "ranked": ranked, "quota": 160,
        "vocabulary": vectorizer.vocabulary_, "weights": coefficients, "cutoff": cutoffs,
    }
    return artifact, {"thresholds": thresholds, **metrics(truth, np.column_stack(predictions), CATEGORIES)}


def measure(artifact: Path, test: tuple[list[str], np.ndarray], refs: list[dict]) -> dict:
    """Прогнать поставляемые веса через обычные этапы пайплайна."""
    previous = os.environ.get(ARTIFACT_ENV)
    os.environ[ARTIFACT_ENV] = str(artifact)
    try:
        model = SubcategoryExtractionPipeline()
    finally:
        if previous is None:
            os.environ.pop(ARTIFACT_ENV, None)
        else:
            os.environ[ARTIFACT_ENV] = previous

    rows = []
    texts, truth = test
    for text, ref, y in zip(texts, refs, truth, strict=True):
        inp = PipelineInput(mcId=SOURCE_MC_ID, description=text)
        raw = super(SubcategoryExtractionPipeline, model).run(inp)
        candidates = model.candidates(inp)
        result, evidence = veto(text, candidates)
        assert set(result) <= set(candidates)
        rows.append({
            "source_id": ref["source_id"],
            "description_sha256": ref["description_sha256"],
            "expected": [cid for cid, present in zip(CATEGORIES, y, strict=True) if present],
            "logreg": raw, "candidates": candidates, "after_veto": result, "refusals": evidence,
        })
    scores = {
        phase: {
            **metrics(truth, np.array([[cid in row[phase] for cid in CATEGORIES] for row in rows]), CATEGORIES),
            "exact": sum(row[phase] == row["expected"] for row in rows),
        }
        for phase in ("logreg", "candidates", "after_veto")
    }
    return {
        "scores": scores,
        "ads_changed_by_veto": sum(row["candidates"] != row["after_veto"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    """Обучить отдельно два логрега и выполнить один общий тест."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("split", "dedup", "yandex", "youla", "report"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    outputs = [args.report, *(args.artifact_dir / f"{name}.json.gz" for name in split["runs"])]
    if any(path.exists() for path in outputs):
        parser.error("Артефакт или отчёт уже существует")
    sources = read_sources({name: getattr(args, name) for name in ("dedup", "yandex", "youla")}, split["source_sha256"])
    group_test = {ref["group"] for ref in split["test"]}
    models = {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for name, run in split["runs"].items():
        fit_groups = {ref["group"] for ref in run["fit"]}
        tune_groups = {ref["group"] for ref in run["tune"]}
        if fit_groups & tune_groups or (fit_groups | tune_groups) & group_test:
            raise ValueError(f"Пересечение групп для модели {name}")
        fit, tune = (resolve(run[part], sources) for part in ("fit", "tune"))
        logger.info("%s: обучение на %d, калибровка на %d", name, len(fit[0]), len(tune[0]))
        artifact, validation = train(fit, tune)
        artifact["split_sha256"] = sha256(args.split)
        path = args.artifact_dir / f"{name}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(artifact, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
        models[name] = {
            "fit": len(fit[0]), "tune": len(tune[0]), "validation": validation,
            "artifact_sha256": sha256(path), "artifact": path.name,
        }

    test = resolve(split["test"], sources)
    for name, info in models.items():
        info["test"] = measure(args.artifact_dir / info["artifact"], test, split["test"])
        logger.info("%s: итоговый micro-F1 %.4f", name, info["test"]["scores"]["after_veto"]["micro_f1"])
    report = {"split_sha256": sha256(args.split), "test_n": len(test[0]), "models": models}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
