"""Обучить веса и оценить раздельные выборки из сохранённых снимков."""

import argparse
import gzip
import hashlib
from datetime import datetime
import json
import logging
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import expit, logit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from dataset_annotation.prepare_group_holdout import SOURCE_MC_ID, split_groups
from main import MICROCATEGORIES
from run import calculate_metrics, normalize_description, prepare_rows
from ..snippet_logistic_regression.train import duplicate_groups
from .pipeline import ARTIFACTS, DEFAULT_MODEL, MODEL_NAMES, TOKEN, features, identity, normalize_token, windows


def load_jsonl(path: Path) -> list[dict]:
    """Прочитать локальный снимок в порядке источника."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def main() -> None:
    """Обучить модель и записать проверяемый отчёт о пересечениях."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("local_dedup", type=Path)
    parser.add_argument("--provider-sources", nargs=2, type=Path, metavar=("YANDEX", "YOULA"))
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    providers = args.provider_sources
    model_name = MODEL_NAMES[1] if providers else DEFAULT_MODEL
    artifact = args.artifact_dir / f"{model_name}.json.gz"
    report_dir = args.snapshots / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{model_name}.json"
    metadata_path = artifact.with_name(f"{model_name}.metadata.json")
    if any(path.exists() for path in (artifact, metadata_path, report_path)):
        parser.error(f"Артефакт или отчёт {model_name} уже существует")
    paths = ({"yandex": providers[0], "youla": providers[1]} if providers else
             {name: args.snapshots / f"{name}.jsonl" for name in ("case_train", "case_test", "manual_test")})
    paths["local_dedup"] = args.local_dedup
    rows = {}
    unlabeled_by_source = {}
    for name, path in paths.items():
        records = load_jsonl(path)
        if name in ("yandex", "youla"):
            unlabeled_by_source[name] = sum(record["found_microcategories"] is None for record in records)
            records = [{"description": record["source"]["description"],
                        "found_microcategories": record["found_microcategories"]}
                       for record in records if record["found_microcategories"] is not None]
        if name in ("local_dedup", "yandex", "youla"):
            for record in records:
                record["found_microcategories"] = [
                    label if isinstance(label, dict) else {"mcId": label}
                    for label in record["found_microcategories"]
                ]
        rows[name] = prepare_rows(records, SOURCE_MC_ID)
    if providers:
        rows["provider_train"] = rows.pop("yandex") + rows.pop("youla")
    train_key = "provider_train" if providers else "case_train"
    index = json.loads(args.index.read_text())
    categories = sorted(set(MICROCATEGORIES) - {SOURCE_MC_ID})
    descriptions = [row.inp.description for part in rows.values() for row in part]
    lexical = [" ".join(re.findall(r"[^\W_]+", normalize_description(text))) or normalize_description(text)
               for text in descriptions]
    logging.info("Группировка %d описаний", len(lexical))
    groups, group_info = duplicate_groups(lexical)
    group_by_split = dict(zip(rows, np.split(groups, np.cumsum([len(v) for v in rows.values()])[:-1]), strict=True))
    if providers:
        excluded = group_by_split["local_dedup"]
    else:
        excluded = np.union1d(group_by_split["case_test"], group_by_split["manual_test"])
    train_groups = group_by_split[train_key]
    remaining = np.flatnonzero(~np.isin(train_groups, excluded))
    if providers:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=43)
        fit_idx, tune_idx = next(splitter.split(remaining, groups=train_groups[remaining]))
        fit, tune = remaining[fit_idx], remaining[tune_idx]
    else:
        assignments = split_groups(train_groups[remaining])
        fit, tune = (remaining[assignments[name]] for name in ("fit", "tune"))
    used_groups = np.union1d(train_groups[fit], train_groups[tune])
    assert not np.intersect1d(used_groups, excluded).size
    assert not np.intersect1d(train_groups[fit], train_groups[tune]).size

    def extract(part):
        for row in part:
            snippets = windows(row.inp.description, index)
            hits = {int(cid) for token in TOKEN.finditer(row.inp.description)
                    for cid, _ in index.get(normalize_token(token.group()), ())}
            yield features(snippets, "pairs"), bool(snippets), hits

    train = rows[train_key]
    fit_truth = np.array([[cid in train[i].expected for cid in categories] for i in fit])
    logging.info("Обучение на %d строках, калибровка на %d", len(fit), len(tune))
    vectorizer = CountVectorizer(analyzer=identity, binary=True, min_df=2)
    x_fit = vectorizer.fit_transform(values for values, _, _ in extract(train[i] for i in fit))
    models = [LogisticRegression(solver="liblinear", max_iter=500, random_state=44).fit(x_fit, labels)
              for labels in fit_truth.T]
    weights = np.stack([model.coef_[0] for model in models])
    intercepts = np.array([model.intercept_[0] for model in models])

    def score(part):
        probabilities, seen = [], []
        for start in range(0, len(part), 128):
            feature_rows, hit, hits = zip(*extract(part[start:start + 128]), strict=True)
            batch = expit(vectorizer.transform(feature_rows) @ weights.T + intercepts)
            batch[~np.array(hit)] = 0
            probabilities.extend(batch)
            seen.extend(hits)
        return np.array(probabilities), seen

    values, _ = score([train[i] for i in tune])
    y = np.array([[cid in train[i].expected for cid in categories] for i in tune])
    grid = np.arange(0.10, 0.91, 0.025)
    predictions = values[:, :, None] >= grid
    f1 = 2 * (predictions & y[:, :, None]).sum(axis=0) / np.maximum(
        1, predictions.sum(axis=0) + y.sum(axis=0)[:, None])
    thresholds = grid[f1.argmax(axis=1)]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_data = {"index": index, "categories": categories, "vocabulary": vectorizer.vocabulary_,
                  "weights": weights.tolist(), "cutoffs": (logit(thresholds) - intercepts).tolist()}
    with gzip.open(artifact, "wt", encoding="utf-8") as file:
        json.dump(model_data, file, ensure_ascii=False, allow_nan=False)
    report = {"model": model_name, "date": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds"),
              "training_source": train_key, "unlabeled_by_source": unlabeled_by_source,
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("train.py", "pipeline.py")}, "fit": len(fit), "tune": len(tune), "excluded_train": len(train) - len(remaining),
              "grouping": group_info, "thresholds": thresholds.tolist(), "features": weights.shape[1],
              "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                               for key, path in {**paths, "index": args.index}.items()},
              "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
              "results": {}}
    for name in (("local_dedup",) if providers else ("manual_test", "local_dedup")):
        part = rows[name]
        values, seen = score(part)
        predicted = [{cid for j, cid in enumerate(categories) if values[i, j] >= thresholds[j] and cid != row.inp.mc_id}
                     for i, row in enumerate(part)]
        overlap = np.isin(group_by_split[name], used_groups)

        def summarize(indices):
            gold = [part[i].expected for i in indices]
            found = [predicted[i] for i in indices]
            return {"n": len(indices), **calculate_metrics(gold, found),
                    "gold_with_anchor": sum(len(part[i].expected & seen[i]) for i in indices),
                    "gold_labels": sum(map(len, gold))}

        report["results"][name] = {"full": summarize(np.arange(len(part))),
                                  "without_fit_or_tune_groups": summarize(np.flatnonzero(~overlap)),
                                  "overlap_rows": int(overlap.sum()),
                                  "rows": [{"expected": sorted(row.expected), "predicted": sorted(predicted[i]),
                                            "anchor_categories": sorted(seen[i]), "overlap": bool(overlap[i])}
                                           for i, row in enumerate(part)]}
        logging.info("%s: %s", name, report["results"][name]["without_fit_or_tune_groups"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    metadata_path.write_text(json.dumps(
        {key: value for key, value in report.items() if key != "results"}, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
