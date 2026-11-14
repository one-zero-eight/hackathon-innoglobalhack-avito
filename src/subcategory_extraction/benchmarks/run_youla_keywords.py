"""Сравнить признаки из Юлы на фиксированном holdout InnoGlobalHack."""

import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from main import MICROCATEGORIES, PipelineConfig, load_pipeline
from subcategory_extraction.benchmarks.metrics import score
from subcategory_extraction.benchmarks.train_tfidf import DATASET, load_rows, normalize
from subcategory_extraction.learned_keywords.pipeline import DEFAULT_ARTIFACT

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "out/youla-300-decisions.jsonl"
REPORT = ROOT / "out/youla-learned-keywords-metrics.json"
SOURCE = 101
KEYWORDS_PER_CATEGORY = 40


def group(text: str) -> str:
    """Объединить варианты описания с разными числами и пунктуацией."""
    return " ".join(re.findall(r"[^\W\d_]+", normalize(text)))


def main() -> None:
    """Обучить все варианты на Юле и измерить один и тот же holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--size", type=int)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    with args.train.open(encoding="utf-8") as stream:
        train = [json.loads(line) for line in stream]
    if args.size is not None:
        if not 1 <= args.size <= len(train):
            parser.error("--size превышает число размеченных строк или меньше 1")
        train = train[:args.size]
    rows = load_rows(DATASET)
    groups = np.array([group(row["inp"].description) for row in rows])
    _, rest = next(GroupShuffleSplit(test_size=0.2, random_state=44).split(rows, groups=groups))
    dev, test = next(GroupShuffleSplit(test_size=0.5, random_state=45).split(rest, groups=groups[rest]))
    train_groups = {group(row["description"]) for row in train}
    indices = [i for i in rest[test] if groups[i] not in train_groups]
    dev_indices = [i for i in rest[dev] if groups[i] not in train_groups]
    inputs = [rows[i]["inp"] for i in [*indices, *dev_indices]]
    expected = [rows[i]["targets"] for i in indices]
    expected_dev = [rows[i]["targets"] for i in dev_indices]
    n_test = len(indices)
    categories = list(MICROCATEGORIES)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    full = vectorizer.fit_transform([row["description"] for row in train])
    binary = full.sign()
    held = vectorizer.transform([inp.description for inp in inputs])
    held_binary = held.sign()
    terms = vectorizer.get_feature_names_out()
    results, keywords = {}, {}
    selected = {}
    for cid in categories:
        if cid == SOURCE:
            continue
        targets = np.array([cid in row["target_mc_ids"] for row in train])
        statistics, _ = chi2(binary, targets)
        more_common_in_positive = (
            np.asarray(binary[targets].mean(axis=0)).ravel()
            > np.asarray(binary[~targets].mean(axis=0)).ravel()
        )
        eligible = np.flatnonzero(more_common_in_positive)
        selected[cid] = eligible[np.argsort(-statistics[eligible], kind="stable")[:KEYWORDS_PER_CATEGORY]]
        keywords[cid] = terms[selected[cid]].tolist()

    # Context is fitted only on Youla spans around the selected phrases.
    tokenize = vectorizer.build_tokenizer()
    preprocess = vectorizer.build_preprocessor()

    def snippets(texts: list[str], phrases: set[str]) -> list[str]:
        contexts = []
        for text in texts:
            tokens = tokenize(preprocess(text))
            windows = [
                " ".join(tokens[max(0, i - 5):i + 7])
                for i, token in enumerate(tokens)
                if token in phrases or " ".join(tokens[i:i + 2]) in phrases
            ]
            contexts.append(" ".join(windows))
        return contexts

    text_train = [row["description"] for row in train]
    text_held = [inp.description for inp in inputs]
    context_features = {}
    for cid, phrases in keywords.items():
        local = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        context_features[cid] = (
            local.fit_transform(snippets(text_train, set(phrases))),
            local.transform(snippets(text_held, set(phrases))),
        )

    thresholds = (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)

    def sweep(logits: np.ndarray, plastering_only: bool) -> dict:
        """Сравнить пороги с выбором по micro-F1 на validation."""
        trials = {}
        for threshold in thresholds:
            cutoff = np.log(threshold / (1 - threshold))
            cutoffs = np.full(len(categories), 0.0 if plastering_only else cutoff)
            if plastering_only:
                cutoffs[categories.index(102)] = cutoff
            predictions = [
                {cid for cid, matched in zip(categories, row, strict=True) if matched and cid != SOURCE}
                for row in logits >= cutoffs
            ]
            trials[threshold] = {
                "validation": score(expected_dev, predictions[n_test:]),
                "holdout": score(expected, predictions[:n_test]),
            }
        chosen = max(trials, key=lambda t: trials[t]["validation"]["micro_f1"])
        return {"selected_threshold": chosen, "trials": trials}

    def tune_categories(logits: np.ndarray) -> dict:
        """Найти максимум validation micro-F1 на сетке порогов."""
        target_ids = [cid for cid in categories if cid != SOURCE]
        grid = np.array(sorted({0.0, 1.0, *np.arange(1, 100) / 100}))
        validation_truth = np.array([[cid in labels for cid in target_ids] for labels in expected_dev])
        columns = [categories.index(cid) for cid in target_ids]
        probabilities = 1 / (1 + np.exp(-np.clip(logits[:, columns], -700, 700)))
        predicted = probabilities[n_test:, :, None] >= grid
        tp = (predicted & validation_truth[:, :, None]).sum(axis=0)
        count = predicted.sum(axis=0)
        ratio = 0.0
        # For a fixed F1 ratio, the objective separates by category.
        while True:
            best = (2 * tp - ratio * count).argmax(axis=1)
            total_tp = tp[np.arange(len(columns)), best].sum()
            total_predicted = count[np.arange(len(columns)), best].sum()
            updated = 2 * total_tp / (validation_truth.sum() + total_predicted)
            if updated <= ratio + 1e-12:
                break
            ratio = updated
        cutoffs = grid[best]
        outputs = [{cid for cid, match in zip(target_ids, row, strict=True) if match}
                   for row in probabilities >= cutoffs]
        validation = score(expected_dev, outputs[n_test:])
        if abs(validation["micro_f1"] - updated) > 1e-12:
            raise ValueError("Метрика выбранных порогов не совпадает с расчётом")
        return {
            "thresholds": dict(zip(target_ids, cutoffs.tolist(), strict=True)),
            "validation": validation, "holdout": score(expected, outputs[:n_test]),
        }

    def evaluate(logits: np.ndarray) -> dict:
        """Сохранить результаты двух серий и общую метрику при 0.50."""
        shared = sweep(logits, plastering_only=False)
        plastering = sweep(logits, plastering_only=True)
        fixed = shared["trials"][0.5]["holdout"]
        if fixed != plastering["trials"][0.5]["holdout"]:
            raise ValueError("Результаты при одинаковых порогах различаются")
        chosen = shared["selected_threshold"]
        return {
            **fixed, "validation_selected_threshold": chosen,
            "tuned_holdout": shared["trials"][chosen]["holdout"],
            "shared_cutoffs": shared, "plastering_cutoffs": plastering,
            "per_category_cutoffs": tune_categories(logits),
        }

    shape = (len(categories), full.shape[1])
    for mode in ("whole", "keywords", "combined"):
        whole_weights, keyword_weights = np.zeros(shape), np.zeros(shape)
        intercepts = np.zeros(len(categories))
        for index, cid in enumerate(categories):
            if cid == SOURCE:
                intercepts[index] = -40
                continue
            columns = selected[cid]
            features = full if mode == "whole" else binary[:, columns]
            if mode == "combined":
                features = hstack([full, binary[:, columns]], format="csr")
            labels = [cid in row["target_mc_ids"] for row in train]
            clf = LogisticRegression(C=4, max_iter=500).fit(features, labels)
            intercepts[index] = clf.intercept_[0]
            if mode != "keywords":
                whole_weights[index] = clf.coef_[0, :full.shape[1]]
            if mode != "whole":
                keyword_weights[index, columns] = clf.coef_[0, -len(columns):]
        logits = held @ whole_weights.T + held_binary @ keyword_weights.T + intercepts
        predictions = [{cid for cid, value in zip(categories, row, strict=True) if value >= 0}
                       for row in logits[:n_test]]
        results[mode] = evaluate(logits)
        if mode == "combined":
            artifact = {
                "version": 1, "categories": categories, "vocabulary": vectorizer.vocabulary_,
                "idf": vectorizer.idf_.tolist(), "weights": whole_weights.tolist(),
                "keyword_weights": keyword_weights.tolist(), "intercepts": intercepts.tolist(),
            }
            args.artifact.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
            os.environ["LEARNED_KEYWORDS_ARTIFACT"] = str(args.artifact)
            pipeline = load_pipeline("subcategory_extraction", PipelineConfig(
                subcategory_extraction="learned_keywords", category_validation=None, draft_generation=None,
            ))
            if [set(pipeline.run(inp)) for inp in inputs[:n_test]] != predictions:
                raise ValueError("Предсказания модуля не совпадают с обученной моделью")
        logging.info("%s: %.4f, TP=%d FP=%d FN=%d", mode, results[mode]["micro_f1"],
                     results[mode]["tp"], results[mode]["fp"], results[mode]["fn"])
    for mode in ("keywords_local", "keywords_local_whole"):
        logits = np.full((len(inputs), len(categories)), -40.0)
        for index, cid in enumerate(categories):
            if cid == SOURCE:
                continue
            columns = selected[cid]
            context_train, context_held = context_features[cid]
            train_blocks = [binary[:, columns], context_train]
            held_blocks = [held_binary[:, columns], context_held]
            if mode == "keywords_local_whole":
                train_blocks.append(full)
                held_blocks.append(held)
            labels = [cid in row["target_mc_ids"] for row in train]
            clf = LogisticRegression(C=4, max_iter=500).fit(hstack(train_blocks, format="csr"), labels)
            logits[:, index] = clf.decision_function(hstack(held_blocks, format="csr"))
        results[mode] = evaluate(logits)
        logging.info("%s: %.4f (tuned %.4f)", mode, results[mode]["micro_f1"],
                     results[mode]["tuned_holdout"]["micro_f1"])
    args.report.write_text(json.dumps({
        "train_sha256": hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest(),
        "train_rows": len(train), "holdout_rows": len(indices),
        "overlap_removed": len(test) - len(indices),
        "holdout_groups": [rows[i]["group_id"] for i in indices],
        "validation_groups": [rows[i]["group_id"] for i in dev_indices], "threshold": 0.5,
        "keywords": keywords, "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
