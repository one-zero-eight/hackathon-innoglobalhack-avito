"""Заморозить групповое разбиение и подготовить слепую ручную разметку."""

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit

from main import MICROCATEGORIES
from run import normalize_description, prepare_rows
from subcategory_extraction.snippet_logistic_regression.train import duplicate_groups

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/innoglobalhack_avito"
SOURCE_MC_ID = 101
TARGET_IDS = sorted(set(MICROCATEGORIES) - {SOURCE_MC_ID})


def split_groups(groups: np.ndarray) -> dict[str, np.ndarray]:
    """Разделить группы примерно в пропорции 70/15/15."""
    remaining, holdout = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        .split(groups, groups=groups)
    )
    fit, tune = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=43)
        .split(remaining, groups=groups[remaining])
    )
    return {"fit": remaining[fit], "tune": remaining[tune], "holdout": holdout}


def main() -> None:
    """Сохранить назначения, данные разработки и очередь без подсказок разметчику."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train_path = DATA / "public-train-annotated.jsonl"
    public_path = DATA / "public-test-dataset.parquet"
    # Зафиксированные исходные снимки сохраняют номера строк split-манифеста.
    with train_path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    train = prepare_rows(records, SOURCE_MC_ID)
    if any(type(cid) is not int or cid not in TARGET_IDS for row in train for cid in row.expected):
        raise ValueError("В исходном train есть метки вне контракта")
    public_texts = [row["description"] for row in pq.read_table(public_path, columns=["description"]).to_pylist()]

    descriptions = [row.inp.description for row in train]
    exact = [normalize_description(text) for text in descriptions]
    # Для поиска общих шаблонов игнорируем декоративную пунктуацию.
    lexical = [" ".join(re.findall(r"[^\W_]+", normalize_description(text))) or normalize_description(text)
               for text in descriptions + public_texts]
    groups, grouping = duplicate_groups(lexical)
    train_groups = groups[:len(train)]
    public_groups = np.unique(groups[len(train):])
    quarantine = np.isin(train_groups, public_groups)
    eligible = np.flatnonzero(~quarantine)
    parts = {name: eligible[local] for name, local in split_groups(train_groups[eligible]).items()}
    parts["excluded_public_overlap"] = np.flatnonzero(quarantine)

    partition_groups = [set(train_groups[indices]) for indices in parts.values()]
    assert sum(map(len, partition_groups)) == len(set(train_groups))
    assert sum(map(len, parts.values())) == len(records)
    assert not set.union(*partition_groups[:3]).intersection(public_groups)

    assignments = [{} for _ in records]
    summary = {}
    for name, indices in parts.items():
        label_counts = Counter(cid for i in indices for cid in train[i].expected)
        if name != "excluded_public_overlap" and any(
            not 0 < label_counts[cid] < len(indices) for cid in TARGET_IDS
        ):
            raise ValueError(f"В {name} нужны оба класса для каждой категории")
        summary[name] = {
            "rows": len(indices),
            "groups": len(set(train_groups[indices])),
            "largest_group_rows": max(Counter(train_groups[indices]).values(), default=0),
            "exact_descriptions": len({exact[i] for i in indices}),
            "positive_labels": {cid: label_counts[cid] for cid in TARGET_IDS},
        }
        for i in indices:
            assignments[i] = {"source_line": int(i) + 1, "group": int(train_groups[i]), "split": name}

    copies = defaultdict(list)
    for i in sorted(parts["holdout"]):
        copies[exact[i]].append(i)
    review = []
    review_index = []
    conflicts = 0
    for indices in copies.values():
        i = indices[0]
        review_id = f"h-{i + 1}"
        review.append({"review_id": review_id, "description": descriptions[i], "labels": None})
        review_index.append({
            "review_id": review_id,
            "group": int(train_groups[i]),
            "source_lines": [int(j) + 1 for j in indices],
        })
        conflicts += len({frozenset(train[j].expected) for j in indices}) > 1

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in {
        "assignments": assignments,
        "fit": (records[i] for i in sorted(parts["fit"])),
        "tune": (records[i] for i in sorted(parts["tune"])),
        "holdout-review": review,
        "holdout-index": review_index,
    }.items():
        with (output / f"{name}.jsonl").open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "public_test_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grouping": {**grouping, "input": "normalized_alphanumeric_tokens", "seeds": [42, 43]},
        "partitions": summary,
        "holdout_review": {"tasks": len(review), "exact_label_conflicts": conflicts},
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(output.glob("*.jsonl"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logging.info("Разбиение %s: %s", output, {name: len(indices) for name, indices in parts.items()})
    logging.info("Ручная разметка: %d описаний, %d конфликтов исходных меток", len(review), conflicts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
