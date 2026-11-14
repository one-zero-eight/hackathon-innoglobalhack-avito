"""Аудит структуры, повторов и спорных меток без обучения классификатора."""

from pathlib import Path

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from main import MICROCATEGORIES

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/innoglobalhack_avito"
WORK = Path(__file__).resolve().parent / "work"
OUTPUT = WORK / "dataset-audit.json"
SIMILARITY_THRESHOLD = 0.92
TAIL = re.compile(
    r"\n[^\n]*(?:характеристики(?: и параметры)?|артикул|уникальный код услуги)"
    r"[^\n]*[:：]", re.I,
)
# Только сигналы для ручного разбора: совпадение не доказывает предложение услуги.
SIGNALS = {
    102: r"штукатур\w*",
    103: r"(?:уклад\w*|облицов\w*|монтаж\w*|затир\w*|запил\w*)"
         r"[^\n.!?;]{0,65}(?:плитк\w*|кафел\w*|керамогранит\w*|мозаик\w*|шв\w*)",
    104: r"сантехнич\w*\s+работ\w*|разводк\w*\s+(?:труб\w*|вод\w*|канализац\w*)"
         r"|(?:установк\w*|замен\w*|монтаж\w*)[^\n.!?;]{0,45}"
         r"(?:унитаз\w*|смесител\w*|раковин\w*|сантехник\w*|труб\w*)",
    105: r"электромонтаж\w*|(?:замен\w*|монтаж\w*|прокладк\w*|установк\w*)"
         r"[^\n.!?;]{0,45}(?:проводк\w*|кабел\w*|розет\w*|электрощит\w*|светильник\w*)",
    106: r"(?:поклейк\w*|покле\w*|оклейк\w*)\s+обо\w*"
         r"|(?:покраск\w*|окраск\w*|окрашив\w*)\s+(?:стен\w*|потолк\w*)"
         r"|малярн\w*\s+работ\w*|шпа[кт]л[её]в\w*",
}
NEGATION = re.compile(r"\b(?:не\s+(?:выполня|дела|занима|оказыва)|без\s+|кроме\s+)", re.I)


def normalize(text: str) -> str:
    """Нормализовать регистр, ё, Unicode и пробелы."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def excerpt(text: str, start: int, end: int) -> str:
    """Сохранить контекст совпадения для последующей ручной проверки."""
    return text[max(0, start - 100):min(len(text), end + 160)]


def load_rows() -> tuple[list[dict], dict]:
    """Прочитать все строки, собрать ошибки схемы и происхождение файлов."""
    rows, errors = [], []
    train_path = DATA / "public-train-annotated.jsonl"
    test_path = DATA / "public-test-dataset.parquet"
    reverse = {name: cid for cid, name in MICROCATEGORIES.items()}
    for index, line in enumerate(train_path.read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(line)
        found = raw["found_microcategories"]
        labels = [entry["mcId"] for entry in found]
        if any(cid not in MICROCATEGORIES or cid == 101 for cid in labels):
            errors.append({"row": index, "error": "invalid_train_ids"})
        if len(labels) != len(set(labels)):
            errors.append({"row": index, "error": "duplicate_train_labels"})
        for entry in found:
            if MICROCATEGORIES.get(entry["mcId"]) != entry["label"]:
                errors.append({"row": index, "error": "label_name_mismatch"})
        rows.append({
            "split": "train", "row": index, "item_id": raw["item_id"],
            "title": raw["title"], "description": raw["description"],
            "labels": sorted(set(labels)),
        })
    for index, raw in enumerate(pq.read_table(test_path).to_pylist(), 1):
        categories = raw["categories"]
        unknown = [name for name in categories if name not in reverse]
        if unknown:
            errors.append({"split": "test", "row": index, "error": "unknown_categories",
                           "values": unknown})
        if len(categories) != len(set(categories)):
            errors.append({"split": "test", "row": index, "error": "duplicate_labels"})
        rows.append({
            "split": "test", "row": index, "title": raw["title"],
            "description": raw["description"],
            "labels": sorted(reverse[name] for name in categories if name in reverse),
        })
    for row in rows:
        if not isinstance(row["description"], str) or not row["description"].strip():
            errors.append({"split": row["split"], "row": row["row"],
                           "error": "empty_or_invalid_description"})
    return rows, {
        "schema_errors": errors,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (train_path, test_path)},
    }


def duplicate_groups(rows: list[dict], key: str) -> list[list[int]]:
    """Сгруппировать записи по точному или нормализованному описанию."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        text = row["description"]
        groups[text if key == "exact" else normalize(text)].append(index)
    return list(groups.values())


def describe_groups(rows: list[dict], groups: list[list[int]]) -> dict:
    """Посчитать повторы, противоречивые метки и пересечения частей."""
    conflicts = [g for g in groups if len({tuple(rows[i]["labels"]) for i in g}) > 1]
    cross = [g for g in groups if len({rows[i]["split"] for i in g}) > 1]
    return {
        "unique_descriptions": len(groups),
        "duplicate_groups": sum(len(g) > 1 for g in groups),
        "redundant_rows": sum(len(g) - 1 for g in groups),
        "conflicting_groups": conflicts,
        "cross_split_groups": cross,
        "test_rows_overlapping_train": sum(
            rows[i]["split"] == "test" for g in cross for i in g
        ),
    }


def semantic_candidates(rows: list[dict]) -> dict:
    """Найти лексические сигналы для ручного разбора, не исправляя метки."""
    missing, unsupported, negatives = [], [], []
    for index, row in enumerate(rows):
        text = row["description"]
        for cid, pattern in SIGNALS.items():
            matches = list(re.finditer(pattern, text, re.I))
            if matches and cid not in row["labels"]:
                missing.append({
                    "index": index, "category": cid,
                    "evidence": [excerpt(text, m.start(), m.end()) for m in matches[:3]],
                })
            if cid in row["labels"] and not matches:
                unsupported.append({"index": index, "category": cid})
        for match in NEGATION.finditer(text):
            negatives.append({"index": index, "evidence": excerpt(text, match.start(), match.end())})
    return {"missing_signal": missing, "label_without_signal": unsupported,
            "negation_candidates": negatives}


def near_duplicates(rows: list[dict], groups: list[list[int]]) -> dict:
    """Сравнить все уникальные описания по символьному cosine, без классификации."""
    representatives = [g[0] for g in groups]
    # Длинные разделители из подчёркиваний не должны создавать сходство.
    texts = [
        re.sub(r"[\W_]+", " ", normalize(rows[i]["description"])).strip()
        for i in representatives
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=100_000, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    pairs = []
    for start in range(0, len(texts), 128):
        similarities = (matrix[start:start + 128] @ matrix.T).tocoo()
        selected = (similarities.data >= SIMILARITY_THRESHOLD) & (
            similarities.col > similarities.row + start
        )
        for a, b, similarity in zip(
            similarities.row[selected] + start, similarities.col[selected],
            similarities.data[selected], strict=True,
        ):
            left, right = representatives[int(a)], representatives[int(b)]
            pairs.append({
                "left": left, "right": right, "cosine": float(similarity),
                "different_labels": rows[left]["labels"] != rows[right]["labels"],
                "left_group": int(a), "right_group": int(b),
            })
    return {
        "method": "Удалена пунктуация; char_wb TF-IDF 3–5, min_df=2, "
                  "max_features=100000; не семантическая модель",
        "threshold": SIMILARITY_THRESHOLD,
        "pairs": sorted(pairs, key=lambda pair: -pair["cosine"]),
    }


def main() -> None:
    """Выполнить полный аудит и сохранить воспроизводимые диагностические данные."""
    rows, report = load_rows()
    report["records"] = rows
    for split in ("train", "test"):
        part = [row for row in rows if row["split"] == split]
        distribution = Counter(cid for row in part for cid in row["labels"])
        normalized = duplicate_groups(part, "normalized")
        ids = Counter(row.get("item_id") for row in part) if split == "train" else Counter()
        report[split] = {
            "rows": len(part),
            "empty_targets": sum(not row["labels"] for row in part),
            "label_distribution": dict(distribution),
            "label_sets": dict(Counter(str(row["labels"]) for row in part)),
            "unique_normalized": len(normalized),
            "exact_duplicates": describe_groups(part, duplicate_groups(part, "exact")),
            "repeated_item_ids": sum(count > 1 for count in ids.values()),
            "description_length": {
                "min": min(len(row["description"]) for row in part),
                "max": max(len(row["description"]) for row in part),
            },
        }
    for key in ("exact", "normalized"):
        groups = duplicate_groups(rows, key)
        report[key] = describe_groups(rows, groups)
    groups = duplicate_groups(rows, "normalized")
    report["normalized_groups"] = groups
    cores = defaultdict(list)
    for index, row in enumerate(rows):
        core = normalize(TAIL.split(row["description"], maxsplit=1)[0])
        cores[core].append(index)
    report["without_characteristics_tail"] = describe_groups(rows, list(cores.values()))
    report["candidates"] = semantic_candidates(rows)
    report["near_duplicates"] = near_duplicates(rows, groups)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Аудит сохранён: %s", OUTPUT)
    logging.info("Train: %s; test: %s", report["train"], report["test"])
    for key in ("exact", "normalized", "without_characteristics_tail"):
        logging.info("%s: %s", key, {
            k: len(v) if isinstance(v, list) else v for k, v in report[key].items()
        })
    logging.info("Близкие пары: %d; с разными метками: %d",
                 len(report["near_duplicates"]["pairs"]),
                 sum(pair["different_labels"] for pair in report["near_duplicates"]["pairs"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
