"""Выбор трудных train-примеров и привязанная к хешам проверка меток/весов.

Отбор использует локальный CPU baseline; метки автоматически не меняются.
Решения из results/rubert-hard-negatives.json — смысловая проверка ассистента,
не независимо подтверждённый ручной эталон.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np

from main import MICROCATEGORIES

REVIEW_SOURCE = "assistant_semantic_review_not_independent_gold"
EXPECTED_TRAIN_ROWS = 3196
HARD_NEGATIVE_WEIGHT = 1.5
logger = logging.getLogger(__name__)


def description_sha256(description: str) -> str:
    """Хешировать точное UTF-8 описание без нормализации пробелов."""
    return hashlib.sha256(description.encode("utf-8")).hexdigest()


def _targets(values) -> set[int]:
    values = list(values)
    if any(type(value) is not int or value not in MICROCATEGORIES for value in values):
        raise ValueError("Review targets must be integer microcategory IDs")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate target in review or training row")
    return set(values)


def ordered_rows_sha256(rows: list[dict]) -> str:
    """Привязать строки trainer к исходным описаниям, меткам и порядку."""
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            "description_sha256": description_sha256(row["description"]),
            "targets": sorted(_targets(row["targets"])),
        }
        digest.update((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_reviewed_rows(rows: list[dict], review_path: Path) -> tuple[list[dict], np.ndarray, dict]:
    """Применить решения только к точному упорядоченному train из3196 строк.

    Вернуть копии строк с независимыми наборами меток, float32-веса и счётчики.
    Исходный текст не меняется. Неопределённые случаи сохраняют метки и вес1.
    Подтверждённые трудные отрицательные примеры и однозначные исправления
    получают вес1.5. Неизвестные/повторные хеши и изменённые исходные метки
    отвергаются до возврата результата. Остальные части данных не читаются.
    """
    path = Path(review_path)
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("version") != 1 or review.get("source") != REVIEW_SOURCE:
        raise ValueError("Unsupported semantic review provenance/version")
    source = review["source_train"]
    if len(rows) != EXPECTED_TRAIN_ROWS or source.get("rows") != EXPECTED_TRAIN_ROWS:
        raise ValueError("Semantic review applies only to the original 3196 training rows")
    if ordered_rows_sha256(rows) != source.get("ordered_rows_sha256"):
        raise ValueError("Training descriptions, original targets, or source order changed")
    entries = review["entries"]
    if not isinstance(entries, list) or len(entries) > 64:
        raise ValueError("Expected at most 64 explicit semantic review entries")
    positions = {}
    for index, row in enumerate(rows):
        digest = description_sha256(row["description"])
        if digest in positions:
            raise ValueError("Duplicate training description hash")
        positions[digest] = index
    output = [{**row, "targets": _targets(row["targets"])} for row in rows]
    weights = np.ones(len(rows), dtype=np.float32)
    seen = set()
    counts = {"reviewed": 0, "effective_corrections": 0, "weighted": 0, "uncertain": 0, "kept": 0}
    for entry in entries:
        digest = entry["description_sha256"]
        if digest in seen:
            raise ValueError("Duplicate review description hash")
        seen.add(digest)
        if digest not in positions:
            raise ValueError("Reviewed hash not found exactly once in training rows")
        index = positions[digest]
        if entry.get("train_index") != index:
            raise ValueError("Reviewed row source index changed")
        if entry.get("source") != REVIEW_SOURCE or not entry.get("semantic_reason"):
            raise ValueError("Every decision requires explicit semantic review provenance/reason")
        original = _targets(entry["original_targets"])
        reviewed = _targets(entry["reviewed_targets"])
        if original != output[index]["targets"]:
            raise ValueError("Reviewed original targets do not match training row")
        if "description" in entry and entry["description"] != rows[index]["description"]:
            raise ValueError("Review text does not match the exact raw training description")
        decision = entry["decision"]
        verified = entry.get("verified_hard_negative", False)
        unambiguous = entry.get("unambiguous_policy_violation", False)
        if type(verified) is not bool or type(unambiguous) is not bool:
            raise ValueError("Review confidence flags must be booleans")
        if decision in {"keep", "uncertain"}:
            if reviewed != original or unambiguous:
                raise ValueError("Kept/uncertain reviews cannot change labels")
            if decision == "uncertain":
                if verified:
                    raise ValueError("Uncertain reviews cannot be upweighted")
                counts["uncertain"] += 1
            else:
                counts["kept"] += 1
                if verified:
                    weights[index] = HARD_NEGATIVE_WEIGHT
        elif decision == "correct":
            if not unambiguous or reviewed == original:
                raise ValueError("Corrections require an explicit unambiguous policy violation")
            output[index]["targets"] = reviewed
            weights[index] = HARD_NEGATIVE_WEIGHT
            counts["effective_corrections"] += 1
        else:
            raise ValueError("Unknown review decision")
        counts["reviewed"] += 1
    counts["weighted"] = int(np.count_nonzero(weights != 1.0))
    metadata = {
        **counts,
        "review_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_train_sha256": source["sha256"],
        "ordered_rows_sha256": source["ordered_rows_sha256"],
        "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
        "source": REVIEW_SOURCE,
        "independent_gold": False,
    }
    return output, weights, metadata


def candidate_flags(description: str) -> dict[str, bool]:
    """Использовать признаки только для очереди чтения, не для назначения меток."""
    return {
        "separate_refusal": bool(re.search(
            r"(?:отдельн\w*\s+работ\w*.{0,35}(?:не\s+бер|не\s+выполня)|не\s+бер.{0,35}отдельн|только\s+под\s+ключ|только\s+комплекс)",
            description, re.IGNORECASE | re.DOTALL,
        )),
        "negation": bool(re.search(r"\bне\s+(?:дела|выполня|занима|бер|оказыва)", description, re.IGNORECASE)),
        "seo": bool(re.search(r"для\s+поиска|ищут|теги|поисков|SEO|ключевые\s+слов", description, re.IGNORECASE)),
    }


def select_review_candidates(rows: list[dict], probabilities: np.ndarray, classes: list[int]) -> list[dict]:
    """Выбрать48 максимальных отсутствующих меток и до16 трудных пустых ответов.

    Ранжирование — по максимальной вероятности отсутствующей метки без101,
    при равенстве по исходному индексу. Для дополнительной пустой страты
    приоритетны признаки отказа/отрицания/SEO. Описания не повторяются.
    Это кандидаты по предварительным меткам, не подтверждённые ошибки.
    """
    values = np.asarray(probabilities)
    if values.shape != (len(rows), len(classes)) or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite [rows, classes] matrix")
    if set(classes) != set(MICROCATEGORIES) or len(classes) != len(set(classes)):
        raise ValueError("Expected every microcategory exactly once")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Expected probabilities in [0, 1]")
    ranked = []
    for index, row in enumerate(rows):
        targets = _targets(row["targets"])
        absent = [column for column, category in enumerate(classes) if category not in targets and category != 101]
        score = max((float(values[index, column]) for column in absent), default=0.0)
        ranked.append({
            "index": index, "score": score, "targets": sorted(targets),
            "flags": candidate_flags(row["description"]),
        })
    false_positives = sorted((row for row in ranked if row["score"] >= 0.5), key=lambda row: (-row["score"], row["index"]))
    empty = sorted(
        (row for row in ranked if not row["targets"] and row["score"] < 0.5),
        key=lambda row: (-int(any(row["flags"].values())), -row["score"], row["index"]),
    )
    selected = []
    seen = set()
    for pool, limit, stratum in ((false_positives, 48, "top_false_positive"), (empty, 16, "predicted_correct_empty")):
        count = 0
        for candidate in pool:
            digest = description_sha256(rows[candidate["index"]]["description"])
            if digest in seen:
                continue
            seen.add(digest)
            selected.append({**candidate, "stratum": stratum, "description_sha256": digest})
            count += 1
            if count == limit:
                break
    return selected


def score_training_rows(train_path: Path, artifact: Path, batch_size: int = 16) -> dict:
    """Выполнить настоящий локальный CPU-инференс только на исходном train."""
    import torch

    from subcategory_extraction.semantic_encoder.model import (
        collate_documents, load_resource, tokenize_document,
    )

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data = Path(train_path).read_bytes()
    raw_rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    rows = [{"description": row["description"], "targets": {item["mcId"] for item in row["found_microcategories"]}} for row in raw_rows]
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Review is restricted to the original 3196-row training split")
    resource = load_resource(Path(artifact), "cpu")
    digest = hashlib.sha256(data).hexdigest()
    if resource.metadata.get("train_sha256") != digest:
        raise ValueError("Baseline was not trained on these exact training bytes")
    if resource.metadata.get("train_rows") != len(rows):
        raise ValueError("Baseline training row count mismatch")
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            documents = [tokenize_document(resource.tokenizer, row["description"]) for row in rows[start:start + batch_size]]
            batch = collate_documents(resource.tokenizer, documents)
            probabilities.extend(torch.sigmoid(resource.model(batch)).cpu().tolist())
    return {
        "source_train_sha256": digest,
        "ordered_rows_sha256": ordered_rows_sha256(rows),
        "classes": resource.metadata["classes"],
        "probabilities": probabilities,
        "candidates": select_review_candidates(rows, np.asarray(probabilities), resource.metadata["classes"]),
        "inference_device": "cpu",
        "baseline_training_metadata": resource.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {args.train.resolve(), args.artifact.resolve()}:
        raise ValueError("Scoring output must be a new file, not an input or artifact")
    result = score_training_rows(args.train, args.artifact, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("Сохранено %d кандидатов: %s", len(result["candidates"]), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
