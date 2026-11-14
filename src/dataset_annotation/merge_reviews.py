"""Объединить независимые части смыслового разбора без потери исключений."""

import argparse
import json
import logging
from pathlib import Path

from dataset_annotation import build_improved_dataset as builder
from dataset_annotation.audit_dataset import WORK, normalize
from dataset_annotation.finalize_dataset_annotations import OVERRIDES, atomic_json, valid_labels
from dataset_annotation.review_dataset import session_lock


def read_jsonl(path: Path) -> list[dict]:
    """Прочитать полную часть разбора без пропуска повреждённых строк."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def collect_reviews(count: int, resolutions: dict | None = None,
                    allow_partial: bool = False) -> tuple[list[dict], dict]:
    """Проверить покрытие каждой части и сохранить основания решений."""
    resolutions = resolutions or {}
    results = []
    seen = set()
    hashes = {}
    for index in range(1, count + 1):
        source = WORK / f"review-shard-{index}.jsonl"
        result = WORK / f"review-result-{index}.jsonl"
        if not result.exists() and allow_partial:
            continue
        inputs = {
            (row["split"], row["row"], row["description_group_id"]): row
            for row in read_jsonl(source)
        }
        rows = read_jsonl(result)
        keys = [(row["split"], row["row"], row["description_group_id"]) for row in rows]
        if (len(keys) != len(set(keys)) or not set(keys) <= inputs.keys()
                or (not allow_partial and set(keys) != inputs.keys())):
            raise ValueError(f"Неполная или повторная часть {index}")
        for row, key in zip(rows, keys, strict=True):
            if key in seen:
                raise ValueError("Одно описание разобрано в нескольких частях")
            seen.add(key)
            if not valid_labels(row["labels"]) or not isinstance(row.get("reason"), str):
                raise ValueError(f"Некорректное решение: {key}")
            if not row["reason"].strip() or type(row.get("needs_user_question")) is not bool:
                raise ValueError(f"Нет основания или статуса вопроса: {key}")
            if row["needs_user_question"]:
                resolution = resolutions.get((row["split"], row["row"]), {})
                if not resolution.get("question_resolution") and not allow_partial:
                    raise ValueError(f"Нужен ответ для {key}: {row.get('question')}")
            evidence = row.get("evidence", [])
            if allow_partial and isinstance(evidence, str):
                evidence = [evidence] if evidence else []
            if allow_partial and isinstance(evidence, dict):
                evidence = [
                    quote for values in evidence.values()
                    for quote in (values if isinstance(values, list) else [values])
                ]
            if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
                raise ValueError(f"Некорректные цитаты: {key}")
            if row["labels"] and not evidence and not allow_partial:
                raise ValueError(f"Непустые метки без цитат: {key}")
            description = normalize(inputs[key]["description"])
            if (not allow_partial and any(
                not quote.strip() or normalize(quote) not in description for quote in evidence
            )):
                raise ValueError(f"Цитата отсутствует в полном исходном описании: {key}")
            results.append({
                "split": row["split"], "row": row["row"],
                "description_group_id": row["description_group_id"],
                "labels": row["labels"], "reason": row["reason"],
                "evidence": evidence,
                "review_source": f"parallel-semantic-review-shard-{index}",
                "verification_status": "not_independently_verified" if allow_partial else "checked",
                "evidence_verified": not allow_partial,
                "pending_question": row.get("question") if row["needs_user_question"]
                and not resolutions.get((row["split"], row["row"])) else None,
            })
        hashes[str(index)] = {
            "input_sha256": builder.digest(source.read_bytes()),
            "result_sha256": builder.digest(result.read_bytes()),
            "groups": len(rows),
        }
    return results, hashes


def main() -> None:
    """Сохранить проверенные результаты только по явному --apply."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Собрать сохранённые части без смысловой перепроверки; вопросы оставить")
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards должен быть положительным")
    with session_lock():
        snapshot = OVERRIDES.read_bytes()
        overrides = json.loads(snapshot)
        resolutions = {
            (row["split"], row["row"]): row for row in overrides["decisions"]
            if row.get("question_resolution")
        }
        reviews, hashes = collect_reviews(args.shards, resolutions, args.allow_partial)
        # Отдельно проверенные главным агентом случаи приоритетнее первичного
        # параллельного разбора. Индивидуальные ответы защищены финализатором.
        previous = [
            row for row in overrides["decisions"]
            if not row.get("review_source", "").startswith("parallel-semantic-review-shard-")
        ]
        manual_keys = {(row["split"], row["row"]) for row in previous}
        merged = previous + [
            row for row in reviews if (row["split"], row["row"]) not in manual_keys
        ]
        overrides["decisions"] = sorted(
            merged, key=lambda row: (row["split"] != "train", row["row"]),
        )
        overrides["parallel_review"] = {
            "shards": hashes, "reviewed_groups": len(reviews),
            "verification_status": "not_independently_verified" if args.allow_partial else "checked",
            "models_used": ["stepfun/step-3.5-flash-2603", "openai/gpt-6.1-sol"],
            "model_disclosure": "Недорогие первичные разборы; часть с выявленными ошибками повторно проверяется Sol.",
            "method": "Смысловой разбор субагентами по полным description; не ручная человеческая разметка.",
        }
        logging.info("Проверено %d частей / %d описаний; объединено уточнений: %d.",
                     args.shards, len(reviews), len(merged))
        if args.apply:
            if OVERRIDES.read_bytes() != snapshot:
                raise ValueError("Уточнения изменились во время объединения")
            atomic_json(OVERRIDES, overrides)
            logging.info("Уточнения сохранены; датасет ещё не пересобран.")
        else:
            logging.info("Проверка без записи. Для сохранения добавьте --apply.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
