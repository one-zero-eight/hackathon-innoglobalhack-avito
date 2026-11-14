"""Проверить происхождение и согласованность артефактов, не изменяя данные."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from main import MICROCATEGORIES
from dataset_annotation import build_improved_dataset as builder
from dataset_annotation.audit_dataset import DATA, WORK, normalize
from dataset_annotation.finalize_dataset_annotations import HISTORY, valid_labels
from dataset_annotation.review_dataset import USER_REVIEW_TYPES


def require(condition: bool, message: str) -> None:
    """Прервать проверку с указанием нарушенного инварианта."""
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict]:
    """Прочитать JSONL целиком без молчаливого пропуска ошибок."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def user_answers(decisions: dict) -> dict:
    """Выделить индивидуальные ответы, которые общие правила не меняют."""
    return {
        (split, decision["id"]): decision
        for split in ("train", "test") for decision in decisions[split]
        if decision.get("review_type") in USER_REVIEW_TYPES
    }


def main() -> None:
    """Проверить исходники, все группы, индивидуальные ответы и текущие выходы."""
    watched = {
        path: path.read_bytes()
        for path in builder.DESTINATION.rglob("*") if path.is_file()
    }
    decisions = json.loads(builder.DECISIONS.read_bytes())
    for name, expected in decisions["source_sha256"].items():
        require(builder.digest((DATA / name).read_bytes()) == expected,
                f"Изменён исходник: {name}")
    reverse = {name: cid for cid, name in MICROCATEGORIES.items()}
    originals = {
        "train": read_jsonl(DATA / builder.SOURCE_NAMES["train"]),
        "test": pq.read_table(DATA / builder.SOURCE_NAMES["test"]).to_pylist(),
    }
    groups = defaultdict(list)
    records = []
    for split, rows in originals.items():
        for row, original in enumerate(rows, 1):
            labels = sorted(
                entry["mcId"] for entry in original["found_microcategories"]
            ) if split == "train" else sorted(reverse[name] for name in original["categories"])
            require(valid_labels(labels), "Некорректные исходные метки")
            group_id = builder.digest(normalize(original["description"]).encode())
            record = builder.Record(split, row, original, labels, group_id)
            groups[split, group_id].append(record)
            records.append(record)
    require(len(records) == 10208 and len(groups) == 4993, "Изменился состав корпуса")
    answers = user_answers(decisions)
    require(len(answers) >= 24, "Потеряны индивидуальные ответы; проверьте историю")
    if HISTORY.exists():
        previous_answers = user_answers(json.loads(HISTORY.read_bytes()))
        require(all(answers.get(key) == value for key, value in previous_answers.items()),
                "Полная разметка изменила индивидуальные ответы пользователя")
    builder.apply_decisions(records, decisions)
    target_by_row = {(r.split, r.row): r for r in records}
    for split, name in builder.OUTPUT_NAMES.items():
        path = builder.DESTINATION / name
        if not path.exists():
            continue
        output = read_jsonl(path) if split == "train" else pq.read_table(path).to_pylist()
        seen = set()
        for result in output:
            key = split, builder.digest(normalize(result["description"]).encode())
            require(key in groups and key not in seen, "Неизвестная/повторная выходная группа")
            seen.add(key)
            require(any(result == member.output() for member in groups[key]),
                    "Изменены текст, схема или метки выходной записи")
        logging.info("%s: %d выходных групп; схема и поля сохранены.", split, len(output))
    for path in (WORK / "annotation-proposals.jsonl", builder.DESTINATION / "annotations.jsonl"):
        if not path.exists():
            logging.info("%s ещё не создан.", path.name)
            continue
        proposals = read_jsonl(path)
        indexed = {(p["split"], p["description_group_id"]): p for p in proposals}
        require(len(proposals) == len(indexed) and indexed.keys() == groups.keys(),
                f"Неполное покрытие: {path.name}")
        for key, proposal in indexed.items():
            require(valid_labels(proposal["labels"]), f"Некорректные метки: {path.name}")
            require(type(proposal["user_locked"]) is bool, "Нестрогий флаг user_locked")
            if path.name == "annotations.jsonl":
                require(not proposal["uncertainties"], "Не разрешены сомнения в итоговой разметке")
                require(all(proposal["labels"] == member.labels for member in groups[key]),
                        "Итоговая аннотация расходится с decisions")
        logging.info("%s: полное покрытие %d групп.", path.name, len(indexed))
    provenance_path = builder.DESTINATION / "provenance.jsonl"
    if provenance_path.exists():
        provenance = read_jsonl(provenance_path)
        require(len(provenance) == 10208 and len({
            (row["split"], row["source_row"]) for row in provenance
        }) == 10208, "Неполный журнал происхождения")
        included = set()
        for row in provenance:
            source = target_by_row[row["split"], row["source_row"]]
            require(row["description_group_id"] == source.group_id, "Изменена группа в provenance")
            if row["status"] in {"included", "deduplicated"}:
                included.add((source.split, source.group_id))
        require(not ({group for split, group in included if split == "train"}
                     & {group for split, group in included if split == "test"}),
                "Пересечение выходных train/test")
    if "full_annotation" in decisions:
        metadata = decisions["full_annotation"]
        for name, expected in (
            ("annotation_policy.json", metadata["policy_sha256"]),
            ("annotation_overrides.json", metadata["overrides_sha256"]),
            ("annotations.jsonl", metadata["annotations_sha256"]),
        ):
            require(builder.digest((builder.DESTINATION / name).read_bytes()) == expected,
                    f"Изменился артефакт полной разметки: {name}")
    else:
        logging.info("Полная переразметка ещё не применена.")
    require(all(path.read_bytes() == snapshot for path, snapshot in watched.items()),
            "Служебная проверка или параллельный процесс изменили рабочие данные")
    logging.info("Проверено: 10208 исходных строк, 4993 группы, %d ответов; данные не изменены.",
                 len(answers))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
