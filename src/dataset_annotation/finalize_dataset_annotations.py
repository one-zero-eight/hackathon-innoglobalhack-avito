"""Проверить и применить полную разметку, сохранив индивидуальные ответы."""

import argparse
import copy
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from dataset_annotation import build_improved_dataset as builder
from dataset_annotation.review_dataset import USER_REVIEW_TYPES, session_lock
from dataset_annotation.audit_dataset import WORK

POLICY = builder.DESTINATION / "annotation_policy.json"
PROPOSALS = WORK / "annotation-proposals.jsonl"
HISTORY = builder.DESTINATION / "review_history/pre-full-annotation.json"
ANNOTATIONS = builder.DESTINATION / "annotations.jsonl"
OVERRIDES = builder.DESTINATION / "annotation_overrides.json"
REVIEW_TYPE = "assistant_policy_annotation"


def atomic_json(path: Path, value: dict) -> None:
    """Атомарно сохранить JSON, не оставляя незавершённого файла."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
            prefix=f".{path.name}-", suffix=".tmp",
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def valid_labels(labels: object) -> bool:
    """Проверить строгий набор дополнительных ID без bool и повторов."""
    return (
        isinstance(labels, list)
        and all(type(cid) is int and cid in range(102, 107) for cid in labels)
        and labels == sorted(set(labels))
    )


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    """Заменить журнал целиком только после успешной записи всех строк."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
            prefix=f".{path.name}-", suffix=".tmp",
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare(decisions: dict, proposals: list[dict], overrides: dict,
            records: list[builder.Record]) -> tuple[dict, list[dict]]:
    """Проверить полное покрытие, приоритет пользователя и разрешение сомнений."""
    if overrides.get("policy_id") != "user-policy-full-dataset-v2":
        raise ValueError("Смысловые уточнения относятся к другой политике")
    groups = defaultdict(list)
    lookup = {}
    for record in records:
        groups[record.split, record.group_id].append(record)
        lookup[record.split, record.row] = record
    locked = {}
    for split in ("train", "test"):
        for decision in decisions[split]:
            if decision.get("review_type") in USER_REVIEW_TYPES:
                if decision["action"] not in {"correct", "retain"}:
                    raise ValueError("Пользовательское действие требует отдельного разбора")
                if (split, decision["row"]) not in lookup:
                    raise ValueError("Неизвестная строка в пользовательском решении")
                key = split, lookup[split, decision["row"]].group_id
                target = decision.get("new_labels", decision["expected_labels"])
                if key in locked or not valid_labels(target):
                    raise ValueError("Повторное пользовательское решение или некорректные метки")
                locked[key] = target
    indexed = {}
    for proposal in proposals:
        if (type(proposal.get("row")) is not int
                or type(proposal.get("user_locked")) is not bool
                or not isinstance(proposal.get("uncertainties"), list)
                or not all(isinstance(value, str) for value in proposal["uncertainties"])
                or not isinstance(proposal.get("evidence"), dict)
                or not isinstance(proposal.get("review_method"), str)):
            raise ValueError("Некорректная схема предложения разметки")
        key = proposal["split"], proposal["description_group_id"]
        if key not in groups or key in indexed:
            raise ValueError(f"Неизвестное/повторное описание: {key}")
        anchor = groups[key][0]
        if proposal["row"] != anchor.row:
            raise ValueError(f"Изменился порядок исходников: {key}")
        expected = sorted({tuple(r.labels) for r in groups[key]})
        if sorted(tuple(values) for values in proposal["source_labels"]) != expected:
            raise ValueError(f"Не совпадают исходные метки: {key}")
        if not valid_labels(proposal["labels"]):
            raise ValueError(f"Некорректные предложенные метки: {key}")
        if proposal["user_locked"] != (key in locked):
            raise ValueError(f"Изменились индивидуальные решения: {key}")
        if key in locked and proposal["labels"] != locked[key]:
            raise ValueError(f"Попытка изменить ответ пользователя: {key}")
        indexed[key] = copy.deepcopy(proposal)
        if key in locked:
            indexed[key]["uncertainties"] = []
    if indexed.keys() != groups.keys():
        raise ValueError("Не все исходные группы размечены")
    seen_overrides = set()
    for override in overrides.get("decisions", []):
        if (override["split"], override["row"]) not in lookup:
            raise ValueError("Неизвестная строка в смысловом уточнении")
        record = lookup[override["split"], override["row"]]
        key = record.split, record.group_id
        if key in locked or key in seen_overrides:
            raise ValueError(f"Перезапись пользователя/повторное уточнение: {key}")
        seen_overrides.add(key)
        if not valid_labels(override["labels"]) or not override["reason"]:
            raise ValueError(f"Некорректное смысловое уточнение: {key}")
        indexed[key].update(
            labels=override["labels"], uncertainties=[],
            semantic_review=override["reason"],
            semantic_evidence=override.get("evidence", []),
            review_method="assistant_semantic_review_with_user_policy",
        )
    unresolved = [
        f"{p['split']}:{p['row']}: {p['uncertainties']}"
        for p in indexed.values() if p["uncertainties"] and not p["user_locked"]
    ]
    if unresolved:
        raise ValueError("Нужен разбор неопределённости:\n" + "\n".join(unresolved))
    updated = copy.deepcopy(decisions)
    for split in ("train", "test"):
        previous = [d for d in decisions[split]
                    if d.get("review_type") not in USER_REVIEW_TYPES
                    and d.get("review_type") != REVIEW_TYPE]
        if previous:
            archive = updated.setdefault("superseded_decisions", [])
            for decision in previous:
                archived = copy.deepcopy(decision)
                archived["superseded_by"] = "user-policy-full-dataset-v2"
                archived["split"] = split
                archive.append(archived)
        updated[split] = [d for d in decisions[split]
                          if d.get("review_type") in USER_REVIEW_TYPES]
    annotations = [indexed[key] for key in groups]
    for key, members in groups.items():
        if key in locked:
            continue
        proposal = indexed[key]
        anchor = members[0]
        target = proposal["labels"]
        decision = {
            "id": f"policy-v2-{anchor.split}-{anchor.row}",
            "row": anchor.row,
            "expected_labels": anchor.labels,
            "new_labels": target,
            "action": "correct" if any(r.labels != target for r in members) else "retain",
            "review_type": REVIEW_TYPE,
            "description_group_id": anchor.group_id,
            "policy_id": "user-policy-full-dataset-v2",
            "review_method": proposal["review_method"],
            "reason": proposal.get("semantic_review")
            or ("Явный отказ от отдельных работ: " + proposal["restriction"]
                if proposal.get("restriction")
                else "Применены общие пользовательские правила к description без SEO; "
                     "основания по категориям сохранены в annotations.jsonl."),
        }
        if anchor.split == "train":
            decision["item_id"] = anchor.original["item_id"]
        updated[anchor.split].append(decision)
    updated["version"] = 2
    updated["annotation_policy"] = json.loads(POLICY.read_text(encoding="utf-8"))
    updated["full_annotation"] = {
        "groups": len(groups), "individual_user_exceptions": len(locked),
        "unresolved_semantic_groups": 0,
        "policy_sha256": builder.digest(POLICY.read_bytes()),
        "overrides_sha256": builder.digest(OVERRIDES.read_bytes()),
        "finalizer_sha256": builder.digest(Path(__file__).read_bytes()),
        "annotations_file": ANNOTATIONS.name,
        "proposals_file_sha256": builder.digest(
            b"".join(json.dumps(p, sort_keys=True, ensure_ascii=False).encode() + b"\n"
                     for p in proposals)
        ),
        "method": "Автоматические правила + отдельные смысловые уточнения; не ручной эталон.",
    }
    return updated, annotations


def main() -> None:
    """Проверить предложения; записать и пересобрать только с явным --apply."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, default=PROPOSALS)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    with session_lock():
        snapshot = builder.DECISIONS.read_bytes()
        policy_snapshot = POLICY.read_bytes()
        overrides_snapshot = OVERRIDES.read_bytes()
        proposals_snapshot = args.proposals.read_bytes()
        decisions = json.loads(snapshot)
        records, _ = builder.load_records(
            decisions, json.loads(builder.AUDIT.read_text(encoding="utf-8")),
        )
        proposals = [json.loads(line) for line in proposals_snapshot.decode("utf-8").splitlines()]
        overrides = json.loads(overrides_snapshot)
        updated, annotations = prepare(decisions, proposals, overrides, records)
        logging.info("Проверено групп: %d; индивидуальных исключений: %d.",
                     len(annotations), updated["full_annotation"]["individual_user_exceptions"])
        if not args.apply:
            logging.info("Режим проверки. Для сохранения добавьте --apply.")
            return
        if builder.DECISIONS.read_bytes() != snapshot:
            raise ValueError("Решения изменились во время проверки; запись отменена")
        if (POLICY.read_bytes() != policy_snapshot or OVERRIDES.read_bytes() != overrides_snapshot
                or args.proposals.read_bytes() != proposals_snapshot):
            raise ValueError("Политика или предложения изменились во время проверки")
        for name, expected in decisions["source_sha256"].items():
            if builder.digest((builder.DATA / name).read_bytes()) != expected:
                raise ValueError("Исходники изменились во время проверки")
        if not HISTORY.exists():
            atomic_json(HISTORY, decisions)
        if "full_annotation" in decisions:
            previous = HISTORY.parent / f"policy-v2-{builder.digest(snapshot)[:16]}.json"
            if not previous.exists():
                atomic_json(previous, decisions)
        atomic_jsonl(ANNOTATIONS, annotations)
        updated["full_annotation"]["annotations_sha256"] = builder.digest(ANNOTATIONS.read_bytes())
        atomic_json(builder.DECISIONS, updated)
        logging.info("Решения сохранены; пересобираю датасет.")
        builder.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
