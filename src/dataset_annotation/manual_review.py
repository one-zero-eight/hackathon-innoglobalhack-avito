"""Подготовить слепые пакеты и проверить смысловую разметку агентов без классификации."""

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import unicodedata

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/innoglobalhack_avito"
DESTINATION = ROOT / "docs/manual_review"
BATCH_SIZE = 100
ALLOWED = {102, 103, 104, 105, 106}


def digest(path: Path) -> str:
    """Посчитать отпечаток файла для происхождения пакетов."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Сохранить проверяемый JSON с реальными переносами строк."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    """Прочитать JSONL по физическим строкам, сохраняя Unicode внутри текста."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Сохранить строки без изменения исходных описаний."""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare() -> None:
    """Убрать нормализованные дубликаты и разбить описания, не читая целевые метки."""
    sources = {
        "train": SOURCE / "public-train-annotated.jsonl",
        "test": SOURCE / "public-test-dataset.parquet",
    }
    groups: dict[str, dict] = {}
    counts = Counter()
    for split, path in sources.items():
        records = read_jsonl(path) if split == "train" else pq.read_table(path).to_pylist()
        for number, row in enumerate(records, 1):
            text = row["description"]
            normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())
            group_id = hashlib.sha256(normalized.encode()).hexdigest()
            group = groups.setdefault(group_id, {
                "index": len(groups) + 1, "group_id": group_id,
                "description": text, "sources": [],
            })
            group["sources"].append({"split": split, "row": number, "item_id": row.get("item_id")})
            counts[split] += 1
    rows = list(groups.values())
    if (DESTINATION / "inputs.jsonl").exists():
        if read_jsonl(DESTINATION / "inputs.jsonl") != rows:
            raise ValueError("Очередь уже существует и отличается; не перезаписываем разметку")
        return
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "batches").mkdir(exist_ok=True)
    (DESTINATION / "reviews").mkdir(exist_ok=True)
    write_jsonl(DESTINATION / "inputs.jsonl", rows)
    for start in range(0, len(rows), BATCH_SIZE):
        write_jsonl(DESTINATION / "batches" / f"{start // BATCH_SIZE + 1:03}.jsonl", rows[start:start + BATCH_SIZE])
    write_json(DESTINATION / "manifest.json", {
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources.values()},
        "source_rows": dict(counts), "unique_groups": len(rows),
        "deduplication": "NFKC, casefold, ё→е, collapsed whitespace; global train/test groups; original descriptions preserved",
        "near_duplicates_removed": False, "batch_size": BATCH_SIZE,
        "inputs_sha256": digest(DESTINATION / "inputs.jsonl"),
        "source_labels_shown_to_reviewers": False,
        "method": "agent reads each full description; no regex/model proposals or inherited labels",
    })
    logging.info("Исходных строк %d, уникальных групп %d", sum(counts.values()), len(rows))


def validate() -> None:
    """Проверить покрытия, ID, цитаты и вопросы; не придумывать отсутствующие метки."""
    inputs = {row["group_id"]: row for row in read_jsonl(DESTINATION / "inputs.jsonl")}
    seen: dict[str, dict] = {}
    for path in sorted((DESTINATION / "reviews").glob("*.jsonl")):
        batch = int(path.stem)
        for row in read_jsonl(path):
            source = inputs[row["group_id"]]
            if row["group_id"] in seen or source["index"] != row["index"]:
                raise ValueError(f"Повтор или неверный индекс: {path}")
            if (source["index"] - 1) // BATCH_SIZE + 1 != batch:
                raise ValueError(f"Запись не из назначенного пакета: {path}")
            if row["status"] not in {"confident", "needs_user"}:
                raise ValueError(f"Неизвестный статус: {path}")
            labels = row["labels"]
            if row["status"] == "needs_user":
                if labels is not None or not row.get("question", "").strip():
                    raise ValueError(f"Сомнение должно иметь вопрос и labels=null: {path}")
            elif (not isinstance(labels, list) or labels != sorted(set(labels))
                  or any(type(cid) is not int or cid not in ALLOWED for cid in labels)):
                raise ValueError(f"Некорректные метки: {path}")
            if not row.get("reason", "").strip() or not isinstance(row.get("evidence"), list):
                raise ValueError(f"Нет основания или цитат: {path}")
            if row["status"] == "confident" and labels and not row["evidence"]:
                raise ValueError(f"Непустой набор без цитат: {path}")
            if any(not isinstance(quote, str) or not quote or quote not in source["description"] for quote in row["evidence"]):
                raise ValueError(f"Цитата отсутствует в исходном description: {path}, {row['index']}")
            seen[row["group_id"]] = row
    pending = [row for row in seen.values() if row["status"] == "needs_user"]
    revision = json.loads((DESTINATION / "policy.json").read_text(encoding="utf-8"))["revision"]
    receipt_path = DESTINATION / "review_receipts.json"
    receipts = json.loads(receipt_path.read_text(encoding="utf-8")).get("agent_reports", {}) if receipt_path.exists() else {}
    outdated = sum(
        receipts.get(f"{(row['index'] - 1) // BATCH_SIZE + 1:03}", {}).get("policy_used", 0) < revision
        for row in seen.values()
    )
    write_json(DESTINATION / "progress.json", {
        "total": len(inputs), "read_by_agents": len(seen),
        "confident": len(seen) - len(pending), "pending_user": len(pending),
        "remaining_unread": len(inputs) - len(seen),
        "current_policy_revision": revision,
        "decisions_require_policy_recheck": outdated,
        "ready_for_final_dataset": len(seen) == len(inputs) and not pending and not outdated,
        "complete": len(seen) == len(inputs) and not pending and not outdated,
        "first_500_reviewed": sum(row["index"] <= 500 for row in seen.values()),
        "first_500_remaining": min(500, len(inputs)) - sum(row["index"] <= 500 for row in seen.values()),
        "batch_counts": dict(Counter((row["index"] - 1) // BATCH_SIZE + 1 for row in seen.values())),
    })
    write_jsonl(DESTINATION / "questions.jsonl", [
        {**row, "description": inputs[row["group_id"]]["description"],
         "sources": inputs[row["group_id"]]["sources"]} for row in pending
    ])
    logging.info("Прочитано %d/%d; уверенно %d; вопросов %d", len(seen), len(inputs), len(seen) - len(pending), len(pending))


def main() -> None:
    """Подготовить пакеты или проверить уже внесённые человеком/агентом решения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else validate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
