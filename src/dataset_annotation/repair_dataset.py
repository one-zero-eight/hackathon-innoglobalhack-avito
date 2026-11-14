"""Повторно разметить весь корпус без ненадёжных автоматических шардов."""

import json
import logging
from pathlib import Path

from dataset_annotation import annotate_dataset as annotator
from dataset_annotation import build_improved_dataset as builder
from dataset_annotation.finalize_dataset_annotations import (
    ANNOTATIONS, HISTORY, OVERRIDES, POLICY, atomic_json, atomic_jsonl, prepare,
)
from dataset_annotation.review_dataset import session_lock


def main() -> None:
    """Пересчитать все группы, сохранив историю и индивидуальные ответы."""
    with session_lock():
        snapshot = builder.DECISIONS.read_bytes()
        decisions = json.loads(snapshot)
        audit = json.loads(builder.AUDIT.read_bytes())
        records, _ = builder.load_records(decisions, audit)
        old_overrides = json.loads(OVERRIDES.read_bytes())
        archive = HISTORY.parent / f"overrides-before-repair-{builder.digest(OVERRIDES.read_bytes())[:16]}.json"
        if not archive.exists():
            atomic_json(archive, old_overrides)
        archived_shard_count = max(
            (
                sum(
                    row.get("review_source", "").startswith("parallel-semantic-review-shard-")
                    for row in json.loads(path.read_bytes())["decisions"]
                )
                for path in HISTORY.parent.glob("overrides-before-repair-*.json")
            ),
            default=0,
        )
        overrides = {
            key: value for key, value in old_overrides.items()
            if key not in {"decisions", "parallel_review"}
        }
        overrides["decisions"] = [
            row for row in old_overrides["decisions"]
            if not row.get("review_source", "").startswith("parallel-semantic-review-shard-")
        ]
        overrides["repair_method"] = (
            "Полный повторный автоматический проход главным агентом; "
            "неподтверждённые результаты шардов исключены из активной разметки и сохранены в истории."
        )
        atomic_json(OVERRIDES, overrides)
        proposals = annotator.build_proposals(decisions, records, audit)
        updated, annotations = prepare(
            decisions, proposals, overrides, records, allow_unreviewed=True,
        )
        before = {
            (row["split"], row["description_group_id"]): row["labels"]
            for row in (
                json.loads(line) for line in ANNOTATIONS.read_text(encoding="utf-8").splitlines()
            )
        } if ANNOTATIONS.exists() else {}
        changes_path = builder.DESTINATION / "repair_changes.jsonl"
        if changes_path.exists():
            for line in changes_path.read_text(encoding="utf-8").splitlines():
                previous_change = json.loads(line)
                before[previous_change["split"], previous_change["description_group_id"]] = (
                    previous_change["old_labels"]
                )
        changed = [
            {"split": row["split"], "row": row["row"],
             "description_group_id": row["description_group_id"],
             "old_labels": before.get((row["split"], row["description_group_id"])),
             "new_labels": row["labels"]}
            for row in annotations
            if before.get((row["split"], row["description_group_id"])) != row["labels"]
        ]
        if builder.DECISIONS.read_bytes() != snapshot:
            raise ValueError("Решения изменились во время повторного прохода")
        previous = HISTORY.parent / f"before-repair-{builder.digest(snapshot)[:16]}.json"
        if not previous.exists():
            atomic_json(previous, decisions)
        atomic_jsonl(ANNOTATIONS, annotations)
        atomic_jsonl(annotator.DEFAULT_OUTPUT, proposals)
        originals = {(r.split, r.row): r.original for r in records}
        atomic_jsonl(builder.DESTINATION / "pending_review.jsonl", [
            {**row, "original": originals[row["split"], row["row"]]}
            for row in annotations if row["uncertainties"] and not row["user_locked"]
        ])
        atomic_jsonl(changes_path, changed)
        updated["full_annotation"]["annotations_sha256"] = builder.digest(ANNOTATIONS.read_bytes())
        updated["full_annotation"]["repair"] = {
            "groups_recomputed": len(annotations), "changed_groups": len(changed),
            "discarded_unverified_shard_overrides": archived_shard_count,
            "annotator_sha256": builder.digest(Path(annotator.__file__).read_bytes()),
            "policy_sha256": builder.digest(POLICY.read_bytes()),
            "manual_full_read_claimed": False,
        }
        atomic_json(builder.DECISIONS, updated)
        builder.main()
        logging.info("Повторно обработано %d групп; изменены метки %d групп.",
                     len(annotations), len(changed))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
