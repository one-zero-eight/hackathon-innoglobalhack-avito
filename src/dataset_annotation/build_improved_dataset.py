"""Собрать производный датасет: явные исправления, дедупликация и карантин."""

from pathlib import Path

import copy
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq

from main import MICROCATEGORIES
from dataset_annotation.audit_dataset import (
    DATA, ROOT, WORK, normalize, semantic_candidates,
)
from dataset_annotation.render_train_audit import search_text

DESTINATION = ROOT / "docs/improved_dataset"
DECISIONS = DESTINATION / "decisions.json"
AUDIT = WORK / "dataset-audit.json"
OUTPUT_NAMES = {"train": "train.jsonl", "test": "reviewed_test.parquet"}
SOURCE_NAMES = {
    "train": "public-train-annotated.jsonl",
    "test": "public-test-dataset.parquet",
}
FILTER_NAMES = {
    "lexical_signal_without_label", "plaster_without_cues", "near_text_different_labels",
}


def digest(value: bytes) -> str:
    """Вычислить SHA-256 для проверки происхождения и группировки."""
    return hashlib.sha256(value).hexdigest()


@dataclass
class Record:
    """Хранить неизменённую исходную строку отдельно от рабочих меток и решений."""

    split: str
    row: int
    original: dict
    labels: list[int]
    group_id: str
    reasons: set[str] = field(default_factory=set)
    decision_ids: list[str] = field(default_factory=list)
    resolved: bool = False
    status: str = ""
    output_row: int | None = None

    def output(self) -> dict:
        """Сохранить исходную схему и текст, изменяя только утверждённые метки."""
        result = copy.deepcopy(self.original)
        if self.split == "train":
            old = sorted(entry["mcId"] for entry in result["found_microcategories"])
            if self.labels != old:
                result["found_microcategories"] = [
                    {"mcId": cid, "label": MICROCATEGORIES[cid]} for cid in self.labels
                ]
        else:
            old = sorted(cid for cid, name in MICROCATEGORIES.items()
                         if name in result["categories"])
            if self.labels != old:
                result["categories"] = [MICROCATEGORIES[cid] for cid in self.labels]
        return result

    def identity(self) -> dict:
        """Вернуть метаданные, однозначно указывающие на исходную строку."""
        return {
            "split": self.split, "source_file": SOURCE_NAMES[self.split],
            "source_row": self.row, "item_id": self.original.get("item_id"),
            "description_group_id": self.group_id,
        }


def load_records(decisions: dict, audit: dict) -> tuple[list[Record], pa.Schema]:
    """Проверить версии источников и сохранить все исходные поля."""
    for name, expected in decisions["source_sha256"].items():
        if digest((DATA / name).read_bytes()) != expected:
            raise ValueError(f"Неожиданная версия исходника: {name}")
    if decisions["source_sha256"] != audit["sha256"]:
        raise ValueError("Аудит и решения относятся к разным версиям данных")
    train = [json.loads(line) for line in
             (DATA / SOURCE_NAMES["train"]).read_text(encoding="utf-8").splitlines()]
    test_table = pq.read_table(DATA / SOURCE_NAMES["test"])
    reverse = {name: cid for cid, name in MICROCATEGORIES.items()}
    records = []
    for split, source in (("train", train), ("test", test_table.to_pylist())):
        for number, original in enumerate(source, 1):
            text = original["description"]
            if not isinstance(text, str) or not isinstance(original["title"], str):
                raise ValueError("Описание и заголовок должны быть строками")
            if split == "train":
                labels = [entry["mcId"] for entry in original["found_microcategories"]]
                if any(MICROCATEGORIES.get(entry["mcId"]) != entry["label"]
                       for entry in original["found_microcategories"]):
                    raise ValueError("Название категории не соответствует ID")
            else:
                labels = [reverse[name] for name in original["categories"]]
            if len(labels) != len(set(labels)) or any(cid not in MICROCATEGORIES or cid == 101
                                                     for cid in labels):
                raise ValueError("Некорректные дополнительные категории")
            records.append(Record(
                split=split, row=number, original=original, labels=sorted(labels),
                group_id=digest(normalize(text).encode("utf-8")),
            ))
    if len(records) != len(audit["records"]):
        raise ValueError("Аудит содержит другое число строк")
    for record, cached in zip(records, audit["records"], strict=True):
        if (record.split != cached["split"] or record.row != cached["row"]
                or record.labels != cached["labels"]
                or record.original["description"] != cached["description"]
                or record.original["title"] != cached["title"]):
            raise ValueError("Порядок или значения строк аудита изменились")
    return records, test_table.schema


def apply_decisions(records: list[Record], decisions: dict) -> list[dict]:
    """Применить ручное решение ко всем нормализованным копиям в той же части."""
    groups = defaultdict(list)
    lookup = {}
    for record in records:
        groups[record.split, record.group_id].append(record)
        lookup[record.split, record.row] = record
    changes = []
    assigned = {}
    for split in ("train", "test"):
        for decision in decisions[split]:
            anchor = lookup[split, decision["row"]]
            if anchor.labels != decision["expected_labels"]:
                raise ValueError(f"Изменились исходные метки: {decision['id']}")
            if split == "train" and anchor.original["item_id"] != decision["item_id"]:
                raise ValueError(f"Изменился item_id: {decision['id']}")
            action = decision["action"]
            if action not in ("correct", "retain", "quarantine"):
                raise ValueError(f"Неизвестное действие: {action}")
            key = split, anchor.group_id
            target = sorted(decision.get("new_labels", anchor.labels))
            if len(target) != len(set(target)) or any(cid not in range(102, 107) for cid in target):
                raise ValueError("Некорректный целевой набор ручного решения")
            outcome = action, tuple(target)
            if key in assigned and assigned[key] != outcome:
                raise ValueError("Два несовместимых ручных решения на одинаковый текст")
            assigned[key] = outcome
            for member in groups[key]:
                member.decision_ids.append(decision["id"])
                if action == "quarantine":
                    member.reasons.add(f"manual_unresolved:{decision['id']}")
                    continue
                member.resolved = True
                if action == "correct" and member.labels != target:
                    changes.append({
                        **member.identity(), "decision_id": decision["id"],
                        "old_labels": member.labels[:], "new_labels": target,
                        "reason": decision["reason"],
                        "review_type": decision.get("review_type", "assistant_semantic_review"),
                    })
                    member.labels = target[:]
    return changes


def flag_unresolved(records: list[Record], decisions: dict, audit: dict) -> None:
    """Поместить неподтверждённые сигналы в карантин, не исправляя их по словам."""
    policy = decisions["selection_policy"]
    active = set(policy["unresolved_filters"])
    if active - FILTER_NAMES:
        raise ValueError("Неизвестный диагностический фильтр")
    whole_object = re.compile(policy["whole_object_only_pattern"], re.I)
    for record in records:
        if not record.labels and whole_object.search(record.original["description"]):
            record.resolved = True
            record.decision_ids.append("user-clarification:whole-object-only")
        if sum(char.isalpha() for char in record.original["description"]) < policy[
            "minimum_alphabetic_characters"
        ]:
            record.reasons.add("uninformative_description")
    rows = [
        {"description": r.original["description"], "labels": r.labels}
        for r in records
    ]
    by_group = defaultdict(list)
    for record in records:
        by_group[record.group_id].append(record)
    if "lexical_signal_without_label" in active:
        for candidate in semantic_candidates(rows)["missing_signal"]:
            record = records[candidate["index"]]
            if not record.resolved:
                record.reasons.add(f"unresolved_lexical_signal:mc{candidate['category']}")
    if "plaster_without_cues" in active:
        for record in records:
            if (not record.resolved and 102 in record.labels
                    and not re.search(policy["plaster_cues_pattern"],
                                      search_text(record.original["description"]), re.I)):
                record.reasons.add("unresolved_plaster_without_cues")
    if "near_text_different_labels" in active:
        for pair in audit["near_duplicates"]["pairs"]:
            if records[pair["left"]].labels == records[pair["right"]].labels:
                continue
            for side in ("left", "right"):
                key = records[pair[side]].group_id
                for record in by_group[key]:
                    if not record.resolved:
                        record.reasons.add("unresolved_near_text_different_labels")


def test_overlap_reasons(records: list[Record], audit: dict) -> dict[str, set[str]]:
    """Исключить из диагностики test точные и близкие совпадения с исходным train."""
    train_keys = {r.group_id for r in records if r.split == "train"}
    test_keys = {r.group_id for r in records if r.split == "test"}
    reasons: dict[str, set[str]] = defaultdict(set)
    for key in train_keys & test_keys:
        reasons[key].add("normalized_overlap_with_original_train")
    for pair in audit["near_duplicates"]["pairs"]:
        left, right = records[pair["left"]].group_id, records[pair["right"]].group_id
        if left in train_keys and right in test_keys:
            reasons[right].add("near_overlap_with_original_train")
        if right in train_keys and left in test_keys:
            reasons[left].add("near_overlap_with_original_train")
    return reasons


def select_records(records: list[Record], audit: dict) -> dict[str, list[Record]]:
    """Дедуплицировать оставшиеся группы и зафиксировать судьбу каждой исходной строки."""
    groups = defaultdict(list)
    for record in records:
        groups[record.split, record.group_id].append(record)
    overlap = test_overlap_reasons(records, audit)
    selected: dict[str, list[Record]] = {"train": [], "test": []}
    for (split, key), members in groups.items():
        reasons = set().union(*(r.reasons for r in members))
        if len({tuple(r.labels) for r in members}) > 1:
            reasons.add("unresolved_conflicting_labels")
        if split == "test" and key in overlap:
            for record in members:
                record.status = "excluded_overlap"
                record.reasons = reasons | overlap[key]
        elif reasons:
            for record in members:
                record.status = "quarantine"
                record.reasons = reasons.copy()
        else:
            selected[split].append(members[0])
            output_row = len(selected[split])
            for index, record in enumerate(members):
                record.status = "included" if index == 0 else "deduplicated"
                record.output_row = output_row
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Записать UTF-8 JSONL с детерминированным порядком полей и строк."""
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ), encoding="utf-8")


def verify(records: list[Record], selected: dict[str, list[Record]], decisions: dict) -> None:
    """Проверить контракт, сохранение текстов и полный учёт всех исходных строк."""
    allowed = {"included", "deduplicated", "quarantine", "excluded_overlap"}
    if any(r.status not in allowed for r in records):
        raise ValueError("Не у всех исходных строк определена судьба")
    actual = {
        "train": [json.loads(line) for line in
                  (DESTINATION / OUTPUT_NAMES["train"]).read_text().splitlines()],
        "test": pq.read_table(DESTINATION / OUTPUT_NAMES["test"]).to_pylist(),
    }
    for split in ("train", "test"):
        if actual[split] != [r.output() for r in selected[split]]:
            raise ValueError(f"Изменились поля записанного {split}")
        keys = [r.group_id for r in selected[split]]
        if len(keys) != len(set(keys)):
            raise ValueError("В выходе остались нормализованные дубликаты")
        for record in selected[split]:
            result = record.output()
            if set(result) != set(record.original):
                raise ValueError("Изменена исходная схема")
            if any(result[name] != record.original[name] for name in ("title", "description")):
                raise ValueError("Изменены исходные текстовые поля")
    train_keys = {r.group_id for r in selected["train"]}
    if train_keys & {r.group_id for r in selected["test"]}:
        raise ValueError("Пересечение train/test после нормализации")
    known_overlap = test_overlap_reasons(records, json.loads(AUDIT.read_text()))
    if any(r.group_id in known_overlap for r in selected["test"]):
        raise ValueError("В test осталось известное точное или близкое пересечение")
    lookup = {(r.split, r.group_id): r for part in selected.values() for r in part}
    for record in records:
        if record.status in ("included", "deduplicated"):
            canonical = lookup[record.split, record.group_id]
            if record.labels != canonical.labels or record.output_row != canonical.output_row:
                raise ValueError("Несогласованная дедупликация")
    provenance = [json.loads(line) for line in
                  (DESTINATION / "provenance.jsonl").read_text().splitlines()]
    if len(provenance) != len(records) or len({
        (r["split"], r["source_row"]) for r in provenance
    }) != len(records):
        raise ValueError("Неполный журнал происхождения")
    for name, expected in decisions["source_sha256"].items():
        if digest((DATA / name).read_bytes()) != expected:
            raise ValueError("Изменился исходник во время сборки")


def create_manifest(records: list[Record], selected: dict, changes: list[dict],
                    decisions: dict, audit: dict) -> dict:
    """Собрать измеренные счётчики и хеши без заявлений о качестве модели."""
    manifest = {
        "version": decisions["version"],
        "review_date": decisions["review_date"],
        "status": "derived_reviewed_not_organizer_ground_truth",
        "source_sha256": decisions["source_sha256"],
        "normalization": "NFKC, casefold, ё→е, whitespace",
        "near_similarity_threshold": audit["near_duplicates"]["threshold"],
        "test_status": "reviewed_diagnostic_subset_not_independent_holdout",
        "source_mc_id": {"train": 101, "test": None},
        "pyarrow_version": pa.__version__,
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "policy": decisions["selection_policy"],
        "correction_decisions": {
            split: sum(d["action"] == "correct" for d in decisions[split])
            for split in ("train", "test")
        },
        "user_review_decisions": {
            split: sum(d.get("review_type") in {"user_chat_review", "user_cli_review"}
                       for d in decisions[split])
            for split in ("train", "test")
        },
        "normalized_train_test_overlap": 0,
        "known_near_train_test_overlap": 0,
        "splits": {},
    }
    for split in ("train", "test"):
        part = [r for r in records if r.split == split]
        original_labels = [
            sorted(e["mcId"] for e in r.original["found_microcategories"])
            if split == "train" else sorted(
                cid for cid, name in MICROCATEGORIES.items() if name in r.original["categories"]
            ) for r in part
        ]
        manifest["splits"][split] = {
            "source_rows": len(part),
            "source_unique_normalized": len({r.group_id for r in part}),
            "output_rows": len(selected[split]),
            "source_row_statuses": dict(sorted(Counter(r.status for r in part).items())),
            "corrected_source_rows": sum(c["split"] == split for c in changes),
            "corrected_output_rows": sum(
                any(c["split"] == split and c["description_group_id"] == r.group_id
                    for c in changes) for r in selected[split]
            ),
            "output_empty_targets": sum(not r.labels for r in selected[split]),
            "source_label_counts": dict(sorted(Counter(
                cid for targets in original_labels for cid in targets
            ).items())),
            "output_label_counts": dict(sorted(Counter(
                cid for r in selected[split] for cid in r.labels
            ).items())),
        }
    manifest["output_sha256"] = {
        name: digest((DESTINATION / name).read_bytes()) for name in (
            *OUTPUT_NAMES.values(), "quarantine.jsonl", "excluded_overlap.jsonl",
            "corrections.jsonl", "provenance.jsonl", "decisions.json",
        )
    }
    if "full_annotation" in decisions:
        manifest["full_annotation"] = decisions["full_annotation"]
        for name in ("annotation_policy.json", "annotations.jsonl", "annotation_overrides.json",
                     "pending_review.jsonl"):
            manifest["output_sha256"][name] = digest((DESTINATION / name).read_bytes())
        if decisions["full_annotation"].get("verification_status") == "not_independently_verified":
            manifest["status"] = "provisional_annotation_not_independently_verified"
        if "repair" in decisions["full_annotation"]:
            manifest["output_sha256"]["repair_changes.jsonl"] = digest(
                (DESTINATION / "repair_changes.jsonl").read_bytes(),
            )
    return manifest


def write_readme(manifest: dict) -> None:
    """Описать применённые правила, реальные объёмы и ограничения производного набора."""
    train, test = manifest["splits"]["train"], manifest["splits"]["test"]
    text = f"""# improved_dataset

Производная размеченная версия InnoGlobalHack × Avito.
Исходники в `docs/innoglobalhack_avito/` не изменены.
Дата разбора: {manifest['review_date']}.

## Состав

| Файл | Назначение |
| --- | --- |
| `train.jsonl` | {train['output_rows']} уникальных нормализованных описаний; исходная схема train |
| `reviewed_test.parquet` | {test['output_rows']} описаний; исходная схема test, только диагностический набор |
| `quarantine.jsonl` | Неразрешённые случаи и неинформативные описания: полные исходные строки и причины |
| `excluded_overlap.jsonl` | Полные test-строки, исключённые из-за точного/близкого пересечения с исходным train |
| `decisions.json` | Явные решения, ожидаемые исходные метки, новые метки и правила отбора |
| `corrections.jsonl` | Каждое фактическое изменение меток: до/после, причина, строка и item_id |
| `provenance.jsonl` | Судьба каждой из 10208 исходных строк, группа, выходной файл и номер записи |
| `manifest.json` | Счётчики, распределения меток, параметры и SHA-256 источников/результатов |

В `train.jsonl` сохранены поля `item_id`, `title`, `description`,
`found_microcategories`. В parquet сохранены `title`, `description`, `categories`.
Тексты и заголовки не исправлялись и не сокращались. Исходный `mcId` train — 101;
у test он отсутствовал и **не был придуман** при очистке.
Столбец title не входит во входной контракт решения: не подмешивайте его в обучение.

## Измеренный результат

| Показатель | Train | Диагностический test |
| --- | ---: | ---: |
| Исходных строк | {train['source_rows']} | {test['source_rows']} |
| Исходных уникальных нормализованных описаний | {train['source_unique_normalized']} | {test['source_unique_normalized']} |
| Строк в очищенном выходе | {train['output_rows']} | {test['output_rows']} |
| Исходных строк с изменёнными метками, включая копии и исключённые строки | {train['corrected_source_rows']} | {test['corrected_source_rows']} |
| Выходных строк с изменёнными метками | {train['corrected_output_rows']} | {test['corrected_output_rows']} |

Полный учёт исходных строк по статусам:

- train: `{json.dumps(train['source_row_statuses'], ensure_ascii=False)}`;
- test: `{json.dumps(test['source_row_statuses'], ensure_ascii=False)}`.

`included` — каноническая строка в выходе; `deduplicated` — копия включённой
группы; `quarantine` — не включена до разбора; `excluded_overlap` —
test-строка исключена из-за пересечения. При одновременном пересечении и
сомнительной разметке приоритет имеет `excluded_overlap`, но причины сохранены.

## Политика очистки

1. **Явные решения и происхождение.** Исправлены {manifest['correction_decisions']['train']}
   train-групп и {manifest['correction_decisions']['test']} test-групп из `decisions.json`.
   Решение переносится на нормализованные копии только в той же части данных.
   Истинность этих решений не подтверждена организаторами.
   Отдельно зафиксированы пользовательские решения из последовательного разбора:
   train — {manifest['user_review_decisions']['train']}, test — {manifest['user_review_decisions']['test']}.
   Они имеют `review_type: user_chat_review` или `user_cli_review`
   и применяются ко всем копиям описания.
2. **«Только объект целиком» — корректный отрицательный пример.**
   Согласно уточнению пользователя, недоступные отдельно этапы не размечаются.
   Записи с пустым набором и явным ограничением сохранены; слово «под ключ»
   без ограничения само по себе не используется для обнуления меток.
3. **Неразрешённая неопределённость — в карантин.** Применены диагностические
   фильтры аудита к группам без явно сохранённого решения.
   Среди исключённых кандидатов могут быть корректные объявления:
   карантин **не означает ошибку разметки**.
4. **Шум:** описания менее чем с {manifest['policy']['minimum_alphabetic_characters']}
   буквами вынесены в карантин. Это правило информативности, не разметка услуг.
5. **Дубликаты:** NFKC → casefold → ё/е → схлопывание пробелов; первая строка
   исходного порядка становится канонической. Нормализованный текст
   применяется только к ключу группы, а не записывается вместо description.
   Все исходные item_id и номера строк восстанавливаются через provenance.
6. **Пересечения:** исключены test-тексты, совпадающие или достаточно близкие
   исходному train, включая его карантин. Порог символьного cosine:
   {manifest['near_similarity_threshold']}. Известных нормализованных и
   близких пересечений в выходных частях — 0.

## Что эта версия НЕ гарантирует

- Это не полностью вручную проверенный эталон: неотмеченные автоматическими
  фильтрами ошибки могут оставаться.
- Карантин меняет распределение данных. Нельзя трактовать число
  исключений как процент ошибочной разметки или сравнивать качество с исходником
  без учёта изменения выборки.
- Близкие шаблоны внутри train ещё существуют. При внутренних разбиениях
  удерживайте близкие варианты вместе, а не только точные дубликаты.
- `reviewed_test.parquet` уже использован для смыслового аудита и очистки.
  Это **не независимый финальный holdout**. Не подбирайте по нему пороги и промпты.
- Без исходного mcId для test нельзя восстановить все входы публичного контракта.
- Улучшение micro-F1 на этой версии не измерялось. Реализации пайплайнов,
  `config.json` и пути обучения по умолчанию не менялись.

## Воспроизведение

```sh
PYTHONPATH=src uv run python -m dataset_annotation.audit_dataset
PYTHONPATH=src uv run python -m dataset_annotation.build_improved_dataset
```

Сборка проверяет SHA-256 источников и ожидаемые исходные значения решений.
Проверяются полный учёт исходных строк, сохранение схем и текстов, согласованная
дедупликация и отсутствие известных пересечений. Для побайтового воспроизведения
parquet используйте зависимости из текущего `uv.lock`.

## Последовательный разбор в терминале

```sh
PYTHONPATH=src uv run python -m dataset_annotation.review_dataset
```

Вводите номер показанного варианта или собственный JSON-массив ID,
например `[102,106]` или `[]`. Доступны `s` — пропустить в этой сессии,
`q` — выйти, `?` — справка. Полные исходные описания не сокращаются.
После каждого ответа решения атомарно сохраняются, а набор пересобирается.
Повторный запуск продолжает с первого неразобранного описания.
Один ответ применяется ко всем нормализованным копиям в той же части.
Решение по одному SEO-списку не переносится автоматически на другие тексты.

Параметры: `--split train`, `--split test`, `--limit 10` или `--preview`
(показать первый случай без записи решений и пересборки).
Случаи в карантине рассматриваются первыми; исключённые из-за пересечений
test-строки попадают в очередь только при наличии других неразрешённых причин.
Даже после разбора такие test-строки не возвращаются в диагностику:
политика исключения пересечений сохраняется.
Если пересборка завершилась ошибкой, решение уже сохранено; следующий обычный
запуск повторяет сборку. Старые аудиторские репорты отражают исходный разбор,
а актуальные пользовательские решения находятся в `decisions.json`.

Полные аргументы:
[общий репорт](../../src/dataset_annotation/reports/full-report.md),
[train-репорт](../../src/dataset_annotation/reports/train-full-report.md).
"""
    if "full_annotation" in manifest:
        full = manifest["full_annotation"]
        text += f"""
## Полная разметка по пользовательской политике

Охвачено {full['groups']} уникальных групп, индивидуальных пользовательских
исключений — {full['individual_user_exceptions']}.
Метод: **автоматические правила и отдельные смысловые уточнения**,
а не ручное прочтение каждого объявления и не подтверждённый эталон.

- `annotation_policy.json` — принципы, ответы пользователя, их приоритет и область применения.
- `annotations.jsonl` — итог по каждой группе: метки, найденные основания,
  проигнорированный SEO, ограничения, источник решения.
- `annotation_overrides.json` — отдельные смысловые уточнения к автоматическим предложениям.
- `review_history/pre-full-annotation.json` — решения до полной переразметки.
  Старые решения ассистента также сохранены в `superseded_decisions`.

Предлагаемая отделка считается основанием для 102. Явный отказ от отдельных
работ обнуляет набор; одно «под ключ» его не обнуляет и не даёт все категории.
Поисковые списки исключены из анализа, но **не удалены из исходных текстов**.
При содержательной неоднозначности приоритет у полноты. Индивидуальные ответы
пользователя, включая прежние SEO-исключения, сохраняют приоритет.
Шум может остаться в структурном карантине даже после назначения меток.

Повторение полной процедуры:

```sh
PYTHONPATH=src uv run python -m dataset_annotation.annotate_dataset
PYTHONPATH=src uv run python -m dataset_annotation.finalize_dataset_annotations --apply
```

Обычная пересборка использует сохранённые решения, без повторной классификации.
Машинная разметка может содержать ошибки контекста, отрицаний и границ SEO.
Рост числа включённых записей не является измерением качества или F1.
"""
        if full.get("verification_status") == "not_independently_verified":
            text += f"""
## Ограничения смысловой проверки

Это **предварительная разметка без итоговой смысловой проверки**.
Разметка не является вручную подтверждённым эталоном. Результаты
сохранены с диагностическими признаками. Групп с отметками
сомнений — {full['unresolved_semantic_groups']}.
Они находятся в `pending_review.jsonl` с полным исходным объявлением,
текущими метками и причинами сомнений. Остальные машинные решения тоже
не следует считать гарантированно верными.

Техническая сборка проверяет только контракт, происхождение, сохранность
исходных текстов и правила дедупликации/исключения пересечений.
"""
        if "repair" in full:
            repair = full["repair"]
            text += f"""
## Самостоятельный повторный проход

По новому поручению пользователя повторно обработаны **все {repair['groups_recomputed']}
уникальные группы** исправленным офлайн-аннотатором, без субагентов.
По сравнению с предварительной разметкой изменены метки {repair['changed_groups']} групп.
Журнал до/после — `repair_changes.jsonl`.
Из активных правил исключены {repair['discarded_unverified_shard_overrides']}
непроверенных решений шардов; прежние варианты сохранены в `review_history/`.
Все 24 индивидуальных ответа пользователя сохранены.

Исправлены обработка SEO-маркеров, отрицаний до и после услуги,
ограничений только целым объектом, различие тёплого пола и утепления,
контекст покраски труб и сопоставление опечаток.
Проверены известные контрпримеры из предыдущего разбора.
Это сплошной **автоматический пересчёт с исправлением правил**, а не заявление
о ручном прочтении всех 4993 текстов или отсутствии любых смысловых ошибок.
Диагностические отметки в `pending_review.jsonl` сохранены.

```sh
PYTHONPATH=src uv run python -m dataset_annotation.repair_dataset
```
"""
    (DESTINATION / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Собрать и проверить все выходные файлы без изменения исходных данных."""
    decisions = json.loads(DECISIONS.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    records, schema = load_records(decisions, audit)
    changes = apply_decisions(records, decisions)
    flag_unresolved(records, decisions, audit)
    selected = select_records(records, audit)
    write_jsonl(DESTINATION / OUTPUT_NAMES["train"], [r.output() for r in selected["train"]])
    pq.write_table(pa.Table.from_pylist(
        [r.output() for r in selected["test"]], schema=schema,
    ), DESTINATION / OUTPUT_NAMES["test"], compression="zstd")
    for status, name in (("quarantine", "quarantine.jsonl"),
                         ("excluded_overlap", "excluded_overlap.jsonl")):
        write_jsonl(DESTINATION / name, [
            {**r.identity(), "reasons": sorted(r.reasons),
             "decision_ids": r.decision_ids, "working_labels": r.labels,
             "original": r.original}
            for r in records if r.status == status
        ])
    write_jsonl(DESTINATION / "corrections.jsonl", changes)
    write_jsonl(DESTINATION / "provenance.jsonl", [
        {**r.identity(), "status": r.status, "reasons": sorted(r.reasons),
         "decision_ids": r.decision_ids,
         "output_file": OUTPUT_NAMES[r.split] if r.output_row is not None else None,
         "output_row": r.output_row}
        for r in records
    ])
    verify(records, selected, decisions)
    manifest = create_manifest(records, selected, changes, decisions, audit)
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    write_readme(manifest)
    for split, stats in manifest["splits"].items():
        logging.info("%s: %s", split, stats)
    logging.info("Датасет сохранён: %s", DESTINATION)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
