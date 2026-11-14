"""Служебная проверка CLI разбора на временной копии решений.

Запуск из корня:

    PYTHONPATH=src uv run python -m dataset_annotation.check_review_cli

Проверка не добавляет ответов в настоящий датасет: изменения пишутся только
во временную копию ``decisions.json`` внутри ``work/``.
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

from dataset_annotation.audit_dataset import WORK
from dataset_annotation.build_improved_dataset import AUDIT, DECISIONS, load_records
from dataset_annotation.review_dataset import (
    Case, collect_cases, parse_answer, run, save_answer, suggest_options,
)


def main() -> None:
    """Проверить ввод, архивирование, атомарную запись и защиту прогресса."""
    if not AUDIT.exists():
        raise ValueError(
            f"Нет кеша {AUDIT}. Сначала выполните "
            "PYTHONPATH=src uv run python -m dataset_annotation.audit_dataset."
        )
    if not DECISIONS.exists():
        raise ValueError(f"Нет {DECISIONS}. Сначала соберите датасет.")
    logging.info("Служебная проверка использует только временную копию решений; "
                 "настоящие ответы не меняются.")
    snapshot = DECISIONS.read_bytes()
    decisions = json.loads(snapshot)
    records, _ = load_records(decisions, json.loads(AUDIT.read_text()))
    options = [[102, 106], [106], []]
    valid = {
        "1": [102, 106], "2": [106], "3": [], "[106,102]": [102, 106],
        " [] ": [], " [102, 106] ": [102, 106],
        "q": "quit", "выход": "quit", "s": "skip", "?": "help",
    }
    for value, expected in valid.items():
        if parse_answer(value, options) != expected:
            raise ValueError(f"Неверный разбор: {value}")
    invalid = (
        "", "0", "4", "102", "-1", "1.0", "{}", "null", "[101]", "[107]",
        '["102"]', "[true]", "[102.0]", "[102,102]", "102,106", "[102,]",
    )
    for value in invalid:
        try:
            parse_answer(value, options)
        except ValueError:
            continue
        raise ValueError(f"Принят некорректный ввод: {value}")
    cases = collect_cases(decisions, records, "train")
    if len(cases) != len({(case.split, case.group_id) for case in cases}):
        raise ValueError("В очереди повторяются группы")
    # Временная копия решений и настоящие исходники: никаких новых ответов в датасет.
    anchor = next(record for record in records if record.split == "train" and record.row == 47)
    members = [record for record in records
               if record.split == "train" and record.group_id == anchor.group_id]
    case = Case("train", anchor.group_id, members, anchor.labels,
                ["unresolved_plaster_without_cues"], False)
    if suggest_options(case) != [[102, 106], [106], []]:
        raise ValueError("Неверные варианты для спорной штукатурки")
    previous_ids = {
        item["id"] for item in decisions["train"]
        if records[item["row"] - 1].group_id == anchor.group_id
    }
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="improved-review-", dir=WORK) as directory:
        path = Path(directory) / "decisions.json"
        path.write_bytes(snapshot)
        identifier = save_answer(snapshot, case, [106], "[106]", records, path=path)
        saved = json.loads(path.read_bytes())
        selected = next(item for item in saved["train"] if item["id"] == identifier)
        if selected["review_type"] != "user_cli_review" or selected["new_labels"] != [106]:
            raise ValueError("Потеря пользовательского ответа")
        same_group = [item for item in saved["train"] if item["row"] == 47]
        if len(same_group) != 1:
            raise ValueError("Не заменено предыдущее решение")
        archived_ids = {
            item["id"] for item in saved.get("superseded_decisions", [])
            if item.get("superseded_by") == identifier
        }
        if previous_ids != archived_ids:
            raise ValueError("Потеря истории заменённого решения")
        try:
            save_answer(snapshot, case, [], "[]", records, path=path)
        except ValueError:
            pass
        else:
            raise ValueError("Устаревший снимок перезаписал новые решения")
        if list(Path(directory).glob("*.tmp")) or list(Path(directory).glob(".*.tmp")):
            raise ValueError("Не удалены временные файлы")
        path.write_bytes(snapshot)
        with (
            patch("dataset_annotation.review_dataset.DECISIONS", path),
            patch("builtins.input", return_value="[106]"),
            patch("dataset_annotation.review_dataset.rebuild") as rebuilt,
            patch("dataset_annotation.review_dataset.display"),
            patch("dataset_annotation.review_dataset.collect_cases", return_value=[case]),
        ):
            run(argparse.Namespace(preview=False, split="train", limit=1))
            if rebuilt.call_count != 2:
                raise ValueError("После ответа не запущена пересборка")
        updated = json.loads(path.read_bytes())
        user_answer = next(item for item in updated["train"]
                           if item["row"] == 47 and item["review_type"] == "user_cli_review")
        if user_answer["new_labels"] != [106]:
            raise ValueError("Интерактивный цикл не сохранил массив")
        resumed = collect_cases(updated, records, "train")
        if any(item.group_id == case.group_id for item in resumed):
            raise ValueError("После перезапуска снова показан отвеченный случай")
    answers = iter(["invalid", "?", "q"])
    with (
        patch("builtins.input", side_effect=lambda _: next(answers)),
        patch("dataset_annotation.review_dataset.rebuild") as rebuilt,
        patch("dataset_annotation.review_dataset.display"),
        patch("dataset_annotation.review_dataset.collect_cases", return_value=[case]),
    ):
        run(argparse.Namespace(preview=False, split="train", limit=1))
        if rebuilt.call_count != 1:
            raise ValueError("При выходе без ответа лишняя пересборка")
    if DECISIONS.read_bytes() != snapshot:
        raise ValueError("Служебная проверка изменила настоящие пользовательские решения")
    logging.info("Пройдено: %d допустимых, %d недопустимых вводов; "
                 "варианты, замена/история, защита от устаревшего снимка, "
                 "циклы invalid/help/quit и ввод массива, продолжение с места "
                 "остановки и неизменность настоящих решений.",
                 len(valid), len(invalid))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
