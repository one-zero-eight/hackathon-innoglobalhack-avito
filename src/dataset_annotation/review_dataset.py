"""Последовательно разобрать спорные группы: номер варианта или массив mcId."""

import argparse
import copy
import fcntl
import json
import logging
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from main import MICROCATEGORIES
from dataset_annotation import audit_dataset, build_improved_dataset
from dataset_annotation.audit_dataset import ROOT, WORK
from dataset_annotation.build_improved_dataset import (
    AUDIT, DECISIONS, DESTINATION, Record, load_records,
)

USER_REVIEW_TYPES = {"user_chat_review", "user_cli_review"}
SOURCE_MC_ID = 101
ALLOWED_IDS = set(MICROCATEGORIES) - {SOURCE_MC_ID}
LOCK = WORK / "improved-dataset-review.lock"
HELP = (
    "Введите номер варианта или JSON-массив, например [102,106], []. "
    "s — пропустить до следующего запуска, q — сохранить и выйти, ? — справка. "
    "101 недопустима: исходная категория train уже 101, а в разметке test "
    "используются только дополнительные 102–106."
)


@dataclass
class Case:
    """Объединить одно описание и его исходные копии в одной части."""

    split: str
    group_id: str
    members: list[Record]
    current_labels: list[int]
    reasons: list[str]
    excluded_overlap: bool


def parse_answer(value: str, options: list[list[int]]) -> list[int] | str:
    """Разобрать номер варианта, команду или строго проверенный массив ID."""
    text = value.strip()
    command = text.casefold()
    if command in {"q", "quit", "выход"}:
        return "quit"
    if command in {"s", "skip", "пропустить"}:
        return "skip"
    if command in {"?", "help", "помощь"}:
        return "help"
    if text.isdecimal():
        number = int(text)
        if not 1 <= number <= len(options):
            raise ValueError(f"Номер варианта должен быть от 1 до {len(options)}")
        return options[number - 1][:]
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Ожидается номер варианта или массив вида [102,106]") from exc
    if not isinstance(result, list):
        raise ValueError("Свой набор нужно ввести массивом: [102,106] или []")
    if any(type(cid) is not int or cid not in ALLOWED_IDS for cid in result):
        raise ValueError("Допустимы только целые ID 102–106, без 101, строк и bool")
    if len(result) != len(set(result)):
        raise ValueError("В массиве не должно быть повторяющихся ID")
    return sorted(result)


def suggest_options(case: Case) -> list[list[int]]:
    """Предложить прозрачные варианты, не объявляя лексический сигнал эталоном."""
    candidates = [case.current_labels[:]]
    candidates.extend(member.labels[:] for member in case.members)
    missing = {
        int(reason.rsplit("mc", 1)[1])
        for reason in case.reasons if reason.startswith("unresolved_lexical_signal:mc")
    }
    if missing:
        candidates.append(sorted(set(case.current_labels) | missing))
    if "unresolved_plaster_without_cues" in case.reasons:
        candidates.append([cid for cid in case.current_labels if cid != 102])
    candidates.append([])
    options = []
    for candidate in candidates:
        normalized = sorted(candidate)
        if normalized not in options:
            options.append(normalized)
    return options


def explain_reason(reason: str) -> str:
    """Объяснить диагностический сигнал, не выдавая его за доказанную ошибку."""
    if reason.startswith("unresolved_lexical_signal:mc"):
        cid = int(reason.rsplit("mc", 1)[1])
        return (f"В тексте найдено упоминание направления «{MICROCATEGORIES[cid]}» "
                f"({cid}), но метки нет. Это может быть SEO, сравнение или не предложение услуги.")
    if reason.startswith("manual_unresolved:"):
        return f"Ранее оставлено для ручного разбора: {reason.split(':', 1)[1]}."
    return {
        "unresolved_plaster_without_cues":
            "Есть 102, но не найдены упоминания штукатурки/выравнивания. Проверьте контекст и опечатки.",
        "unresolved_near_text_different_labels":
            "У похожих описаний разные метки. Это не обязательно ошибка: состав работ может отличаться.",
        "unresolved_conflicting_labels":
            "Одинаковые нормализованные описания имеют разные наборы меток.",
        "uninformative_description":
            "В описании почти нет букв; возможно, это обрывок или шум.",
    }.get(reason, reason)


def reviewed_groups(decisions: dict, records: list[Record]) -> set[tuple[str, str]]:
    """Найти уже отвеченные пользователем группы независимо от свежести карантина."""
    lookup = {(record.split, record.row): record for record in records}
    return {
        (split, lookup[split, decision["row"]].group_id)
        for split in ("train", "test") for decision in decisions[split]
        if decision.get("review_type") in USER_REVIEW_TYPES
        and decision["action"] in {"retain", "correct"}
    }


def collect_cases(decisions: dict, records: list[Record], split: str) -> list[Case]:
    """Собрать очередь по исходному порядку, показывая каждый текст только один раз."""
    members = defaultdict(list)
    for record in records:
        members[record.split, record.group_id].append(record)
    reviewed = reviewed_groups(decisions, records)
    found = {}
    for name in ("quarantine.jsonl", "excluded_overlap.jsonl"):
        path = DESTINATION / name
        if not path.exists():
            raise ValueError("Нет файлов очереди. Запустите скрипт без --preview для сборки.")
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            key = raw["split"], raw["description_group_id"]
            if (split != "all" and raw["split"] != split) or key in reviewed:
                continue
            substantive = [
                reason for reason in raw["reasons"]
                if reason.startswith(("unresolved_", "manual_unresolved:"))
                or reason == "uninformative_description"
            ]
            if name == "excluded_overlap.jsonl" and not substantive:
                continue
            if key not in members:
                raise ValueError("В очереди неизвестная группа исходных записей")
            anchor = next((r for r in members[key] if r.row == raw["source_row"]), None)
            if anchor is None or anchor.original != raw["original"]:
                raise ValueError("Очередь не соответствует исходному набору")
            if key not in found:
                found[key] = Case(
                    split=raw["split"], group_id=raw["description_group_id"],
                    members=members[key], current_labels=raw["working_labels"],
                    reasons=raw["reasons"], excluded_overlap=name == "excluded_overlap.jsonl",
                )
    return sorted(found.values(), key=lambda case:
                  (case.split != "train", case.members[0].row))


def display(case: Case, options: list[list[int]], position: int, total: int) -> None:
    """Вывести полное описание, метки, причины карантина и варианты ответа."""
    anchor = case.members[0]
    logging.info("\n%s\nСлучай %d из %d · %s:%d · копий: %d",
                 "=" * 72, position, total, case.split, anchor.row, len(case.members))
    logging.info("item_id: %s\nЗаголовок: %s",
                 anchor.original.get("item_id", "нет в исходном test"),
                 anchor.original["title"])
    logging.info("Исходные наборы группы: %s",
                 [list(values) for values in sorted({tuple(r.labels) for r in case.members})])
    logging.info("Текущий набор: %s", case.current_labels)
    logging.info("Что проверяем:")
    for reason in case.reasons:
        logging.info("• %s", explain_reason(reason))
    if case.excluded_overlap:
        logging.info("ВНИМАНИЕ: test останется исключён из диагностики из-за пересечения с train.")
    logging.info("\n--- Полный description ---\n%s\n--- Конец description ---",
                 anchor.original["description"])
    logging.info("\nКатегории: %s", "; ".join(
        f"{cid} — {MICROCATEGORIES[cid]}" for cid in sorted(ALLOWED_IDS)
    ))
    logging.info("Варианты — подсказки, не эталон. Можно ввести свой массив.")
    for index, option in enumerate(options, 1):
        logging.info("%d. %s%s", index, option, " — оставить текущий набор" if index == 1 else "")
    logging.info(HELP)


def save_answer(snapshot: bytes, case: Case, target: list[int], answer: str,
                records: list[Record], path: Path = DECISIONS) -> str:
    """Атомарно записать решение, сохраняя заменённые решения и проверяя конфликты."""
    if path.read_bytes() != snapshot:
        raise ValueError("decisions.json изменился во время ответа. Перезапустите разбор; файл не перезаписан.")
    data = json.loads(snapshot)
    lookup = {(record.split, record.row): record for record in records}
    sequence = max(
        (d.get("review_case", 0) for split in ("train", "test") for d in data[split]),
        default=0,
    ) + 1
    anchor = case.members[0]
    identifier = f"user-review-{sequence:03d}-{case.split}-{anchor.row}"
    replacement = {
        "id": identifier, "row": anchor.row, "expected_labels": anchor.labels[:],
        "new_labels": target,
        "action": "correct" if any(r.labels != target for r in case.members) else "retain",
        "review_type": "user_cli_review", "review_case": sequence,
        "review_answer": answer, "description_group_id": case.group_id,
        "reason": "Пользователь выбрал набор в интерактивном CLI. "
                  "Применяется только к нормализованным копиям этого текста в той же части.",
    }
    if case.split == "train":
        replacement["item_id"] = anchor.original["item_id"]
    remaining = []
    for decision in data[case.split]:
        previous = lookup[case.split, decision["row"]]
        if previous.group_id == case.group_id:
            archived = copy.deepcopy(decision)
            archived["superseded_by"] = identifier
            data.setdefault("superseded_decisions", []).append(archived)
        else:
            remaining.append(decision)
    data[case.split] = [replacement, *remaining]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".decisions-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.read_bytes() != snapshot:
            raise ValueError("Решения изменились перед сохранением; перезапись отменена.")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return identifier


@contextmanager
def session_lock() -> Iterator[None]:
    """Не допустить две одновременно работающие интерактивные сессии."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Уже запущена другая сессия review_dataset.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def rebuild() -> None:
    """Пересобрать набор с инвариантами, не засоряя интерактивный вывод статистикой."""
    logger = logging.getLogger()
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        if not AUDIT.exists():
            audit_dataset.main()
        build_improved_dataset.main()
    finally:
        logger.setLevel(previous_level)


def run(args: argparse.Namespace) -> None:
    """Запустить очередь, немедленно сохраняя каждый подтверждённый ответ."""
    if not args.preview:
        logging.info("Проверяю источники и пересобираю очередь…")
        rebuild()
    if not AUDIT.exists():
        raise ValueError("Нет аудита. Сначала запустите без --preview.")
    snapshot = DECISIONS.read_bytes()
    decisions = json.loads(snapshot)
    records, _ = load_records(decisions, json.loads(AUDIT.read_text(encoding="utf-8")))
    cases = collect_cases(decisions, records, args.split)
    if not cases:
        logging.info("Неразобранных случаев в выбранной части нет.")
        return
    answered = 0
    for position, case in enumerate(cases, 1):
        options = suggest_options(case)
        display(case, options, position, len(cases))
        if args.preview:
            return
        while True:
            try:
                value = input("\nВаш ответ > ")
                answer = parse_answer(value, options)
            except ValueError as exc:
                logging.warning("%s", exc)
                continue
            if answer == "help":
                logging.info(HELP)
                continue
            break
        if answer == "quit":
            logging.info("Завершено. Все предыдущие ответы сохранены.")
            return
        if answer == "skip":
            logging.info("Пропущено в этой сессии; появится снова при следующем запуске.")
            continue
        identifier = save_answer(snapshot, case, answer, value.strip(), records, path=DECISIONS)
        snapshot = DECISIONS.read_bytes()
        logging.info("Сохранено %s: %s. Пересобираю набор…", identifier, answer)
        try:
            rebuild()
        except Exception:
            logging.error("Решение сохранено, но пересборка не завершилась. "
                          "Следующий запуск повторит её; повторять ответ не нужно.")
            raise
        answered += 1
        logging.info("Готово. Решение применено к %d исходным копиям.", len(case.members))
        if args.limit is not None and answered >= args.limit:
            logging.info("Достигнут лимит %d ответов. Прогресс сохранён.", args.limit)
            return
    logging.info("Очередь этой сессии закончилась. Прогресс сохранён.")


def main() -> None:
    """Разобрать параметры CLI и корректно завершить разбор по EOF или Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--limit", type=int, help="Остановиться после указанного числа ответов")
    parser.add_argument("--preview", action="store_true", help="Только показать первый случай")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть положительным")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.preview:
            run(args)
        else:
            with session_lock():
                run(args)
    except (EOFError, KeyboardInterrupt):
        logging.info("\nРазбор остановлен. Подтверждённые ответы сохранены. "
                     "Если пересборка прервалась, следующий запуск повторит её.")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        logging.error("Ошибка: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
