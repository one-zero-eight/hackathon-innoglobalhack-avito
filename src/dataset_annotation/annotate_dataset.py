"""Офлайн-предложения разметки всего InnoGlobalHack по пользовательским правилам.

Скрипт не меняет исходники, decisions.json или собранный датасет. Он читает
только description, объединяет нормализованные дубликаты внутри split и пишет
одно предложение на группу. Это автоматическая эвристика, а не ручная
семантическая проверка: сомнительные места явно попадают в uncertainties.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from main import MICROCATEGORIES
from dataset_annotation.audit_dataset import ROOT, WORK, normalize
from dataset_annotation.build_improved_dataset import (
    AUDIT, DECISIONS, DESTINATION, load_records, test_overlap_reasons,
)
from dataset_annotation.render_train_audit import search_text
from dataset_annotation.review_dataset import USER_REVIEW_TYPES

POLICY = DESTINATION / "annotation_policy.json"
DEFAULT_OUTPUT = WORK / "annotation-proposals.jsonl"
ALLOWED_IDS = tuple(range(102, 107))
FULL_REPAIR_SET = [102, 103, 104, 105, 106]
FULL_FINISHING_SET = [102, 103, 106]
METHOD = "offline_rule_based_repair_v2_not_manual_review"
LOCKED_METHOD = "user_locked_review_decision_preserved"

# Границы слов и фраз подобраны по реальным описаниям. Это не обученная модель,
# поэтому ниже сохраняются следы правил и неоднозначности.
SEO_MARKERS = re.compile(
    r"\b(?:"
    r"для\s+поиска|слов\w*\s+для\s+поиска|маячки\s+для\s+поиска|"
    r"ключи\s+для\s+поиска|нас\s+можно\s+найти|"
    r"(?:нас|меня|команду)\s+(?:также\s+)?находят\s+по\s+запросам|"
    r"нас\s+ищут\s+по\s+запросам|также\s+ищут|"
    r"поисков\w*\s+(?:слов\w*|запрос\w*|систем\w*)|"
    r"ключев\w*\s+(?:слов\w*|запрос\w*)|"
    r"нас\s+находят|"
    r"нас\s+(?:также|тоже)\s+ищут|так\s+же\s+нас\s+ищут|"
    r"также\s+нас\s+ищут|ищут\s+нас|ищут\s+также|"
    r"ищите\s+(?:меня|нас)|ищите\s+нас|также\s+ищите|"
    r"возможно\s*,?\s*(?:вы\s+|нас\s+)?(?:ищ|иска)\w*|"
    r"вы\s+могли\s+искать|по\s+каким\s+запросам|"
    r"запросы\s+для\s+поиска|популярные\s+запросы|"
    r"часто\s+ищут|теги\s+для\s+поиска|теги|tags|keywords"
    r")\s*[:：]?",
    re.I,
)
ACTION = re.compile(
    r"(?:выполня|дела|оказы|занима|предлага|бер[её]м|берем|беру|"
    r"монтир|монтаж|установ|уклад|облицов|поклей|оклей|наклей|"
    r"покрас|окрас|штукатур|шпакл|шпатл|развод|замен|подключ|"
    r"проклад|сборк|нанесен|выравн|ремонт|отдел|работ|услуг)",
    re.I,
)
INSTALL_ACTION = re.compile(
    r"(?:монтир|монтаж|установ|уклад|облицов|поклей|оклей|наклей|"
    r"покрас|окрас|штукатур|шпакл|шпатл|развод|замен|подключ|"
    r"проклад|сборк|нанесен|выравн|демонтаж|снятие|удален|ремонт)",
    re.I,
)
SERVICE_HEADING = re.compile(
    r"(?:что\s+(?:мы\s+)?(?:дела|входит|умеем|предлага)|"
    r"наши\s+услуги|виды\s+работ|перечень\s+работ|наши\s+работ|"
    r"прайс|цены|стоимость\s+работ|выполняем|предлагаем|занимаемся|"
    r"что\s+входит|в\s+работу\s+входит)",
    re.I,
)
BULLET = re.compile(r"^\s*(?:[-–—•*+●▪✔✅☑️🔹🔸💥🔥]|\d+[.)])\s*", re.I)
REMOVAL = re.compile(
    r"(?:демонтаж|демонтир|снятие|снима|снять|удален|удаля|убира|убрать|"
    r"сбить|сбива|сбитие|отбить|отбива|зачист|очист|снос|разборк)",
    re.I,
)
TARGET_BEFORE = re.compile(r"(?:под|для|вместо|взамен)\s+(?:нов\w*\s+)?$", re.I)
REPLACE_ACTION = re.compile(
    r"(?:замен|уклад|облицов|монтаж|установ|покрас|окрас|наклей|поклей|"
    r"оклей|штукатур|шпакл|шпатл)",
    re.I,
)
TEMPORAL_BEFORE = re.compile(
    r"(?:после|до|во\s+время|в\s+процессе|на\s+этапе)\s+"
    r"(?:\w+\s+){0,3}$",
    re.I,
)
MATERIAL = re.compile(
    r"(?:продаж\w*|прода[её]м|продаю|реализац\w*|поставк\w*|"
    r"магазин|склад|шоурум|опт(?:ов\w*)?|"
    r"подбор\w*\s+материал\w*|закупк\w*\s+материал\w*|"
    r"материал\w*\s+в\s+наличии|"
    r"цена\s+за\s+(?:м2|м²|кв\.?\s*м))",
    re.I,
)
TOOL = re.compile(
    r"(?:инструмент|плиткорез|станок|электроинструмент|оборудован|оснастк)",
    re.I,
)
NEGATION = re.compile(
    r"(?:не\s+(?:дела(?:ю|ем|ет|ют)?|выполня\w*|занима\w*|оказы\w*|"
    r"бер[её]м|берем|берусь|предлага\w*|включ\w*|входит)|"
    r"кроме\s+|исключая\s+|за\s+исключением\s+)",
    re.I,
)
NEGATION_SUFFIX = re.compile(
    r"^(?:\s+[^.\n]{0,40})?(?:"
    r"не\s+(?:дела|выполня|входит|включ|оказы)|"
    r"не\s+наша|делают\s+другие|выполняют\s+другие|оказывают\s+другие"
    r")",
    re.I,
)
WARNING = re.compile(
    r"(?:^\s*осторожно[^.\n]{0,40}мошен|не\s+оплачено|"
    r"стоп\s*[-–—]\s*не\s+оплачено|аванс\w*[^.\n]{0,40}скрыл|"
    r"ищу\s+людей[^.\n]{0,80}(?:наказать|аванс)|осторожно[^.\n]{0,40}мошен)",
    re.I,
)
DEMAND = re.compile(
    r"^\s*(?:требуется|требуются|ищу|ищем|рассмотрю|приму|"
    r"кто\s+(?:может|делает|занимается))\b",
    re.I,
)
DEMAND_PROVIDER = re.compile(
    r"^\s*(?:ищу|ищем)\s+(?:заказчик|клиент|работу|объект|объ[её]м)",
    re.I,
)
GENERIC_ALL_REPAIR = re.compile(
    r"(?:все|любые|полн\w*\s+(?:спектр|комплекс))\s+"
    r"(?:виды\s+)?(?:ремонтн\w*|ремонтн\w*[-\s]+отделочн\w*|ремонт\w*)\s+"
    r"(?:работ|услуг)"
    r"|(?:все|любые)\s+виды\s+работ\s+(?:по\s+)?ремонт\w*",
    re.I,
)
GENERIC_ALL_FINISHING = re.compile(
    r"(?:все|любые|весь|полн\w*)\s+(?:виды\s+)?"
    r"(?:спектр\s+|комплекс\s+)?(?:отделочн\w*|отделк\w*)"
    r"(?:\s+(?:работ|услуг))?",
    re.I,
)
GENERIC_ALL_WORKS = re.compile(
    r"(?:все|любые)\s+виды\s+(?:ремонтн\w*\s+)?работ", re.I,
)
BATHROOM_TURNKEY = re.compile(
    r"(?:(?:ремонт|отделка|отделк)\w*\s+)?"
    r"(?:ванн\w*(?:\s+комнат\w*)?|сан\s*узел\w*|сан\s*узл\w*|туалет\w*)"
    r"\s+под\s+ключ",
    re.I,
)
BATHROOM_REPAIR = re.compile(
    r"(?:ремонт|отделк|обнов)\w*\s+"
    r"(?:ванн\w*(?:\s+комнат\w*)?|сан\s*узел\w*|сан\s*узл\w*|туалет\w*)",
    re.I,
)
BATHROOM_ONLY = re.compile(
    r"(?:только|исключительно)\s+(?:с\s+)?"
    r"(?:ванн\w*|санузел\w*|санузл\w*|туалет\w*)\s+(?:целиком|полностью)"
    r"|(?:работаем|бер[её]м|берем)\s+только\s+(?:с\s+)?"
    r"(?:ванн\w*|санузел\w*|санузл\w*|туалет\w*)\s+(?:целиком|полностью)"
    r"|(?:ванн\w*|санузел\w*|санузл\w*|туалет\w*)\s+(?:целиком|полностью)"
    r"[^.\n]{0,40}(?:отдельн\w*\s+(?:работ\w*|вид\w*)\s+не\s+"
    r"(?:бер[её]м|выполняем|делаем))",
    re.I,
)
WHOLE_OBJECT = [
    re.compile(
        r"\bтолько\s+(?:комплексн\w*\s+(?:объект|ремонт)|"
        r"полн\w*\s+(?:ремонт|проект))"
        r"|без\s+частичн\w*\s+(?:работ|задач)"
        r"|не\s+бер\w*[^.\n]{0,45}частичн"
        r"|частичн\w*\s+ремонт\w*\s+не\s+занима\w*"
        r"|работ\w*\s+с\s+объект\w*\s+целиком",
        re.I,
    ),
    re.compile(
        r"(?:не\s+(?:рассматрива\w*|принима\w*|бер\w*|выполня\w*|дела\w*))"
        r"[^.\n]{0,45}(?:отдельн\w*\s+(?:этап|комнат|вид|работ)|"
        r"частичн\w*\s+(?:ремонт|работ)|локальн\w*\s+задач)"
        r"|(?:частичн\w*\s+ремонт|отдельн\w*\s+(?:работ|этап|вид\w*\s+работ))"
        r"[^.\n]{0,20}не\s+(?:бер\w*|выполня\w*|дела\w*|рассматрива\w*)"
        r"|(?:только|исключительно)\s+(?:по\s+формату\s+)?"
        r"(?:ремонт\w*\s+)?(?:квартир\w*\s+)?под\s+ключ"
        r"|бер[её]м\s+(?:в\s+работу\s+)?(?:квартир\w*|дом\w*|объект\w*)"
        r"[^.\n]{0,25}целиком",
        re.I,
    ),
    re.compile(
        r"(?:работ\w*|бер[её]м|берем|выполня\w*|дела\w*)\s+"
        r"(?:только|исключительно)\s+"
        r"(?:под\s+ключ|комплексн\w*|объект\w*\s+(?:целиком|полностью)|"
        r"квартир\w*(?:\s+\w+){0,3}\s+(?:целиком|полностью)|"
        r"дом\w*(?:\s+\w+){0,3}\s+(?:целиком|полностью)|"
        r"полн\w*\s+(?:ремонт|отделк))",
        re.I,
    ),
    re.compile(
        r"(?:только|исключительно)[^.\n]{0,80}"
        r"(?:квартир\w*|дом\w*|объект\w*|помещени\w*)\s+"
        r"(?:целиком|полностью)",
        re.I,
    ),
    re.compile(
        r"(?:квартир\w*|дом\w*|объект\w*|помещени\w*)\s+"
        r"(?:целиком|полностью)[^.\n]{0,20}под\s+ключ",
        re.I,
    ),
    re.compile(
        r"(?:не\s+(?:бер[её]м|берем|выполняем|делаем|занимаемся)|"
        r"не\s+беремся)\s+(?:за\s+)?(?:отдельн\w*\s+)?"
        r"(?:работ\w*|заказ\w*|вид\w*\s+работ)",
        re.I,
    ),
    re.compile(
        r"(?:отдельн\w*\s+)?(?:работ\w*|заказ\w*|вид\w*\s+работ)\s+"
        r"не\s+(?:бер[её]м|берем|выполняем|делаем|оказываем)",
        re.I,
    ),
    re.compile(
        r"(?:работаем|бер[её]м|берем)\s+только\s+(?:с\s+)?"
        r"(?:объект\w*|квартир\w*|дом\w*)\s+(?:целиком|полностью)",
        re.I,
    ),
    re.compile(
        r"специализируемся\s+(?:только|исключительно)\s+на\s+комплексн\w*",
        re.I,
    ),
]
PARTIAL_OFFER = re.compile(
    r"(?:частичн\w*\s+(?:обновлен|ремонт|отделк|работ|переделк|восстановлен)\w*|"
    r"завершен\w*\s+начат\w*\s+ремонт\w*|"
    r"(?:выполня\w*|дела\w*|оказыва\w*|бер[её]м)\s+"
    r"(?:также\s+)?отдельн\w*\s+(?:работ|вид|этап)\w*|"
    r"(?:выполня\w*|дела\w*|оказыва\w*)\s+(?:также\s+)?"
    r"(?:конкретн\w*\s+)?(?:работ\w*|услуг\w*)\s+по\s+"
    r"(?:плитк|электр|сантех|маляр|штукатур|обо)\w*)",
    re.I,
)
NEGATION_SHORT = re.compile(r"(?:не|без|кроме)\s+", re.I)


def _is_positive_offer(text: str, match_start: int) -> bool:
    """Убедиться, что найденное упоминание частичных работ — положительное предложение, а не отказ.

    Ищем отрицание «не/без/кроме» в небольшом контексте непосредственно перед совпадением.
    """
    window = text[max(0, match_start - 40):match_start]
    return not bool(NEGATION_SHORT.search(window))


def _partial_offer_matches(text: str) -> tuple[bool, str | None]:
    """Найти положительное предложение частичных работ и вернуть цитату."""
    for match in PARTIAL_OFFER.finditer(text):
        suffix = text[match.end():match.end() + 55].split("\n")[0]
        rejected = re.search(
            r"\bне\s+(?:дела|бер|выполня|принима|рассматрива|занима)", suffix, re.I,
        )
        line = _line(text, match.start(), match.end())
        affirmative = bool(re.search(
            r"выполня|предлага|можно\s+заказ|бер[её]м|делаем|возмож|также|"
            r"частичн\w*\s+обновлен|завершен\w*\s+начат",
            line, re.I,
        )) or line.strip().strip("—-•✅✔ ") == match.group(0)
        if _is_positive_offer(text, match.start()) and not rejected and affirmative:
            return True, _excerpt(text, match.start(), match.end())
    return False, None
TOKEN_RULES: tuple[tuple[str, int, str], ...] = (
    ("отделочн", 102, "strong"), ("отделк", 102, "strong"),
    ("штукатур", 102, "strong"), ("оштукатур", 102, "strong"),
    ("выравн", 102, "strong"), ("выровн", 102, "strong"),
    ("плиточн", 103, "strong"), ("плитк", 103, "strong"),
    ("кафел", 103, "strong"), ("керамогранит", 103, "strong"),
    ("мозаик", 103, "strong"), ("облицовочн", 103, "weak"),
    ("сантех", 104, "strong"), ("труб", 104, "strong"),
    ("отоплен", 104, "strong"), ("канализац", 104, "strong"),
    ("водоснабж", 104, "strong"), ("водопровод", 104, "strong"),
    ("унитаз", 104, "strong"), ("смесител", 104, "strong"),
    ("раковин", 104, "strong"), ("инсталляц", 104, "strong"),
    ("полотенцесушител", 104, "strong"), ("радиатор", 104, "strong"),
    ("котельн", 104, "strong"), ("душев", 104, "weak"),
    ("электромонтаж", 105, "strong"), ("электропроводк", 105, "strong"),
    ("электрощит", 105, "strong"), ("электроустанов", 105, "strong"),
    ("электрик", 105, "strong"), ("проводк", 105, "strong"),
    ("розет", 105, "strong"), ("выключател", 105, "strong"),
    ("освещ", 105, "strong"), ("светильник", 105, "strong"),
    ("люстр", 105, "strong"), ("щиток", 105, "strong"),
    ("кабел", 105, "strong"), ("слаботоч", 105, "strong"),
    ("электрич", 105, "weak"),
    ("шпакл", 106, "strong"), ("шпатл", 106, "strong"),
    ("обои", 106, "strong"), ("обоев", 106, "strong"),
    ("обоям", 106, "strong"), ("обоями", 106, "strong"),
    ("обоях", 106, "strong"), ("обойн", 106, "strong"),
    ("покрас", 106, "strong"), ("окрас", 106, "strong"),
    ("окрашив", 106, "strong"), ("маляр", 106, "strong"),
    ("красим", 106, "strong"), ("покрасим", 106, "strong"),
    ("красить", 106, "strong"),
)
PLURAL_SHIELD = {"щит", "щита", "щиты", "щитов", "щитами", "щитах"}
AUTOMAT = {"автомат", "автомата", "автоматы", "автоматов", "автоматами", "автоматах"}
FUZZY_EXCLUDE = {
    "плиткорез", "трубогиб", "электроинструмент", "электроприбор",
    "автоматический", "автоматическая", "автоматическое",
}
TYPO_ALIASES = {
    "пллитка": 103, "керомогранит": 103, "мозайка": 103,
    "штуктурка": 102, "оделка": 102, "эллектрика": 105,
    "включатели": 105, "инсталяции": 104, "молярные": 106,
}


@dataclass(frozen=True)
class SeoSpan:
    """Хранить точный вырезанный поисковый фрагмент для аудита."""

    start: int
    end: int
    text: str
    marker: str
    kind: str


@dataclass(frozen=True)
class Occurrence:
    """Одно упоминание категории в тексте без SEO."""

    start: int
    end: int
    cid: int
    token: str
    strength: str
    channel: str


def _normalized(text: str) -> str:
    """Нормализовать mixed Cyrillic без изменения длины строки."""
    mapped = search_text(text).lower().replace("ё", "е").replace("ë", "е")
    if len(mapped) == len(text):
        return mapped
    # Крайне редкие многобуквенные lower-преобразования: сохраняем позиции.
    chars = []
    for char in search_text(text):
        low = char.lower()
        chars.append(low if len(low) == 1 else char)
    return "".join(chars).replace("ё", "е")


def _excerpt(text: str, start: int, end: int, radius: int = 130) -> str:
    """Вернуть читаемый контекст совпадения без переносов строк."""
    return " ".join(text[max(0, start - radius):min(len(text), end + radius)].split())


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Найти границы строки для контекстных правил."""
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    return left, len(text) if right < 0 else right


def _line(text: str, start: int, end: int) -> str:
    """Вернуть строку, в которой находится совпадение."""
    left, right = _line_bounds(text, start, end)
    return text[left:right]


def _alpha_count(text: str) -> int:
    """Посчитать буквы для правила информативности."""
    return sum(char.isalpha() for char in text)


def _fuzzy_match(token: str) -> tuple[int, str] | None:
    """Сопоставить опечатку с канонической основой только при близком слове."""
    cid = TYPO_ALIASES.get(token)
    return (cid, token) if cid is not None else None


def _term_match(token: str) -> tuple[int, str, str, str] | None:
    """Определить категорию токена: прямое совпадение или редкая опечатка.

    Четвёртым элементом возвращается канал распознавания, чтобы не выдавать
    толерантное к опечаткам совпадение за прямое.
    """
    if token in FUZZY_EXCLUDE:
        return None
    if token in PLURAL_SHIELD or token in AUTOMAT:
        return 105, token, "strong", "token"
    if token == "узо":
        return 105, token, "strong", "token"
    for stem, cid, strength in TOKEN_RULES:
        if token.startswith(stem):
            if stem == "плитк" and token.startswith("плиткорез"):
                continue
            return cid, stem, strength, "token"
    fuzzy = _fuzzy_match(token)
    if fuzzy is not None:
        return fuzzy[0], fuzzy[1], "strong", "typo_tolerant"
    return None


def _find_seo_spans(text: str) -> list[SeoSpan]:
    """Найти явные поисковые секции и длинные keyword-хвосты."""
    spans: list[SeoSpan] = []

    def add(start: int, end: int, marker: str, kind: str) -> None:
        start = max(0, start)
        end = min(len(text), end)
        if start < end:
            spans.append(SeoSpan(start, end, text[start:end], marker, kind))

    for match in SEO_MARKERS.finditer(_normalized(text)):
        tail = text[match.end():match.end() + 45]
        line_start = text.rfind("\n", 0, match.start()) + 1
        prefix = text[line_start:match.start()]
        at_line_start = len(prefix.strip(" \t•·-–—*✅✔️🔎🔍➡️")) <= 3
        if (not at_line_start and ":" not in match.group(0)
                and "：" not in match.group(0) and ":" not in tail and "：" not in tail):
            continue
        start = line_start if at_line_start else match.start()
        add(start, len(text), match.group(0), "explicit_marker")

    # Keyword-хвосты без маркера: длинная строка, мало глаголов, много повторов
    # «ремонт/отделка». Не трогаем короткие перечни реальных услуг.
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = offset
        offset += len(line)
        if not stripped or any(span.start <= start < span.end for span in spans):
            continue
        low = _normalized(stripped)
        if len(low) < 160 or re.search(r"[.!?]", low):
            continue
        cue_count = sum(low.count(stem) for stem in (
            "плитк", "кафел", "керамогранит", "мозаик", "сантех", "труб",
            "отоплен", "канализац", "электр", "проводк", "розет", "освещ",
            "штукатур", "шпакл", "шпатл", "обои", "покрас", "маляр",
            "отделк", "ремонт",
        ))
        repeat_repair = len(re.findall(r"ремонт", low))
        actions = len(ACTION.findall(low))
        if cue_count >= 6 and actions <= 1 and (
            repeat_repair >= 3 or len(low) >= 260
        ):
            add(start, start + len(line), stripped[:80], "keyword_tail")

    return sorted(spans, key=lambda span: (span.start, span.end))


def _masked_text(text: str, spans: list[SeoSpan]) -> str:
    """Заменить SEO пробелами, сохранив длины и позиции для evidence."""
    chars = list(text)
    for span in spans:
        for index in range(span.start, span.end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _phrases(masked: str) -> list[Occurrence]:
    """Добавить категории, которые задаются устойчивыми словосочетаниями."""
    found: list[Occurrence] = []
    rules = (
        (re.compile(r"\bзатирк\w*\s+(?:межплиточн\w*\s+)?шв\w*", re.I),
         103, "strong"),
        (re.compile(r"\b(?:т[её]пл(?:ый|ые|ого|ых|ым|ыми|ом)\s+пол\w*|"
                    r"пол\w*\s+с\s+подогревом)", re.I),
         104, "weak"),
        (re.compile(
            r"(?:декоративн\w*\s+покрыти\w*|покрыти\w*\s+кожа|"
            r"жидк\w*\s+обои)", re.I), 106, "weak"),
        (re.compile(r"укладк\w*\s+платк\w*", re.I), 103, "strong"),
        (re.compile(
            r"(?:подготовк\w*|подготов\w*)[^.\n]{0,35}"
            r"под\s+(?:покраск\w*|обои|обоев)", re.I), 106, "weak"),
    )
    for pattern, cid, strength in rules:
        for match in pattern.finditer(masked):
            if cid == 104:
                context = masked[max(0, match.start() - 25):match.end() + 25].lower()
                if not re.search(r"водян|водяно", context):
                    found.append(Occurrence(
                        match.start(), match.end(), 105, match.group(0), "weak",
                        "phrase_electric_warm_floor",
                    ))
                if "электрическ" in context:
                    continue
            found.append(Occurrence(
                match.start(), match.end(), cid, match.group(0), strength, "phrase",
            ))
    return found


def _occurrences(masked: str) -> list[Occurrence]:
    """Найти все токенные упоминания категорий в уже очищенном тексте."""
    mapped = _normalized(masked)
    found = list(_phrases(masked))
    for match in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+", mapped):
        token = match.group(0)
        if not token or token.isdigit():
            continue
        direct = _term_match(token)
        if direct is None:
            continue
        cid, _stem, strength, channel = direct
        found.append(Occurrence(
            match.start(), match.end(), cid, token, strength, channel,
        ))
    return sorted(found, key=lambda item: (item.start, item.cid))


def _is_demand(core: str) -> bool:
    """Отличить спрос/вакансию от предложения исполнителя."""
    stripped = core.lstrip()
    if DEMAND_PROVIDER.search(stripped):
        return False
    return bool(DEMAND.search(stripped))


def _is_material_only(core: str) -> bool:
    """Отличить магазин материалов от исполнителя работ."""
    if not re.search(
        r"(?:продажа|продаем|продаём|реализация|поставка|оптов\w*|"
        r"магазин|склад)\s+(?:строительн|отделочн|сантехник|плитк|материал|оборудован)",
        core, re.I,
    ):
        return False
    return not re.search(
        r"(?:выполня|дела|оказы|занима|монтаж|уклад|установ|покрас|штукатур|"
        r"шпакл|ремонт|отдел)", core, re.I,
    )


def _negated(text: str, start: int, end: int) -> bool:
    """Проверить отказ/исключение непосредственно перед упоминанием."""
    prefix = re.split(r"[.!?;\n]", text[max(0, start - 80):start].lower())[-1]
    suffix = text[end:end + 60].lower()
    for match in NEGATION.finditer(prefix):
        between = prefix[match.end():]
        if len(between) > 70 or re.search(r"[.!?;]", between):
            continue
        before = prefix[max(0, match.start() - 12):match.start()]
        if re.search(r"не\s+только\s*$", before):
            continue
        return True
    return bool(NEGATION_SUFFIX.search(suffix))


def _removal(text: str, start: int, end: int) -> bool:
    """Не считать демонтаж старого покрытия предложением услуги.

    «Замена старой плитки» — это укладка, поэтому локальное действие установки
    отменяет признак чистого удаления.
    """
    prefix = text[max(0, start - 60):start].lower()
    local = text[max(0, start - 25):min(len(text), end + 25)].lower()
    removal = bool(re.search(
        REMOVAL.pattern + r"[^.\n]{0,35}$", prefix, re.I,
    ))
    if not removal:
        return False
    return not REPLACE_ACTION.search(local)


def _target(text: str, start: int) -> bool:
    """Не считать «под плитку/обои/покраску/отделку» предложением работы."""
    prefix = text[max(0, start - 30):start].lower()
    return bool(TARGET_BEFORE.search(prefix))


def _temporal(text: str, start: int) -> bool:
    """Игнорировать ссылки «после/до/во время» выполнения работ."""
    prefix = text[max(0, start - 45):start].lower()
    if re.search(r"\bот\b[^.\n]{0,40}\bдо\b", text[max(0, start - 100):start], re.I):
        return False
    return bool(TEMPORAL_BEFORE.search(prefix))


def _material_context(text: str, start: int, end: int) -> bool:
    """Отличить продажу/подбор материала от монтажа."""
    window = text[max(0, start - 85):min(len(text), end + 85)]
    if not MATERIAL.search(window):
        return False
    return not INSTALL_ACTION.search(window)


def _tool_context(text: str, start: int, end: int) -> bool:
    """Игнорировать упоминание инструмента, а не работы."""
    window = text[max(0, start - 45):min(len(text), end + 45)]
    return bool(TOOL.search(window)) and not INSTALL_ACTION.search(window)


def _service_context(text: str, start: int, end: int, core: str) -> bool:
    """Проверить, что упоминание находится в перечне услуг, а не в прозе."""
    line = _line(text, start, end)
    window = text[max(0, start - 130):min(len(text), end + 130)]
    if ACTION.search(line) or SERVICE_HEADING.search(line):
        return True
    if ACTION.search(window) and len(line) > 20:
        return True
    stripped = line.strip()
    if len(stripped) <= 110 and not re.search(r"[?!]", stripped):
        if BULLET.match(line) or stripped.lower().startswith((
            "штукатур", "отделк", "плитк", "кафел", "керамогранит", "мозаик",
            "сантех", "труб", "отоплен", "канализац", "электр", "проводк",
            "розет", "освещ", "шпакл", "шпатл", "обои", "покрас", "маляр",
            "декоративн", "покрыт", "умн", "тепл", "подготов",
        )):
            return True
        if stripped.count(",") + stripped.count(";") >= 1 and len(stripped) <= 80:
            return True
    # Заголовок секции услуг мог быть строкой выше.
    previous = text[max(0, start - 450):start]
    if SERVICE_HEADING.search(previous):
        return True
    return False


def _generic_all_repair(masked: str) -> tuple[bool, str]:
    """Найти обещание полного спектра именно ремонтных работ."""
    for pattern in (GENERIC_ALL_REPAIR,):
        for match in pattern.finditer(masked):
            prefix = masked[max(0, match.start() - 35):match.start()]
            if re.search(r"(?:не|без|кроме)\s+(?:\w+\s+){0,3}$", prefix, re.I):
                continue
            return True, _excerpt(masked, match.start(), match.end())
    return False, ""


def _generic_all_finishing(masked: str) -> tuple[bool, str]:
    """Найти полный спектр отделочных работ: 102, 103, 106 всегда вместе."""
    match = GENERIC_ALL_FINISHING.search(masked)
    if not match:
        return False, ""
    prefix = masked[max(0, match.start() - 35):match.start()]
    if re.search(r"(?:не|без|кроме)\s+(?:\w+\s+){0,3}$", prefix, re.I):
        return False, ""
    return True, _excerpt(masked, match.start(), match.end())


def _generic_all_works(masked: str) -> tuple[bool, str, bool]:
    """Вернуть «все виды работ» и признак явного ремонтного контекста."""
    for match in GENERIC_ALL_WORKS.finditer(masked):
        window = masked[max(0, match.start() - 70):min(len(masked), match.end() + 70)]
        if re.search(r"(?:отделочн|плиточн|кафельн|сантехническ|"
                     r"электромонтажн|малярн|штукатурн)", window, re.I):
            continue
        prefix = masked[max(0, match.start() - 35):match.start()]
        if re.search(r"(?:не|без|кроме)\s+(?:\w+\s+){0,3}$", prefix, re.I):
            continue
        return True, _excerpt(masked, match.start(), match.end()), bool(
            re.search(r"ремонт", window, re.I)
        )
    return False, "", False


def _restriction(text: str) -> tuple[str | None, list[str]]:
    """Найти явный отказ от отдельных работ и ограничение целым объектом."""
    restrictions: list[str] = []
    for pattern in (BATHROOM_ONLY, *WHOLE_OBJECT):
        for match in pattern.finditer(text):
            context = _line(text, match.start(), match.end())
            if re.search(r"за\s+исключением\s+редк", context, re.I):
                continue
            prefix = text[max(0, match.start() - 20):match.start()]
            if re.search(r"не\s+только\s*$", prefix, re.I):
                continue
            restrictions.append(_excerpt(text, match.start(), match.end(), 90))
            break
    if not restrictions:
        return None, []
    unique = list(dict.fromkeys(restrictions))
    return " | ".join(unique), unique


def _bathroom_turnkey(masked: str) -> tuple[bool, str]:
    """Найти ремонт ванной/санузла под ключ или как отдельную услугу."""
    for pattern in (BATHROOM_TURNKEY, BATHROOM_REPAIR):
        match = pattern.search(masked)
        if match:
            window = masked[max(0, match.start() - 30):match.end() + 30]
            if re.search(r"(?:только|исключительно)[^.\n]{0,35}"
                         r"(?:целиком|полностью)", window, re.I):
                continue
            return True, _excerpt(masked, match.start(), match.end())
    return False, ""


def _add_evidence(evidence: dict[str, list[str]], cid: int, value: str) -> None:
    """Добавить не больше трёх разных подтверждений на категорию."""
    items = evidence.setdefault(str(cid), [])
    if value not in items and len(items) < 3:
        items.append(value)


def _weak_uncertainty(cid: int, occurrence: Occurrence) -> str:
    """Сформулировать конкретную причину сомнения для weak-сигнала."""
    if occurrence.channel.startswith("phrase"):
        return f"weak_phrase_mention:{cid}:{occurrence.token}"
    if occurrence.token.startswith("облицов"):
        return "oblicovochnye_works_without_material"
    if occurrence.token.startswith("электрич"):
        return "electric_context_may_be_tool_or_heating"
    return f"weak_signal_used:{cid}:{occurrence.token}"


def propose(description: str) -> dict:
    """Предложить метки и объяснения только по description, без title."""
    if not isinstance(description, str):
        raise ValueError("description должен быть строкой")
    seo_spans = _find_seo_spans(description)
    masked = _normalized(_masked_text(description, seo_spans))
    core = " ".join(masked.split())
    ignored_seo = [span.text.strip() for span in seo_spans if span.text.strip()]

    evidence: dict[str, list[str]] = {}
    uncertainties: list[str] = []
    notes: list[str] = []
    restrictions: list[str] = []
    refused: dict[int, str] = {}
    accepted: dict[int, list[Occurrence]] = defaultdict(list)
    ignored_notes: Counter[str] = Counter()

    restriction, restriction_parts = _restriction(masked)
    restrictions.extend(restriction_parts)

    # Конкретное частичное предложение отменяет общий «только под ключ».
    has_partial, partial_excerpt = _partial_offer_matches(masked)
    if restriction and has_partial:
        restriction = None
        restrictions.clear()
        notes.append("partial_offer_overrides_only_key_restriction")

    all_repair, all_repair_text = _generic_all_repair(masked)
    all_finishing, all_finishing_text = _generic_all_finishing(masked)
    all_works, all_works_text, all_works_repair = _generic_all_works(masked)
    bathroom, bathroom_text = _bathroom_turnkey(masked)
    warning = bool(WARNING.search(core))
    demand = _is_demand(core)
    material_only = _is_material_only(core)
    non_informative = _alpha_count(description) < 3

    for occurrence in _occurrences(masked):
        cid = occurrence.cid
        start, end = occurrence.start, occurrence.end
        excerpt = _excerpt(description, start, end)
        if cid in (102, 103, 106) and _target(masked, start):
            ignored_notes[f"target_{cid}"] += 1
            continue
        if _removal(masked, start, end):
            ignored_notes[f"removal_{cid}"] += 1
            continue
        if _negated(masked, start, end):
            refused[cid] = excerpt
            continue
        if _temporal(masked, start):
            ignored_notes[f"temporal_{cid}"] += 1
            continue
        if _tool_context(masked, start, end):
            ignored_notes[f"tool_{cid}"] += 1
            continue
        if _material_context(masked, start, end):
            ignored_notes[f"material_{cid}"] += 1
            continue
        prefix = masked[max(0, start - 55):start]
        if cid == 102 and re.search(r"\bбез\s+(?:\w+\s+){0,2}$", prefix):
            ignored_notes["existing_unfinished_object"] += 1
            continue
        if cid == 104 and re.search(
            r"(?:покраск|окраск|укрыв|защищ|укрыва)\w*[^.\n]{0,40}$", prefix,
        ):
            ignored_notes["painted_or_protected_plumbing"] += 1
            continue
        if cid == 103 and re.search(r"потолочн\w*\s+$", prefix):
            ignored_notes["ceiling_tile_not_ceramic_tiling"] += 1
            continue
        if cid == 102 and occurrence.token.startswith(("выравн", "выровн")):
            nearby = masked[max(0, start - 45):min(len(masked), end + 45)]
            if not re.search(r"(?:стен|потолк|поверхност|основан|откос)", nearby, re.I):
                ignored_notes["leveling_without_wall"] += 1
                continue
        if cid == 104 and occurrence.token.startswith("душев"):
            line = _line(masked, start, end)
            if not re.search(r"(?:установ|монтаж|замен|подключ|сантехник)", line, re.I):
                ignored_notes["shower_room_not_fixture"] += 1
                continue
        if not occurrence.channel.startswith("phrase") and not _service_context(
            masked, start, end, core,
        ):
            ignored_notes[f"incidental_{cid}"] += 1
            continue
        accepted[cid].append(occurrence)
        _add_evidence(evidence, cid, excerpt)
        if occurrence.strength == "weak":
            uncertainties.append(_weak_uncertainty(cid, occurrence))
        if occurrence.channel == "typo_tolerant":
            uncertainties.append(f"typo_tolerant_match:{cid}:{occurrence.token}")

    labels = sorted(accepted)
    generic_notes: list[str] = []

    if all_finishing:
        labels = sorted(set(labels) | set(FULL_FINISHING_SET))
        for cid in FULL_FINISHING_SET:
            _add_evidence(evidence, cid, all_finishing_text)
        generic_notes.append("all_finishing_works_gives_102_103_106")

    if all_repair:
        labels = sorted(set(labels) | set(FULL_REPAIR_SET))
        for cid in FULL_REPAIR_SET:
            _add_evidence(evidence, cid, all_repair_text)
        generic_notes.append("all_repair_works_gives_all_five")

    if all_works and all_works_repair and not all_repair:
        labels = sorted(set(labels) | set(FULL_REPAIR_SET))
        for cid in FULL_REPAIR_SET:
            _add_evidence(evidence, cid, all_works_text)
        generic_notes.append("all_works_with_repair_context_gives_all_five")

    # Санузел даёт минимум 103/104; явный список может добавить другие услуги.
    # Ограничение целым санузлом имеет приоритет над этой импликацией.
    if bathroom and not restriction:
        added = False
        if 103 not in labels:
            labels = sorted(set(labels) | {103})
            _add_evidence(evidence, 103, bathroom_text)
            added = True
        if 104 not in labels:
            labels = sorted(set(labels) | {104})
            _add_evidence(evidence, 104, bathroom_text)
            added = True
        if added:
            generic_notes.append("bathroom_turnkey_gives_103_104")
        if re.search(r"\bпанел\w*[^.\n]{0,20}пвх|\bпвх[^.\n]{0,20}панел", masked):
            labels = sorted(set(labels) | {102})
            _add_evidence(evidence, 102, bathroom_text)

    if refused:
        # Локальный отказ от мелкого ремонта не отменяет явно предложенную
        # крупную услугу в другой строке того же объявления.
        refused = {cid: value for cid, value in refused.items() if not accepted[cid]}
        labels = [cid for cid in labels if cid not in refused]
        for cid, excerpt in sorted(refused.items()):
            evidence[f"refusal_{cid}"] = [excerpt]
        refusal_note = "category_refused:" + ",".join(map(str, sorted(refused)))
        if labels:
            notes.append(refusal_note)
        else:
            uncertainties.append(refusal_note)

    if restriction:
        labels = []
        evidence["restriction"] = [restriction]
        notes.append("whole_object_only_restriction")

    if warning:
        labels = []
        evidence["not_a_service_offer"] = [_excerpt(description, 0, min(len(description), 180))]
        uncertainties.append("safety_or_scam_notice_not_service_offer")
    elif demand:
        labels = []
        evidence["not_a_service_offer"] = [_excerpt(description, 0, min(len(description), 180))]
        uncertainties.append("customer_demand_or_recruitment_not_service_offer")
    elif non_informative:
        labels = []
        uncertainties.append("non_informative_description")
    elif material_only:
        labels = []
        uncertainties.append("material_sale_not_service_offer")

    if all_works and not labels and not restriction and not warning and not demand:
        uncertainties.append("generic_all_works_without_specialization")
    if not labels and not uncertainties:
        if all_works:
            uncertainties.append("generic_all_works_without_specialization")
        elif re.search(r"под\s+ключ", core, re.I):
            uncertainties.append("generic_turnkey_without_specific_work_types")
        elif re.search(r"ремонт", core, re.I):
            uncertainties.append("generic_repair_without_specific_work_types")
        elif ignored_seo and not core:
            uncertainties.append("seo_only_text_without_work_evidence")
        else:
            uncertainties.append("no_supported_categories")

    material_categories = sorted(
        int(key.rsplit("_", 1)[1]) for key in ignored_notes
        if key.startswith("material_") and key.rsplit("_", 1)[1].isdigit()
    )
    if material_categories:
        notes.append("material_mention_ignored:" + ",".join(map(str, material_categories)))

    if labels and not evidence.get(str(labels[0])) and not restriction:
        uncertainties.append("label_without_direct_evidence")

    # Убрать повторы, сохранив порядок; автоматическая природа видна в review_method.
    uncertainties = list(dict.fromkeys(uncertainties))
    notes = list(dict.fromkeys(notes))
    restrictions = list(dict.fromkeys(restrictions))

    return {
        "labels": labels,
        "evidence": evidence,
        "ignored_seo": "\n---\n".join(ignored_seo),
        "ignored_seo_spans": ignored_seo,
        "restriction": restriction,
        "restrictions": restrictions,
        "refused_categories": sorted(refused),
        "uncertainties": uncertainties,
        "notes": notes,
        "generic_rules": generic_notes,
        "ignored_notes": dict(sorted(ignored_notes.items())),
    }


def _locked_from_decisions(
    decisions: dict,
    records: list,
) -> dict[tuple[str, str], dict]:
    """Собрать индивидуальные ответы пользователя по нормализованным группам."""
    lookup = {(record.split, record.row): record for record in records}
    locked: dict[tuple[str, str], dict] = {}
    for split in ("train", "test"):
        for decision in decisions[split]:
            if (decision.get("review_type") not in USER_REVIEW_TYPES
                    or decision.get("action") not in {"retain", "correct"}):
                continue
            record = lookup[split, decision["row"]]
            key = split, record.group_id
            target = decision.get("new_labels", decision["expected_labels"])
            if target is None:
                target = decision["expected_labels"]
            if (not isinstance(target, list)
                    or any(type(cid) is not int or cid not in ALLOWED_IDS for cid in target)
                    or target != sorted(set(target))):
                raise ValueError(f"Некорректная цель пользовательского решения: {decision['id']}")
            payload = {
                "id": decision["id"],
                "review_type": decision["review_type"],
                "action": decision["action"],
                "reason": decision.get("reason", ""),
                "new_labels": target[:],
            }
            if key in locked and locked[key]["new_labels"] != target:
                raise ValueError("Два несовместимых пользовательских решения на одну группу")
            locked[key] = payload
    return locked


def _record_for_group(
    split: str,
    group_id: str,
    members: list,
    locked: dict[tuple[str, str], dict],
    overlap: dict[str, set[str]],
    policy_id: str,
) -> dict:
    """Собрать одну строку JSONL для нормализованной группы."""
    members = sorted(members, key=lambda record: record.row)
    anchor = members[0]
    source_variants = sorted({tuple(record.labels) for record in members})
    proposal = propose(anchor.original["description"])
    key = split, group_id
    user_locked = key in locked
    labels = proposal["labels"]
    uncertainties = proposal["uncertainties"]
    review_method = METHOD
    evidence = dict(proposal["evidence"])
    restriction = proposal["restriction"]
    if user_locked:
        source = locked[key]
        labels = list(source["new_labels"])
        if labels != sorted(set(labels)):
            raise ValueError(f"Цель пользователя не отсортирована: {source['id']}")
        if proposal["labels"] != labels:
            uncertainties = list(dict.fromkeys(
                uncertainties + ["user_locked_overrides_rule_based_proposal"]
            ))
        evidence["user_locked"] = [
            f"{source['id']} ({source['review_type']}, {source['action']}): "
            f"{source['reason']}"
        ]
        for cid in labels:
            if str(cid) not in evidence:
                evidence[str(cid)] = [f"user_locked_target:{source['id']}"]
        review_method = LOCKED_METHOD
    exact_overlap = split == "test" and group_id in overlap
    exclusion_reasons = sorted(overlap.get(group_id, set())) if exact_overlap else []
    if len(source_variants) > 1:
        uncertainties = list(dict.fromkeys(
            uncertainties + ["source_label_variants_conflict"]
        ))
    confidence = "low" if not labels else (
        "medium" if uncertainties else "high"
    )
    return {
        "split": split,
        "row": anchor.row,
        "description_group_id": group_id,
        "source_labels": [list(values) for values in source_variants],
        "labels": labels,
        "evidence": evidence,
        "uncertainties": uncertainties,
        "notes": proposal["notes"],
        "user_locked": user_locked,
        "restriction": restriction,
        "ignored_seo": proposal["ignored_seo"],
        "review_method": review_method,
        "user_locked_source": locked.get(key),
        "source_rows": [record.row for record in members],
        "group_size": len(members),
        "source_label_conflict": len(source_variants) > 1,
        "excluded_overlap": exact_overlap,
        "exclusion_reasons": exclusion_reasons,
        "status": "excluded_overlap" if exact_overlap else "proposed",
        "restrictions": proposal["restrictions"],
        "refused_categories": proposal["refused_categories"],
        "generic_rules": proposal["generic_rules"],
        "ignored_notes": proposal["ignored_notes"],
        "ignored_seo_spans": proposal["ignored_seo_spans"],
        "proposal_confidence": confidence,
        "proposal_type": "offline_rule_based_first_pass_not_manual_review",
        "policy_id": policy_id,
        "taxonomy": {str(cid): MICROCATEGORIES[cid] for cid in ALLOWED_IDS},
    }


def _validate(rows: list[dict], records: list, locked: dict) -> None:
    """Проверить покрытие, контракт полей и неизменность пользовательских целей."""
    keys = [(row["split"], row["description_group_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Повторная группа в предложениях")
    groups = defaultdict(list)
    for record in records:
        groups[record.split, record.group_id].append(record)
    if set(keys) != set(groups):
        raise ValueError("Предложения покрывают не все нормализованные группы")
    for row in rows:
        key = row["split"], row["description_group_id"]
        members = sorted(groups[key], key=lambda record: record.row)
        if row["row"] != members[0].row:
            raise ValueError("Каноническая строка группы изменилась")
        expected = sorted({tuple(record.labels) for record in members})
        if sorted(tuple(values) for values in row["source_labels"]) != expected:
            raise ValueError("Исходные варианты меток не совпадают")
        labels = row["labels"]
        if (not isinstance(labels, list) or labels != sorted(set(labels))
                or any(type(cid) is not int or cid not in ALLOWED_IDS for cid in labels)):
            raise ValueError("Некорректные предложенные метки")
        if row["user_locked"] != (key in locked):
            raise ValueError("Неверный флаг user_locked")
        if key in locked and labels != locked[key]["new_labels"]:
            raise ValueError("Изменена цель пользовательского решения")
        if row["split"] == "test" and row["status"] == "excluded_overlap" and not row["excluded_overlap"]:
            raise ValueError("Несогласованный статус test-пересечения")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Записать детерминированный UTF-8 JSONL, не меняя другие файлы."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")


def _log_statistics(rows: list[dict], records: list, locked: dict, output: Path) -> None:
    """Вывести измеряемую статистику предложений, а не оценку качества."""
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    logging.info("Предложения: %s", output)
    logging.info("Всего строк: %d, групп: %d", len(records), len(rows))
    for split in ("train", "test"):
        part = by_split[split]
        row_count = sum(row["group_size"] for row in part)
        label_counts = Counter(cid for row in part for cid in row["labels"])
        source_counts = Counter(
            cid for row in part for variant in row["source_labels"] for cid in variant
        )
        logging.info(
            "%s: исходных строк %d, групп %d, locked %d, excluded_overlap %d, "
            "пустых групп %d, метки групп %s, метки строк %s",
            split, row_count, len(part),
            sum(row["user_locked"] for row in part),
            sum(row["excluded_overlap"] for row in part),
            sum(not row["labels"] for row in part),
            dict(sorted(label_counts.items())),
            dict(sorted(Counter(
                cid for row in part for cid in row["labels"]
                for _ in range(row["group_size"])
            ).items())),
        )
        logging.info("%s: исходные метки строк (для сравнения) %s",
                     split, dict(sorted(source_counts.items())))
    exact = 0
    changed = 0
    added = Counter()
    removed = Counter()
    for row in rows:
        variants = {tuple(values) for values in row["source_labels"]}
        target = tuple(row["labels"])
        if variants == {target}:
            exact += 1
        else:
            changed += 1
            baseline = set(variants.pop() if len(variants) == 1 else set().union(*map(set, variants)))
            added.update(set(target) - baseline)
            removed.update(baseline - set(target))
    logging.info("Совпало с исходным набором групп: %d; изменено: %d", exact, changed)
    logging.info("Добавлено категорий в группах: %s; убрано: %s",
                 dict(sorted(added.items())), dict(sorted(removed.items())))
    logging.info("SEO вырезано у групп: %d; ограничение целым объектом: %d",
                 sum(bool(row["ignored_seo"]) for row in rows),
                 sum(bool(row["restriction"]) for row in rows))
    logging.info("Групп с отказами от категорий: %d; с конфликтом исходных вариантов: %d",
                 sum(bool(row["refused_categories"]) for row in rows),
                 sum(bool(row["source_label_conflict"]) for row in rows))
    uncertain = [row for row in rows if row["uncertainties"]]
    logging.info("Групп с uncertainties: %d (locked: %d)",
                 len(uncertain), sum(row["user_locked"] for row in uncertain))
    reasons = Counter(
        uncertainty.split(":", 1)[0] for row in uncertain for uncertainty in row["uncertainties"]
    )
    logging.info("Причины uncertainties: %s", dict(reasons.most_common()))
    logging.info("Методы разметки: %s",
                 dict(Counter(row["review_method"] for row in rows)))
    logging.info("Пользовательские locked-цели: %s",
                 dict(sorted(Counter(
                     tuple(row["labels"]) for row in rows if row["user_locked"]
                 ).items(), key=lambda item: str(item[0]))))
    for row in uncertain[:20]:
        logging.info("Пример uncertainty: %s:%d %s", row["split"], row["row"],
                     row["uncertainties"])


def build_proposals(decisions: dict, records: list, audit: dict) -> list[dict]:
    """Пересчитать все группы одним общим способом для CLI и исправлений."""
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy_id = policy.get("id", "unknown")
    locked = _locked_from_decisions(decisions, records)
    overlap = test_overlap_reasons(records, audit)

    groups = defaultdict(list)
    for record in records:
        groups[record.split, record.group_id].append(record)

    rows = []
    for split in ("train", "test"):
        keys = sorted(
            (key for key in groups if key[0] == split),
            key=lambda key: min(record.row for record in groups[key]),
        )
        for key in keys:
            rows.append(_record_for_group(
                key[0], key[1], groups[key], locked, overlap, policy_id,
            ))

    _validate(rows, records, locked)
    return rows


def main() -> None:
    """Прочитать источники, построить предложения и записать только JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    decisions = json.loads(DECISIONS.read_bytes())
    audit = json.loads(AUDIT.read_bytes())
    records, _ = load_records(decisions, audit)
    rows = build_proposals(decisions, records, audit)
    _write_jsonl(args.output, rows)
    logging.info("Политика: %s", rows[0]["policy_id"])
    locked = _locked_from_decisions(decisions, records)
    _log_statistics(rows, records, locked, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
