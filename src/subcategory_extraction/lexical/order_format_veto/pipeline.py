"""Выделить услуги из объявления и применить только подтверждённые запреты."""

import re

from main import PipelineInput
from ..order_format_phrase_logreg.pipeline import HOMOGLYPHS, phrases
from ..order_format_scope.pipeline import ALTERNATIVE, PARTIAL_ORDER, scope_evidence, service_scope
from ..snippet_logistic_regression.pipeline import SubcategoryExtractionPipeline as CategoryPipeline

SEARCH_HEADER = re.compile(
    r"(?<!\w)(?:возможно[,\s]+вы\s+искали|вы\s+искали\s+меня\s+по\s+запросу|"
    r"(?:поисковые|ключевые|похожие)\s+(?:запросы|фразы|слова|теги)|"
    r"ключи\s+для\s+поиска|для\s+поиска|(?:как\s+)?нас\s+ищут|"
    r"теги(?:\s+поиска)?|х[эе]штеги)\s*(?::|$)", re.I,
)
NEW_SECTION = re.compile(
    r"^(?:характеристики|параметры|тип\s+работ|объект|исполнитель|условия|"
    r"перечень\s+работ|что\s+мы\s+делаем|наши\s+услуги)\s*:", re.I,
)
SERVICE_LIST_HEADER = re.compile(
    r"(?:наши\s+услуги|перечень\s+работ|что\s+(?:мы\s+)?делаем|"
    r"выполняем(?:\s+следующие)?(?:\s+виды)?\s+работ(?:ы)?|"
    r"(?:в\s+\w+\s+)?входит)\s*:\s*$", re.I,
)
HASHTAG = re.compile(r"(?<!\w)#[\w-]+")
TAG_SEPARATOR = re.compile(r"[,;|]")
CLAUSE_BREAK = re.compile(r"[;.!?]|,\s*(?:но|зато|однако)\s+")
CONTACT_REFUSAL = re.compile(
    r"\b(?:не\s+(?:звоните|пишите|обращайтесь|беспокойте|предлагайте)|"
    r"просьба\s+(?:воздержаться\s+от|не\s+обращаться))\b"
)
SERVICE_REFUSAL = re.compile(
    r"^(?:\W|мы\s+|я\s+)*(?:никак(?:ой|их)\s+(?P<without>[\w\s]+)|"
    r"(?P<object>[\w\s]{3,65}?)\s+не\s+(?:делаю|делаем|выполняю|выполняем|беру|берем)|"
    r"не\s+(?:делаю|делаем|выполняю|выполняем|беру|берем|занимаюсь|занимаемся)\s+(?P<after>[\w\s]{3,65}))$"
)
LIST_PREFIX = re.compile(r"^[\W\d_]+")
NARROW = re.compile(
    r"\b(?:замен\w*|ремонт\w*|одн\w*|розет\w*|стояк\w*|счетчик\w*|"
    r"венец|венециан\w*|декоративн\w*|точечн\w*|мелк\w*|разов\w*|демонтаж\w*|частичн\w*)\b"
)
PERMISSION_ACTION = re.compile(
    r"\b(?:можно|можем|возможен|возможна|возможно|беру(?:сь)?|берем(?:ся)?|принимаем|"
    r"делаем|сделаем|выполняем|работаем|поможем|заключаем|заключить)\b"
)


def tag_list(line: str) -> bool:
    """Распознать отдельную плотную строку поисковых тегов без заголовка."""
    line = line.rstrip(" .")
    tags = [part.strip() for part in TAG_SEPARATOR.split(line) if part.strip()]
    return (
        len(tags) >= 8
        and all(1 <= len(part.split()) <= 8 for part in tags)
        and not re.search(r"[.!?\n]|\b(?:делаем|выполняем|укладываем|предлагаем|монтируем)\b", line, re.I)
    )


def clean_description(description: str) -> str:
    """Убрать поисковые блоки и хэштеги перед обоими этапами."""
    kept = []
    in_search = False
    for line in description.splitlines():
        if match := SEARCH_HEADER.search(line):
            kept.append(line[:match.start()].rstrip())
            in_search = True
            continue
        if NEW_SECTION.match(line.lstrip()):
            in_search = False
        if in_search:
            continue
        previous = next((part for part in reversed(kept) if part.strip()), "")
        if not tag_list(line) or SERVICE_LIST_HEADER.search(previous):
            kept.append(HASHTAG.sub("", line).rstrip())
    return "\n".join(kept).strip()


def veto(description: str, categories: list[int]) -> tuple[list[int], list[dict]]:
    """Снять только категории, на которые есть подтверждённое ограничение."""
    if not categories:
        return categories, []
    _, scoped = scope_evidence(description, categories)
    candidates = set(categories)
    general = False
    specific: set[int] = set()
    exceptions: set[int] = set()
    evidence = []
    for item in scoped:
        label, affected = item["label"], set(item["mc_ids"])
        normalized = item["text"].casefold().translate(HOMOGLYPHS)
        if label == "allow":
            if not CONTACT_REFUSAL.search(normalized) and (
                PERMISSION_ACTION.search(normalized) or ALTERNATIVE.search(normalized)
            ):
                exceptions.update(affected)
            continue
        if label == "block_all":
            general = True
        else:
            specific.update(affected)
        evidence.append(item)

    for part in phrases(description):
        for clause in CLAUSE_BREAK.split(part.casefold().translate(HOMOGLYPHS)):
            clause = LIST_PREFIX.sub("", " ".join(clause.split()).strip(" :—–"))
            if CONTACT_REFUSAL.search(clause) and re.search(PARTIAL_ORDER, clause):
                named = service_scope(clause, candidates)
                if named:
                    specific.update(named)
                else:
                    general = True
                evidence.append({
                    "text": part, "label": "block_category" if named else "block_all",
                    "mc_ids": sorted(named or candidates),
                })
            elif match := SERVICE_REFUSAL.fullmatch(clause):
                subject = next(value for value in match.groups() if value)
                named = service_scope(subject, candidates)
                if named and not NARROW.search(subject):
                    specific.update(named)
                    evidence.append({"text": part, "label": "block_category", "mc_ids": sorted(named)})
    return [
        cid for cid in categories if cid not in specific and (not general or cid in exceptions)
    ], evidence


class SubcategoryExtractionPipeline(CategoryPipeline):
    """Применить к категориям логрега фильтр явных запретов."""

    name = "order_format_veto"

    def run(self, inp: PipelineInput) -> list[int]:
        clean = inp.model_copy(update={"description": clean_description(inp.description)})
        return veto(clean.description, super().run(clean))[0]
