"""Проверка действия и области ограничения в кандидатном фрагменте."""

import re

from main import PipelineInput
from ..order_format_phrase_logreg.pipeline import (
    CATEGORY_TERMS,
    HOMOGLYPHS,
    phrases,
)
from ..snippet_logistic_regression.pipeline import SubcategoryExtractionPipeline as CategoryPipeline

WORDS = r"(?:[\s«»—–-]+\w+)"
NEGATIVE_VERB = (
    r"\bне\s+(?:бер[еуы]\w*|дела\w*|выполня\w*|занима\w*|рассматрива\w*|"
    r"принима\w*|прим\w*|выезжа\w*|отправля\w*|работа\w*|смож\w*|мож\w*)\b"
)
ORDER_MODIFIER = r"(?:отдельн\w+|частичн\w+|разов\w+|мелк\w+|локальн\w+|одиночн\w+|минимальн\w+)"
PARTIAL_ORDER = (
    rf"{ORDER_MODIFIER}(?:(?:\s+и|\s+или)?\s+{ORDER_MODIFIER})*"
    r"\s+(?:вид\w*\s+)?(?:работ\w*|ремонт\w*|заказ\w*|услуг\w*|задач\w*|заяв\w*|"
    r"этап\w*|комнат\w*)|(?:один|одну)\s+(?:вид\s+)?(?:этап|работу|комнату)"
)
PACKAGE = (
    r"(?:комплексн\w+|полн\w+)\s+(?:ремонт\w*|проект\w*|отделк\w*|цикл\w*|заказ\w*)"
    r"|(?:ремонт\w*\s+|отделк\w*\s+)?(?:помещени\w*|объект\w*|квартир\w*|дом\w*)\s+целиком"
    r"|ремонт\w*\s+целиком"
    r"|(?:весь|всю|всей)\s+(?:цикл\w*|квартир\w*|дом\w*)"
    r"|под\s+ключ"
)
EXCLUSIVE = r"(?<!не )\b(?:только|исключительно|лишь)\b"
GLOBAL = re.compile(
    rf"{EXCLUSIVE}(?:\s+(?:по|на|с|в|для|объекты|заказы|полностью|выполняем|всю|дома|квартиры)){{0,4}}"
    rf"[\s«»🔥]*?(?:{PACKAGE})"
    rf"|\bбез\s+(?:{PARTIAL_ORDER})"
    rf"|(?:{NEGATIVE_VERB}){WORDS}{{0,3}}\s*[:—-]?\s*(?:{PARTIAL_ORDER})"
    rf"|(?:{PARTIAL_ORDER}){WORDS}{{0,6}}\s+(?:{NEGATIVE_VERB})"
)
GLOBAL_RELATION = re.compile(
    rf"\b(?:выезжа\w*|договор\w*|бер[еуы]\w*)\b.{{0,32}}"
    rf"\b(?:если нужен|лишь на|только на)\b.{{0,18}}(?:{PACKAGE}|отделк\w+\s+всего\s+дома)"
    rf"|\bа не с\s+(?:{PARTIAL_ORDER})"
    rf"|(?:{PARTIAL_ORDER})[^.!?]{{0,30}}\b(?:бригад\w*\s+)?(?:{NEGATIVE_VERB})"
    rf"|(?:{NEGATIVE_VERB})[^.!?]{{0,40}}\bзаказ\w*\b[^.!?]{{0,25}}(?:{PARTIAL_ORDER})"
    rf"|(?:{PARTIAL_ORDER}){WORDS}{{0,3}}\s+заказ\w*\s+(?:{NEGATIVE_VERB})"
)
GLOBAL_ACTION = re.compile(
    r"\b(?:работа\w*|дела\w*|ремонт\w*|выполня\w*|реализу\w*|принима\w*|бер[еуы]\w*|"
    r"подряд\w*|договор\w*|специализир\w*)\b"
)
ALTERNATIVE = re.compile(
    rf"\bкак\b.+\bтак\s+и\b.+|\bне\s+только\b|\bтакже\b|\bи\s+(?:{PARTIAL_ORDER})"
    rf"|\bили\s+(?:{PARTIAL_ORDER})"
)
PERMISSION = re.compile(
    rf"\b(?:можно|можем|возмож\w*|принима\w*|бер[еуы]\w*|заключ\w*|сдела\w*|выполня\w*|"
    rf"помож\w*|работа\w*)\b"
)
SEPARATE = re.compile(
    rf"\b(?:отдельно|самостоятельн\w+\s+заказ\w*|отдельн\w+\s+договор\w*|{PARTIAL_ORDER})\b"
)
CLAUSE_BOUNDARY = re.compile(
    r"[,.;]\s*\b(?:но|однако|зато)\s+|,\s*(?=[^,;]{1,50}\s+можно(?:\s|[.!?]|$))"
)


def service_scope(text: str, categories: set[int]) -> set[int]:
    """Услуги, которые прямо названы в проверяемой части фразы."""
    return {cid for cid in categories if cid in CATEGORY_TERMS and CATEGORY_TERMS[cid].search(text)}


def scope_evidence(
    description: str, categories: list[int], required_quotes: set[str] | None = None,
) -> tuple[list[int], list[dict]]:
    """Потребовать связь ограничения с заказом и разрешить явные исключения."""
    all_categories = set(categories)
    allowed: set[int] = set()
    blocked_by_service: set[int] = set()
    general_refusal = False
    general_permission = False
    evidence = []
    for part in phrases(description):
        text = " ".join(part.casefold().translate(HOMOGLYPHS).split())
        # «Но» отделяет ограничение от исключения, «или» и «как … так и» остаются целыми.
        explicit_separate = False
        for clause in CLAUSE_BOUNDARY.split(text):
            negative = bool(re.search(NEGATIVE_VERB, clause))
            alternative = ALTERNATIVE.search(clause) and not negative and not re.search(r"\bбез\b", clause)
            denies = (GLOBAL.search(clause) or GLOBAL_RELATION.search(clause)) if not alternative else None
            order_scope = SEPARATE.search(clause)
            named = service_scope(clause, all_categories)
            short_permission = explicit_separate and named and re.search(r"\bможно[.!?]?$", clause)
            explicit_separate = explicit_separate or bool(order_scope)
            can_block = required_quotes is None or part in required_quotes
            affected: set[int] = set()
            label = ""
            if can_block and denies and (negative or GLOBAL_ACTION.search(clause) or "выезжаем" in clause):
                # «Не доделываем за другими» и «не берём мелкий ремонт»
                # говорят о части заказов. Без общего условия не снимаем весь класс.
                if re.search(PACKAGE, denies[0]) or re.search(r"отдельн|частичн|разов|один|одну", denies[0]) or (
                    re.search(r"полн\w+\s+ремонт|полноценн\w+\s+ремонт", clause) and negative
                ):
                    affected, label = all_categories, "block_all"
                    general_refusal = True
            elif can_block and negative and order_scope and named and not alternative:
                affected, label = named, "block_category"
                blocked_by_service.update(affected)
            elif not negative and (
                short_permission or (order_scope and (PERMISSION.search(clause) or alternative))
            ):
                affected = named
                label = "allow"
                allowed.update(affected)
                if not named and re.search(PARTIAL_ORDER, clause):
                    general_permission = True
                    affected = all_categories
            if affected:
                evidence.append({"text": part, "label": label, "mc_ids": sorted(affected)})
    remaining = [
        cid for cid in categories
        if cid in allowed or (cid not in blocked_by_service and (not general_refusal or general_permission))
    ]
    return remaining, evidence


class SubcategoryExtractionPipeline(CategoryPipeline):
    """Проверить область ограничения для найденных категорий."""

    name = "order_format_scope"

    def run(self, inp: PipelineInput) -> list[int]:
        return scope_evidence(inp.description, super().run(inp))[0]
