"""Микрокатегории кейса «Ремонт и отделка» и хелперы для работы с ними.

Таблица зафиксирована условиями кейса (docs/Innoglobalhack-Avito_2026.pdf):
новые микрокатегории не создаются, выходить за таблицу нельзя.
"""

MICROCATEGORIES: dict[int, str] = {
    101: "Ремонт под ключ",
    102: "Штукатурные работы",
    103: "Плиточные работы",
    104: "Сантехника",
    105: "Электрика",
    106: "Поклейка обоев и малярные работы",
}

VALID_MCIDS: frozenset[int] = frozenset(MICROCATEGORIES)


def is_valid_mcid(mcid: int) -> bool:
    """Проверить, что mcId есть в таблице микрокатегорий кейса."""
    return mcid in VALID_MCIDS


def mcid_name(mcid: int) -> str | None:
    """Название микрокатегории по mcId или None, если такой категории нет."""
    return MICROCATEGORIES.get(mcid)
