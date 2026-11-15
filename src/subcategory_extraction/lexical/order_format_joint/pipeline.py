"""Совместная проверка предложения услуг и запретов в объявлении."""

import re

from main import PipelineInput
from ..order_format_phrase_logreg.pipeline import HOMOGLYPHS
from ..order_format_veto.pipeline import clean_description, veto
from ..snippet_logistic_regression.pipeline import SubcategoryExtractionPipeline as CategoryPipeline

PLASTER = re.compile(r"\b(?:штукатур\w*|оштукатур\w*|выравнива\w*\s+стен\w*\s+по\s+маяк\w*)\b")
TILE = re.compile(r"\b(?:плитк\w*|плиточ\w*|керамогранит\w*|мозаик\w*)\b")
ELECTRICAL = re.compile(
    r"\b(?:электрик\w*|электромонтаж\w*|электропровод\w*|проводк\w*|"
    r"электрощит\w*|розет\w*|выключател\w*|светильник\w*|подрозетник\w*)\b"
)
GROUNDED = {102: PLASTER, 103: TILE, 105: ELECTRICAL}
MATERIAL_HEADER = re.compile(
    r"\b(?:работа\w*\s+с\s+материалами|наши\s+материалы|список\s+материалов|"
    r"материалы(?:\s+для\s+работы)?)\s*:?[\s.!]*$"
)
LIST_ITEM = re.compile(r"^[\W_]+")
SERVICE_HEADER = re.compile(r"\b(?:что (?:мы )?делаем|наши услуги|выполняем|перечень работ)\s*:$")
MATERIAL_REFERENCE = re.compile(
    r"\b(?:мешк\w*|закупаем|доставляем|материал\w*\s*[:—]|штукатурн\w*\s+состав\w*)\b"
)
DIRECT_TILE = re.compile(
    r"\b(?:облицовк\w*|облицую|облицуем|укладк\w*|укладываю|укладываем|"
    r"укладывается|уложу|уложим|кладу|кладем|кладет|положим)\b"
    r"[^.!?;,:\n]{0,65}\b(?:плитк\w*|керамогранит\w*)\b"
)
NEGATIVE_OFFER = re.compile(
    r"\b(?:без|никакой|не\s+(?:делаем|делаю|берем|беру|выполняем|"
    r"входит|укладываем|укладываю|облицуем|кладем|кладут|кладет))\b"
)


def service_text(description: str) -> str:
    """Отделить список материалов от текста, в котором ищем услуги."""
    lines = []
    materials = False
    for line in description.casefold().translate(HOMOGLYPHS).splitlines():
        line = line.strip()
        if MATERIAL_HEADER.search(line):
            materials = True
            continue
        if not line or SERVICE_HEADER.search(line):
            materials = False
        elif materials and LIST_ITEM.match(line):
            continue
        else:
            materials = False
        lines.append(line)
    return "\n".join(lines)


class SubcategoryExtractionPipeline(CategoryPipeline):
    """Получить кандидатов из основного логрега и применить фильтр запретов."""

    name = "order_format_joint"

    def candidates(self, inp: PipelineInput) -> list[int]:
        """Проверить слабые кандидаты и передать распознанную облицовку логрегу."""
        text = service_text(inp.description)
        evidence_text = "\n".join(line for line in text.splitlines() if not MATERIAL_REFERENCE.search(line))
        candidates = {
            cid for cid in super().run(inp)
            if cid not in GROUNDED or GROUNDED[cid].search(evidence_text)
        }
        if 103 not in candidates and inp.mc_id != 103:
            for clause in re.split(r"[.!?;\n]", evidence_text):
                match = DIRECT_TILE.search(clause)
                if not match or NEGATIVE_OFFER.search(clause):
                    continue
                # Веса логрега ожидают знакомую формулировку «укладка плитки».
                hint = inp.model_copy(update={"description": "Укладка плитки. " + match[0]})
                if 103 in super().run(hint):
                    candidates.add(103)
                    break
        return sorted(candidates)

    def run(self, inp: PipelineInput) -> list[int]:
        clean = inp.model_copy(update={"description": clean_description(inp.description)})
        return veto(clean.description, self.candidates(clean))[0]
