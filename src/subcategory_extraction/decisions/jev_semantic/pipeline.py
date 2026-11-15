"""Jev с общими границами категорий и проверкой текущего предложения услуги."""

import hashlib
import json
import os
from pathlib import Path

from main import MICROCATEGORIES, PipelineInput
from subcategory_extraction.decisions.jev.pipeline import SubcategoryExtractionPipeline as JevPipeline

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "artifacts/calibration.json"
VARIANTS = ("task", "strict")
DEFINITIONS = {
    101: "Комплексный ремонт квартиры, дома, комнаты или санузла с несколькими этапами и готовым результатом. Косметический, капитальный и дизайнерский ремонт помещения подходят. Одна операция, например плитка под ключ, и только чистовая отделка не доказывают ремонт целого помещения.",
    102: "Выравнивание стен штукатурным составом, штукатурка по маякам, механизированное оштукатуривание, штукатурка углов и откосов. Шпаклёвка, шлифовка и грунтование под обои/краску —106, не102. Тонкослойное декоративное покрытие без выравнивания не доказывает штукатурные работы.",
    103: "Укладка плитки, кафеля, керамогранита и мозаики, облицовка, кухонный фартук, замена плиток, затирка и обновление швов, запил45°, плиточный поддон. Только снятие/демонтаж старой плитки и продажа материалов не доказывают предложение этих работ.",
    104: "Монтаж, замена, подключение и ремонт водоснабжения, канализации, труб, смесителей, сифонов, сантехприборов, инсталляций, прочистка засоров, устранение протечек, водяное отопление, радиаторы, подключение стиральной/посудомоечной машины к воде. Само помещение санузла и его облицовка не означают сантехнику.",
    105: "Монтаж, замена, перенос и диагностика электропроводки, розеток, выключателей, освещения, щитов, автоматов, УЗО, электрического тёплого пола. Водяное отопление, работа исправным электроинструментом и техника, упомянутая только как товар, не являются электромонтажом.",
    106: "Поклейка/замена обоев, окраска стен/потолков и малярная подготовка: шпаклёвка, шлифовка, грунтование под обои или краску, стеклохолст, снятие старых обоев при обновлении отделки. Шпаклёвка относится сюда даже рядом со штукатуркой. Печать/продажа обоев без поклейки и только облицовка плиткой не подходят.",
}


def prompt_sha256(variant: str) -> str:
    """Связать калибровку одновременно с инструкцией и границами всех категорий."""
    payload = (HERE / "prompts" / f"{variant}.md").read_bytes() + json.dumps(DEFINITIONS, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class SubcategoryExtractionPipeline(JevPipeline):
    """Применить общие правила ко всем объявлениям без узнавания отдельных текстов."""

    name = "jev_semantic"

    def __init__(self, variant: str | None = None) -> None:
        super().__init__()
        selected = variant or os.getenv("JEV_SEMANTIC_PROMPT", "auto")
        self.thresholds = dict.fromkeys(MICROCATEGORIES, 0.5)
        self.calibration = None
        if selected == "auto":
            data = json.loads(CALIBRATION.read_text())
            selected = data["prompt"]
            if (data["provider"] != self.client.provider or data["model"] != self.client.model_id
                    or data["endpoint"] != self.client.endpoint or data["prompt_sha256"] != prompt_sha256(selected)):
                raise ValueError("Калибровка относится к другому Jev или промпту")
            self.thresholds = {int(cid): float(value) for cid, value in data["thresholds"].items()}
            if set(self.thresholds) != set(MICROCATEGORIES) or any(not 0 <= value <= 1 for value in self.thresholds.values()):
                raise ValueError("Некорректная калибровка")
            self.calibration = data
        if selected not in VARIANTS:
            raise ValueError("JEV_SEMANTIC_PROMPT: auto, task или strict")
        self.variant = selected
        self.prompt = (HERE / "prompts" / f"{selected}.md").read_text()

    def probabilities(self, inp: PipelineInput) -> dict[int, float]:
        """Получить нативные вероятности всех разрешённых категорий по полному тексту."""
        questions = {f"mc{cid}": {"type": "noul", "instructions": self.prompt.format(category=name, definition=DEFINITIONS[cid])}
                     for cid, name in MICROCATEGORIES.items() if cid != inp.mc_id}
        result = json.loads(self.client.complete(inp.description, questions=questions))
        self.last_probabilities = {int(name[2:]): float(answer["noul"]) for name, answer in result["answers"].items()}
        return self.last_probabilities.copy()

    def run(self, inp: PipelineInput) -> list[int]:
        """Применить train-пороги; явные экспериментальные промпты используют фиксированный0.5."""
        return [cid for cid, score in self.probabilities(inp).items() if score >= self.thresholds[cid]]

    def benchmark_metadata(self) -> dict:
        """Записать происхождение без ключей и без private-меток в настройках модели."""
        return {"provider": self.client.provider, "model": self.client.model_id, "endpoint": self.client.endpoint,
                "prompt": self.variant, "prompt_sha256": prompt_sha256(self.variant), "thresholds": self.thresholds,
                "training_dataset": "manual_train" if self.calibration else None,
                "calibration_sha256": hashlib.sha256(CALIBRATION.read_bytes()).hexdigest() if self.calibration else None}
