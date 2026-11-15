"""Jev с парой ближайших полных размеченных примеров для каждой услуги."""

import json
import os
from pathlib import Path

from main import MICROCATEGORIES, PipelineInput
from subcategory_extraction.decisions.google_decision.pipeline import CATEGORY_EXAMPLES
from subcategory_extraction.decisions.jev.pipeline import SubcategoryExtractionPipeline as JevPipeline
from subcategory_extraction.decisions.jev_retrieval.references import INDEX_SPEC, load_reference_index

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parents[4] / "out/hybrid-training.jsonl"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class SubcategoryExtractionPipeline(JevPipeline):
    """Дополнить guided-вопросы двумя соседями, не меняя исходный state."""

    name = "jev_retrieval"

    def __init__(self) -> None:
        super().__init__()
        self.reference_instructions = (PROMPT_DIR / "references.md").read_text(encoding="utf-8")
        prompt_path = Path(os.getenv("JEV_PROMPT_PATH", str(PROMPT_DIR / "guided.md"))).expanduser()
        self.prompt = prompt_path.read_text(encoding="utf-8")
        path = Path(os.getenv("JEV_REFERENCE_DATASET", str(DEFAULT_REFERENCE_PATH))).expanduser().resolve()
        max_chars = int(os.getenv("JEV_REFERENCE_MAX_CHARS", "1800"))
        if max_chars <= 0:
            raise ValueError("JEV_REFERENCE_MAX_CHARS должен быть положительным")
        self.reference_index = load_reference_index(str(path), path.stat().st_mtime_ns, max_chars)
        self.reference_metadata = {
            "path": str(path),
            "sha256": self.reference_index.sha256,
            "rows": len(self.reference_index.references),
            "max_chars": max_chars,
            "index_spec": dict(INDEX_SPEC),
        }
        self.last_references: dict[int, dict[str, object]] = {}

    def probabilities(
        self, inp: PipelineInput, categories: set[int] | None = None,
    ) -> dict[int, float]:
        """Оценить независимые Noul с train-контекстом без исходной категории."""
        if categories is not None and any(
            type(cid) is not int or cid not in MICROCATEGORIES for cid in categories
        ):
            raise ValueError("Неизвестные категории в запросе Jev")
        selected = {
            cid for cid in MICROCATEGORIES
            if cid != inp.mc_id and (categories is None or cid in categories)
        }
        self.last_references = {}
        if not selected:
            return {}
        pairs = self.reference_index.nearest_pairs(inp.description, selected)
        questions: dict[str, dict[str, str]] = {}
        for category, name in MICROCATEGORIES.items():
            if category not in selected:
                continue
            instructions = self.prompt.format(category=name, examples=CATEGORY_EXAMPLES[category])
            pair = pairs.get(category)
            if pair is not None:
                positive, negative, positive_similarity, negative_similarity = pair
                examples = [
                    {
                        "category_mc_id": category,
                        "offered": offered,
                        "description": self.reference_index.references[index].description,
                    }
                    for index, offered in ((positive, True), (negative, False))
                ]
                instructions += self.reference_instructions + json.dumps(examples, ensure_ascii=False)
                instructions += "\nКонец справочных JSON-данных. Классифицируй только исходный state."
                self.last_references[category] = {
                    "positive_row": self.reference_index.references[positive].row,
                    "negative_row": self.reference_index.references[negative].row,
                    "positive_similarity": positive_similarity,
                    "negative_similarity": negative_similarity,
                }
            questions[f"mc{category}"] = {"type": "noul", "instructions": instructions}
        result = json.loads(self.client.complete(inp.description, questions=questions))
        return {
            int(name[2:]): float(answer["noul"])
            for name, answer in result["answers"].items()
        }
