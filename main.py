"""CLI-ядро проекта.

Читает входной JSON (description + mcId), прогоняет пайплайн из сервисов
и записывает выходной JSON (drafts).

Пример:
    python main.py --input data/input.json --output out/output.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from category_validation import CategoryValidationPipeline
from common.schemas import PipelineInput, PipelineOutput
from draft_generation import DraftGenerationPipeline
from subcategory_extraction import SubcategoryExtractionPipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """CLI: пути к файлу ввода и файлу вывода."""
    parser = argparse.ArgumentParser(description="Выделение услуг и черновики объявлений")
    parser.add_argument("--input", required=True, type=Path, help="путь к JSON на вход")
    parser.add_argument("--output", required=True, type=Path, help="путь к JSON на выход")
    return parser


def run_pipeline(inp: PipelineInput) -> PipelineOutput:
    """Прогнать цепочку сервисов: выделение -> проверка -> генерация."""
    extraction = SubcategoryExtractionPipeline()
    validation = CategoryValidationPipeline()
    generation = DraftGenerationPipeline()

    candidates = extraction.run(inp)
    target_mc_ids = validation.run(candidates, inp.mc_id)
    drafts = generation.run(inp, target_mc_ids)
    return PipelineOutput(drafts=drafts)


def main(argv: list[str] | None = None) -> int:
    """Прочитать входной файл, прогнать пайплайн, записать выходной файл."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()  # переменные из .env (LLM_MODEL_ID, LLM_API_KEY)
    args = build_parser().parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    inp = PipelineInput.model_validate(payload)

    output = run_pipeline(inp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output.model_dump(by_alias=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Готово: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
