"""Проверить модуль Google без сети на сохранённых ответах и ошибках схемы."""

import json
import logging
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from main import PipelineConfig, PipelineInput, load_pipeline
from subcategory_extraction.benchmarks.metrics import score
from subcategory_extraction.google_decision.client import GoogleDecisionClient

ROOT = Path(__file__).resolve().parents[3]
HISTORICAL = ROOT / "src/subcategory_extraction/benchmarks/historical"


def replay() -> dict:
    """Пропустить сохранённые вероятности через публичный run модуля."""
    config = PipelineConfig(
        subcategory_extraction="google_decision",
        draft_generation=None,
    )
    results = {}
    saved = json.loads((HISTORICAL / "2026-11-13_decision-1_seed43.json").read_text())
    with patch.dict("os.environ", {"GOOGLE_DECISION_THRESHOLD": "0.5"}):
        pipeline = load_pipeline("subcategory_extraction", config)
    for variant in ("baseline", "guided"):
        expected, predicted = [], []
        for row, probabilities, old in zip(
            saved["sample"], saved["results"][variant]["probabilities"],
            saved["results"][variant]["predicted"], strict=True,
        ):
            inp = PipelineInput(description="Офлайн-повтор сохранённых вероятностей",
                                mcId=row["source_mc_id"])
            response = json.dumps({"answers": {
                f"mc{cid}": {"noul": probability} for cid, probability in probabilities.items()
            }})
            with patch.object(pipeline.client, "complete", return_value=response):
                output = pipeline.run(inp)
            if set(output) != set(old):
                raise ValueError("Перенос изменил предсказание")
            expected.append(set(row["expected"]))
            predicted.append(set(output))
        results[variant] = {"rows": len(predicted), **score(expected, predicted)}
    inp = PipelineInput(description="Сантехнические работы", mcId=101)
    fresh = json.loads((HISTORICAL / "2026-11-13_decision-1_fresh32.json").read_text())
    expected, predicted = [], []
    for row in fresh["rows"]:
        response = json.dumps({"answers": {
            f"mc{cid}": {"noul": value} for cid, value in row["probabilities"].items()
        }})
        with patch.object(pipeline.client, "complete", return_value=response):
            output = pipeline.run(PipelineInput(
                description="Офлайн-повтор свежей выборки", mcId=row["source_mc_id"],
            ))
        if set(output) != set(row["decision_1"]):
            raise ValueError("Изменилось предсказание на fresh32")
        expected.append(set(row["expected"]))
        predicted.append(set(output))
    results["fresh32"] = {"rows": len(predicted), **score(expected, predicted)}
    invalid = ["not json", "[]", '{"answers":[]}', '{"answers":{}}']
    for value in (True, "0.5", -0.1, 1.1, None, float("nan")):
        invalid.append(json.dumps({"answers": {
            f"mc{cid}": {"noul": value if cid == 104 else 0.1} for cid in range(102, 107)
        }}))
    for response in invalid:
        with patch.object(pipeline.client, "complete", return_value=response):
            try:
                pipeline.run(inp)
            except (ValueError, TypeError):
                continue
        raise ValueError("Некорректный ответ не отклонён")
    with patch.object(pipeline.client, "complete") as complete:
        if pipeline.probabilities(inp, set()) != {} or complete.called:
            raise ValueError("Пустой список категорий должен обходиться без API")
    client = GoogleDecisionClient()
    client.api_key = "offline-placeholder"
    response = json.dumps({
        "answers": {"mc104": {"noul": 0.8}}, "usage": {"total_input_tokens": 123},
    })
    with patch("subcategory_extraction.google_decision.client.urlopen",
               return_value=StringIO(response)) as transport:
        decoded = json.loads(client.complete(inp.description, questions={"mc104": {"type": "noul"}}))
        sent = json.loads(transport.call_args.args[0].data)
        if (
            decoded["answers"]["mc104"]["noul"] != 0.8
            or client.budget.last_actual != 123
            or sent["input"] != [{"type": "text", "text": inp.description}]
            or sent["questions"] != {"mc104": {"type": "noul"}}
        ):
            raise ValueError("Транспорт клиента изменил payload или расход токенов")
    with patch.dict("os.environ", {"LLM_API_KEY": "", "GOOGLE_API_KEY": ""}):
        client = GoogleDecisionClient()
        try:
            client.complete("Описание", questions={"mc104": {"type": "noul"}})
        except ValueError:
            pass
        else:
            raise ValueError("Отсутствие ключа не обнаружено")
    results["invalid_responses_rejected"] = len(invalid)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = replay()
    output = ROOT / "out/google-replay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logging.info("Google replay: %s", json.dumps(results))
