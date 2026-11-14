"""Офлайн-проверка генератора sample_llm без вызова платного API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from time import perf_counter
from benchmark_support import metadata, save_report

import json
import logging
from io import StringIO
from urllib.error import HTTPError
from unittest.mock import patch

from main import PipelineConfig, PipelineInput, load_pipeline, run_pipeline

from draft_generation.sample_llm.client import SampleLLMClient

logger = logging.getLogger(__name__)


def replay() -> dict[str, int]:
    """Проверить контракт, транспорт, пустые цели и некорректные ответы."""
    config = PipelineConfig(
        subcategory_extraction="google_decision",
        draft_generation="sample_llm",
    )
    inp = PipelineInput(description="Устанавливаю розетки и смесители.", mcId=101)
    pipeline = load_pipeline("draft_generation", config)
    answer = '{"title":"Монтаж","text":"Установка смесителей."}'
    with patch.object(pipeline.client, "complete", return_value=answer) as complete:
        drafts = pipeline.run(inp, [104, 104, 105])
        assert [draft["mcId"] for draft in drafts] == [104, 105]
        assert complete.call_count == 2
        assert complete.call_args.kwargs["response_schema"]["required"] == ["title", "text"]
        assert "Сантехника" in complete.call_args_list[0].args[0]
        assert inp.description in complete.call_args_list[0].args[0]
    with patch.object(pipeline.client, "complete") as complete:
        assert pipeline.run(inp, []) == []
        complete.assert_not_called()
    for targets in ([101], [999], [True], [104, 999]):
        with patch.object(pipeline.client, "complete") as complete:
            try:
                pipeline.run(inp, targets)
            except ValueError:
                pass
            else:
                raise AssertionError("Некорректные цели приняты")
            complete.assert_not_called()
    invalid = [
        "not json", "[]", "{}", '{"title":"x","text":null}',
        '{"title":"x","text":" "}', '{"title":42,"text":"x"}',
        '{"title":"x","text":"y","mcId":105}',
    ]
    for response in invalid:
        with patch.object(pipeline.client, "complete", return_value=response):
            try:
                pipeline.run(inp, [104])
            except ValueError:
                continue
            raise AssertionError("Некорректный ответ принят")
    with patch.dict("os.environ", {
        "LLM_API_KEY": "offline-placeholder",
        "SAMPLE_LLM_API_KEY": "ignored-placeholder",
        "SAMPLE_LLM_MODEL_ID": "offline-model",
        "SAMPLE_LLM_API_URL": "https://example.invalid/models/{model}:generateContent",
    }):
        client = SampleLLMClient()
        response = json.dumps({"candidates": [{"content": {"parts": [{"text": answer}]}}]})
        with patch("draft_generation.sample_llm.client.urlopen",
                   return_value=StringIO(response)) as transport:
            schema = {"type": "object", "required": ["title", "text"]}
            assert client.complete("Контроль", response_schema=schema) == answer
            sent = json.loads(transport.call_args.args[0].data)
            assert transport.call_args.args[0].get_header("X-goog-api-key") == "offline-placeholder"
            assert transport.call_args.args[0].full_url.endswith("/offline-model:generateContent")
            assert sent["contents"] == [{"role": "user", "parts": [{"text": "Контроль"}]}]
            assert sent["generationConfig"] == {
                "responseMimeType": "application/json", "responseJsonSchema": schema,
            }
        client.max_attempts = 2
        with (
            patch("draft_generation.sample_llm.client.urlopen", side_effect=[
                HTTPError(client.api_url, 429, "quota", {}, None), StringIO(response),
            ]) as transport,
            patch("draft_generation.sample_llm.client.sleep") as sleep,
        ):
            assert client.complete("Контроль") == answer
            assert transport.call_count == 2
            sleep.assert_called_once_with(client.retry_delay)
        for status in (401, 429):
            with (
                patch("draft_generation.sample_llm.client.urlopen",
                      side_effect=HTTPError(client.api_url, status, "error", {}, None)) as transport,
                patch("draft_generation.sample_llm.client.sleep"),
            ):
                try:
                    client.complete("Контроль")
                except HTTPError as exc:
                    assert exc.code == status
                else:
                    raise AssertionError("Ошибка HTTP скрыта")
                assert transport.call_count == (1 if status == 401 else 2)
        for body in ({}, {"candidates": []}, {"candidates": [{"content": {"parts": []}}]}):
            with patch("draft_generation.sample_llm.client.urlopen",
                       return_value=StringIO(json.dumps(body))):
                try:
                    client.complete("Контроль")
                except ValueError:
                    continue
                raise AssertionError("Некорректная оболочка ответа принята")
    with patch.dict("os.environ", {"SAMPLE_LLM_API_KEY": "ignored-placeholder", "LLM_API_KEY": ""}):
        try:
            SampleLLMClient().complete("Контроль")
        except ValueError:
            pass
        else:
            raise AssertionError("Отсутствие ключа не обнаружено")
    # Настоящие два модуля и оркестрация, только ответы API заменены контрольными.
    with (
        patch("subcategory_extraction.google_decision.client.GoogleDecisionClient.complete",
              return_value=json.dumps({"answers": {
                  f"mc{cid}": {"noul": 0.9 if cid == 104 else 0.0}
                  for cid in range(102, 107)
              }})),
        patch("draft_generation.sample_llm.client.SampleLLMClient.complete",
              return_value=answer),
        patch.dict("os.environ", {"GOOGLE_DECISION_THRESHOLD": "0.5"}),
    ):
        output = run_pipeline(inp, config)
        assert [draft.mc_id for draft in output.drafts] == [104]
        assert output.drafts[0].text == "Установка смесителей."
    return {"invalid_json_rejected": len(invalid), "transport_errors_rejected": 3,
            "drafts": len(drafts), "full_pipeline": 1}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    started = perf_counter()
    report = {**metadata(), "scenario": "offline_contract"}
    report["checks"] = replay()
    report.update(status="complete", elapsed_seconds=perf_counter() - started)
    save_report(Path(__file__).with_name("results.json"), report)
    logger.info("sample_llm: %s", report)
