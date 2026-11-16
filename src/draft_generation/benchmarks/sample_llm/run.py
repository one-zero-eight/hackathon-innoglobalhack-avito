"""Офлайн-проверка sample_llm и local_llm без API и загрузки весов."""

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

from main import PipelineConfig, PipelineInput, discover_modules, load_pipeline, run_pipeline

from draft_generation.sample_llm.client import SampleLLMClient

logger = logging.getLogger(__name__)


def replay() -> dict[str, int]:
    """Проверить контракт, транспорт, пустые цели и некорректные ответы."""
    config = PipelineConfig(
        subcategory_extraction="decisions.google_decision",
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
    with patch.dict("os.environ", {"SELECTEL_API_KEY": "offline-placeholder"}, clear=True):
        client = SampleLLMClient()
        assert client.model_id == "qwen/qwen3.5-9b"
        assert client.api_url == "https://api.selectel.ru/aig/v1/chat/completions"
    with patch.dict("os.environ", {
        "SELECTEL_API_KEY": "offline-placeholder",
        "LLM_API_KEY": "ignored-placeholder",
        "SAMPLE_LLM_MODEL_ID": "offline-model",
        "SAMPLE_LLM_API_URL": "https://example.invalid/v1/chat/completions",
    }):
        client = SampleLLMClient()
        response = json.dumps({"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]})
        with patch("draft_generation.sample_llm.client.urlopen",
                   return_value=StringIO(response)) as transport:
            schema = {"type": "object", "required": ["title", "text"]}
            assert client.complete("Контроль", response_schema=schema) == answer
            sent = json.loads(transport.call_args.args[0].data)
            assert transport.call_args.args[0].get_header("Authorization") == "Bearer offline-placeholder"
            assert transport.call_args.args[0].full_url == "https://example.invalid/v1/chat/completions"
            assert sent == {"model": "offline-model", "messages": [{"role": "user", "content": "Контроль"}],
                            "response_format": {"type": "json_object"}}
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
        for body in ({}, {"choices": []}, {"choices": [{"message": {"content": ""}}]},
                     {"choices": [{"message": {"content": answer}, "finish_reason": "length"}]}):
            with patch("draft_generation.sample_llm.client.urlopen",
                       return_value=StringIO(json.dumps(body))):
                try:
                    client.complete("Контроль")
                except ValueError:
                    continue
                raise AssertionError("Некорректная оболочка ответа принята")
    with patch.dict("os.environ", {"SELECTEL_API_KEY": "", "LLM_API_KEY": "ignored-placeholder"}):
        try:
            SampleLLMClient().complete("Контроль")
        except ValueError:
            pass
        else:
            raise AssertionError("Отсутствие ключа не обнаружено")
    # Настоящие два модуля и оркестрация, только ответы API заменены контрольными.
    with (
        patch("subcategory_extraction.decisions.google_decision.client.GoogleDecisionClient.complete",
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


def replay_local() -> dict[str, int]:
    """Проверить буллеты, сохранность эмодзи, разделы и строгую схему ответа."""
    config = PipelineConfig(
        subcategory_extraction="decisions.jev", draft_generation="local_llm",
    )
    assert "local_llm" in discover_modules("draft_generation")
    pipeline = load_pipeline("draft_generation", config)
    inp = PipelineInput(description="Устанавливаю смесители и розетки.", mcId=101)
    structured = {
        "title": "Установка смесителей",
        "intro": "🔧 Выполняю сантехнические работы!",
        "items": ["Установка смесителей."],
        "additional_info": ["📍 Работаю в Казани.", "Стоимость согласуем до начала работ."],
    }
    electrical = {
        "title": "Установка розеток",
        "intro": "Выполняю электромонтажные работы.",
        "items": ["Установка розеток."],
        "additional_info": [],
    }
    with patch.object(pipeline.client, "complete", side_effect=[
        json.dumps(structured), json.dumps(electrical),
    ]) as complete:
        drafts = pipeline.run(inp, [104, 104, 105])
    assert complete.call_count == 2
    assert drafts == [
        {"mcId": 104, "title": structured["title"], "text":
         "🔧 Выполняю сантехнические работы!\n\n• Установка смесителей.\n\n"
         "Дополнительная информация:\n• 📍 Работаю в Казани.\n"
         "• Стоимость согласуем до начала работ."},
        {"mcId": 105, "title": electrical["title"], "text":
         "Выполняю электромонтажные работы.\n\n• Установка розеток."}
    ]
    styled_input = PipelineInput(description="🔧 Устанавливаю смесители. 📍 Работаю в Казани.", mcId=101)
    plain = {**structured, "intro": "Выполняю сантехнические работы!",
             "additional_info": ["Работаю в Казани.", "Стоимость согласуем до начала работ."]}
    style = {"intro": 1, "items": [1], "additional_info": [2, 0]}
    with patch.object(pipeline.client, "complete", side_effect=[
        json.dumps(plain), json.dumps(style),
    ]) as complete:
        styled = pipeline.run(styled_input, [104])
    assert complete.call_count == 2
    assert styled == [{
        "mcId": 104, "title": plain["title"],
        "text": "🔧 Выполняю сантехнические работы!\n\n• 🔧 Установка смесителей.\n\n"
                "Дополнительная информация:\n• 📍 Работаю в Казани.\n"
                "• Стоимость согласуем до начала работ.",
    }]
    bad_styles = [
        {**style, "intro": "Гарантия 10 лет"},
        {**style, "intro": 3},
        {**style, "intro": -1},
        {**style, "intro": True},
        {**style, "items": []},
        {**style, "items": [1.0]},
        {**style, "additional_info": [2]},
        {**style, "title": "Новый текст"},
    ]
    for bad_style in bad_styles:
        with patch.object(pipeline.client, "complete", side_effect=[
            json.dumps(plain), json.dumps(bad_style),
        ]) as complete:
            try:
                pipeline.run(styled_input, [104])
            except ValueError:
                assert complete.call_count == 2
                continue
            raise AssertionError("Стилизация изменила факты или добавила несуществующий эмодзи")
    with patch.object(pipeline.client, "complete") as complete:
        assert pipeline.run(inp, []) == []
        for targets in ([101], [999], [True], [104, 999]):
            try:
                pipeline.run(inp, targets)
            except ValueError:
                pass
            else:
                raise AssertionError("Некорректные цели приняты локальным генератором")
        complete.assert_not_called()
    with patch.object(pipeline.client, "complete", return_value="null") as complete:
        assert pipeline.run(styled_input, [104]) == []
        complete.assert_called_once()
    invalid = [
        "not json", "[]", "{}",
        json.dumps({**structured, "title": ""}),
        json.dumps({**structured, "intro": " "}),
        json.dumps({**structured, "items": []}),
        json.dumps({**structured, "items": [" "]}),
        json.dumps({**structured, "additional_info": [" "]}),
        json.dumps({**structured, "additional_info": "Работаю в Казани."}),
        json.dumps({key: value for key, value in structured.items() if key != "additional_info"}),
        json.dumps({**structured, "outro": ""}),
        json.dumps({**structured, "mcId": 105}),
    ]
    for answer in invalid:
        with patch.object(pipeline.client, "complete", return_value=answer) as complete:
            try:
                pipeline.run(inp, [104])
            except ValueError:
                complete.assert_called_once()
                continue
            raise AssertionError("Некорректные варианты приняты")
    with (
        patch("subcategory_extraction.decisions.jev.client.JevClient.complete",
              return_value=json.dumps({"answers": {
                  f"mc{cid}": {"noul": 0.9 if cid == 104 else 0.0}
                  for cid in range(102, 107)
              }})),
        patch("draft_generation.local_llm.client.LocalLLMClient.complete",
              return_value=json.dumps(plain)),
        patch.dict("os.environ", {"JEV_THRESHOLD": "0.5"}),
    ):
        output = run_pipeline(inp, config)
        assert [draft.mc_id for draft in output.drafts] == [104]
        assert output.drafts[0].text == (
            "Выполняю сантехнические работы!\n\n• Установка смесителей.\n\n"
            "Дополнительная информация:\n• Работаю в Казани.\n"
            "• Стоимость согласуем до начала работ."
        )
    return {"drafts": len(drafts), "invalid_responses_rejected": len(invalid),
            "invalid_styles_rejected": len(bad_styles), "full_pipeline": 1}


def replay_local_transport() -> dict[str, int]:
    """Проверить локальные адреса, запрос и отклонение неполных ответов без повторов."""
    from draft_generation.local_llm.client import LocalLLMClient

    with patch.dict("os.environ", {
        "LOCAL_LLM_MODEL_ID": "default_model",
        "LOCAL_LLM_API_URL": "http://host.docker.internal:8080/v1/chat/completions",
    }):
        client = LocalLLMClient()
    result = {"choices": [{"message": {"content": '{"title":"Монтаж"}'},
                           "finish_reason": "stop"}]}
    with patch.object(client, "request_json", return_value=result) as transport:
        assert client.complete("Контроль") == '{"title":"Монтаж"}'
        transport.assert_called_once()
        request, timeout = transport.call_args.args
        sent = json.loads(request.data)
        assert request.full_url == "http://host.docker.internal:8080/v1/chat/completions"
        assert timeout == client.timeout
        assert sent == {
            "model": "default_model",
            "messages": [{"role": "user", "content": "Контроль"}],
            "temperature": client.temperature,
            "max_tokens": client.max_tokens,
            "seed": client.seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    invalid = [
        {}, {"choices": []}, {"choices": [None]},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": 42}, "finish_reason": "stop"}]},
    ]
    for response in invalid:
        with patch.object(client, "request_json", return_value=response) as transport:
            try:
                client.complete("Контроль")
            except ValueError:
                transport.assert_called_once()
                continue
            raise AssertionError("Некорректный ответ локального сервера принят")
    with patch.object(client, "request_json",
                      side_effect=HTTPError(client.api_url, 503, "offline", {}, None)) as transport:
        try:
            client.complete("Контроль")
        except HTTPError:
            transport.assert_called_once()
        else:
            raise AssertionError("Ошибка локального HTTP скрыта")
    for host in ("127.0.0.1", "localhost", "[::1]", "host.docker.internal"):
        with patch.dict("os.environ", {"LOCAL_LLM_API_URL": f"http://{host}:8080/v1/chat/completions"}):
            LocalLLMClient()
    for host in ("example.com", "host.docker.internal.example.com", "192.168.1.2"):
        with patch.dict("os.environ", {"LOCAL_LLM_API_URL": f"http://{host}:8080/v1/chat/completions"}):
            try:
                LocalLLMClient()
            except ValueError:
                continue
            raise AssertionError("Удалённый endpoint принят локальным клиентом")
    return {"invalid_responses_rejected": len(invalid), "remote_hosts_rejected": 3}


def replay_seo() -> dict[str, int]:
    """Не терять реальные услуги и не генерировать их из поисковых хвостов."""
    from main import find_seo_spans, strip_seo_sections
    from draft_generation.local_llm.pipeline import DraftGenerationPipeline

    body = "Устанавливаю розетки.\nРаботаю в Казани."
    cases = [
        (body + "\n🔎 Также нас ищут как:\nшпаклёвка, обои, плитка", body),
        (body + "\nКлючевые слова для поиска:\nсантехника, штукатурка", body),
        (body + " Также ищут: шпаклёвка стен, укладка плитки", body),
        (body + "\r\nПоисковые запросы:\r\nшпаклёвка, обои", body),
        ("Наши услуги:\n• Штукатурка стен\n• Шпаклёвка стен\nРаботаем по договору.",
         "Наши услуги:\n• Штукатурка стен\n• Шпаклёвка стен\nРаботаем по договору."),
        ("Выполняю диагностику. Для поиска причины протечки использую прибор.",
         "Выполняю диагностику. Для поиска причины протечки использую прибор."),
        ("Также ищут: штукатурка, шпаклёвка, плитка", ""),
    ]
    for text, expected in cases:
        assert strip_seo_sections(text) == expected
        for span in find_seo_spans(text):
            assert text[span.start:span.end] == span.text
    keyword_tail = (
        "сантехника, водоснабжение, канализация, отопление, электрика, проводка, "
        "розетки, освещение, плитка, керамогранит, мозаика, обои, сантехника, "
        "водоснабжение, канализация, отопление, электрика, проводка, розетки, освещение"
    )
    assert strip_seo_sections(body + "\n" + keyword_tail) == body
    pipeline = DraftGenerationPipeline()
    with patch.object(pipeline.client, "complete") as complete:
        assert pipeline.run(PipelineInput(description=cases[-1][0], mcId=101), [102]) == []
        complete.assert_not_called()
    with patch.object(pipeline.client, "complete", return_value="null"):
        assert pipeline.run(PipelineInput(description=cases[0][0], mcId=101), [106]) == []
    return {"cleaning_cases": len(cases) + 1, "seo_only_drafts": 0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    started = perf_counter()
    report = {**metadata(), "scenario": "offline_contract"}
    report["checks"] = replay()
    report["local_checks"] = replay_local()
    report["local_transport_checks"] = replay_local_transport()
    report["seo_checks"] = replay_seo()
    report.update(status="complete", elapsed_seconds=perf_counter() - started)
    save_report(Path(__file__).with_name("results.json"), report)
    logger.info("sample_llm: %s", report)
