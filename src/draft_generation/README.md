# draft_generation

Шаг 2 пайплайна: идентификаторы из выделения услуг передаются напрямую в генерацию.
Под каждую целевую микрокатегорию создаётся черновик
объявления (`title` + `text`), написанный только по фактам исходного объявления.

- Интерфейс: `DraftGenerationPipeline.run(PipelineInput, target_mc_ids) -> list[dict]`
  (dict контракта кейса, валидация — в `main.PipelineOutput`)
- Доступен `sample_llm`. Варианты добавляются в `<module>/pipeline.py`,
  промпты — в `<module>/prompts/`.
- Выбор: ключ `draft_generation` в корневом `config.json`; в HTTP — query `generator`
  для отдельного запроса. Список и defaults: `GET /modules`, Swagger: `/docs`.
- `null` — модуль не выбран.
- Бенчмарки: `benchmarks/<name>/` с `run.py`, `README.md` и результатами.

Полный HTTP-пайплайн запускается через `uv run python -m src`; [описание API](../../docs/http-api.md).
Оба шага используют общий loader и runtime из main.py. Пустые цели пропускают загрузку
генератора. Runtime проверяет ровно один черновик на каждую полученную категорию,
не возвращая частичный результат при ошибке.

## sample_llm

Минимальный генератор для проверки полного пайплайна: один запрос LLM
на категорию, на входе название категории и исходное описание.
JSON-ответ с непустыми `title` и `text` валидируется через Pydantic.
`mcId` задаётся кодом, а не моделью. Пустой список категорий не вызывает API.

Выбор в `config.json`: `"draft_generation": "sample_llm"`.
В `.env` нужен общий `LLM_API_KEY` для Google Gemini.
По умолчанию используется модель `gemini-3.5-flash-lite`.
При необходимости задайте `SAMPLE_LLM_MODEL_ID`, `SAMPLE_LLM_API_URL`
и `SAMPLE_LLM_TIMEOUT` (секунды, по умолчанию 60).
При HTTP 429/500/502/503/504 и таймауте запрос повторяется:
`SAMPLE_LLM_MAX_ATTEMPTS` (4 попытки), `SAMPLE_LLM_RETRY_DELAY` (10 секунд).
Остальные ошибки не скрываются и прерывают прогон.
Endpoint должен поддерживать Google GenerateContent с JSON mode;
`{model}` в URL заменяется именем модели.
JSON-схема `title`/`text` передаётся в API из той же модели Pydantic,
которой валидируется ответ.
`LLM_MODEL_ID` не используется генератором: модель `gemini-decision-1`
остаётся настройкой классификатора.
Генератор не проверяет фактическую достоверность текста автоматически —
это простой пробный модуль, а не гарантия отсутствия галлюцинаций.

Офлайн-проверка без расходов на API:

```bash
uv run python src/draft_generation/benchmarks/sample_llm/run.py
```
