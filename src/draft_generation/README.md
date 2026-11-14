# draft_generation

Шаг 2 пайплайна: идентификаторы из выделения услуг передаются напрямую в генерацию.
Под каждую целевую микрокатегорию создаётся черновик
объявления (`title` + `text`), написанный только по фактам исходного объявления.

- Интерфейс: `DraftGenerationPipeline.run(PipelineInput, target_mc_ids) -> list[dict]`
  (dict контракта кейса, валидация — в `main.PipelineOutput`)
- Доступны `sample_llm` и `local_llm`. Варианты добавляются в `<module>/pipeline.py`,
  промпты — в `<module>/prompts/`.
- Выбор: ключ `draft_generation` в корневом `config.json`
- `null` — модуль не выбран.
- Бенчмарки: `benchmarks/<name>/` с `run.py`, `README.md` и результатами.

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

## local_llm

Тот же промпт, обработка категорий и Pydantic-валидация, что у `sample_llm`,
с локальным OpenAI-compatible Chat Completions транспортом. Конфиг:
`"draft_generation": "local_llm"`. Ключ облачного API не используется.

`LOCAL_LLM_MODEL_ID` — обязательное имя/локальный путь модели;
`LOCAL_LLM_API_URL` по умолчанию
`http://127.0.0.1:8080/v1/chat/completions` (только loopback).
Параметры: `LOCAL_LLM_TEMPERATURE=0.2`, `LOCAL_LLM_MAX_TOKENS=2048`,
`LOCAL_LLM_SEED=43`, `LOCAL_LLM_TIMEOUT=180`.
`LOCAL_LLM_REVISION` фиксирует ревизию весов в отчётах сравнения.
Рантайм модели устанавливается отдельно от зависимостей проекта.

Передаётся `chat_template_kwargs.enable_thinking=false`. Сервер должен
поддерживать этот параметр. JSON ограничен промптом и локальной проверкой,
не constrained decoding; невалидный или обрезанный ответ — явная ошибка,
без исправлений и повторной генерации. Это отличие от Google JSON Schema
фиксируется при сравнении. Автоматические цитаты и метрики не гарантируют
фактическую достоверность каждого ответа.

### Локальный рантайм на Apple Silicon

Зависимости изолированы от основного `uv.lock`:

```bash
uv venv out/draft-model-runtime --python .venv/bin/python
uv pip install --python out/draft-model-runtime/bin/python "mlx-lm==0.31.0" "mlx==0.30.4" "transformers==5.3.0"
```

Для сравнения используются 4-битные веса:

| Модель | Репозиторий | Зафиксированная ревизия |
| --- | --- | --- |
| Qwen3.5-4B | `mlx-community/Qwen3.5-4B-4bit` | `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` |
| Qwen3.5-9B | `mlx-community/Qwen3.5-9B-4bit` | `8b2b98c00a6b4d291155e4890773ca8f769aee53` |
| Gemma 3 12B IT (text-only) | `mlx-community/gemma-3-text-12b-it-4bit` | `83eda451975103fff6f11e97bf75cd0f1c61af2c` |

Загрузить выбранный репозиторий через `huggingface_hub.snapshot_download`
с указанным `revision` и `local_dir=out/draft-model-weights/qwen35-4b`
(либо `qwen35-9b` / `gemma3-12b`). Интернет нужен для первоначальной загрузки,
не для последующей генерации из локального каталога.

```bash
out/draft-model-runtime/bin/mlx_lm.server \
  --model "$PWD/out/draft-model-weights/qwen35-4b" \
  --host 127.0.0.1 --port 18086 \
  --chat-template-args '{"enable_thinking":false}' \
  --max-tokens 2048 --prompt-cache-size 1 \
  --decode-concurrency 1 --prompt-concurrency 1
```

В другом терминале задать `LOCAL_LLM_API_URL=http://127.0.0.1:18086/v1/chat/completions`
и **тот же абсолютный путь** в `LOCAL_LLM_MODEL_ID`, что передан серверу.
Иначе MLX может попытаться загрузить другой экземпляр модели.
Для другой модели остановить текущий сервер и указать её локальный каталог.
MLX-сервер предупреждает об ограниченных security checks; наружу его не открываем.
