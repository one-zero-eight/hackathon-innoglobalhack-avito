# Общие API-клиенты

Новые клиенты находятся в `src/main.py`, используют стандартную библиотеку Python
и реализуют существующий `LLMClient.complete(prompt, **kwargs) -> str`.
Этот шаг добавляет транспорт; выбранные в `config.json` прежние модули пока
используют свои прежние клиенты. Подключение новых модулей — следующие изменения.

## Google Decision

`GoogleDecisionClient` отправляет один `POST` на
`https://generativelanguage.googleapis.com/v1beta/decisions`:
`model`, строковый `input`, словарь `questions`. Авторизация — `x-goog-api-key`.
Ответ возвращается JSON-строкой; соответствие вопросов и вероятностей
проверяется в сервисе выделения.

Настройки: `EXTRACTION_MODEL_ID`, `EXTRACTION_API_KEY`, `EXTRACTION_API_URL`.
Модель по умолчанию — `gemini-decision-1`. При отсутствии `EXTRACTION_MODEL_ID`
сохраняется поддержка `LLM_MODEL_ID`. При отсутствии отдельного ключа используются
`LLM_API_KEY`, затем `GOOGLE_API_KEY`.

## Генерация

`generation_client()` выбирает клиент через `GENERATION_PROVIDER`:

| Значение | Клиент | Запрос и ответ |
| --- | --- | --- |
| `google` (по умолчанию) | `GoogleGenerationClient` | GenerateContent `contents`; текст из `candidates[0].content.parts`, без thinking |
| `openai_compatible` | `OpenAICompatibleClient` | Chat Completions `model`, `messages`, `stream: false`; `choices[0].message.content` |

Google по умолчанию использует `gemma-4-31b-it`. Модель можно заменить через
`GENERATION_MODEL_ID`. Полный URL задаётся через `GENERATION_API_URL`; если
он пуст, Google-клиент подставляет URL GenerateContent для выбранной модели.
Ключ: `GENERATION_API_KEY`, затем `GOOGLE_API_KEY`, затем `LLM_API_KEY`.

OpenAI-compatible требует явных `GENERATION_MODEL_ID`, `GENERATION_API_KEY`
и полного `GENERATION_API_URL`, включая `/chat/completions`. Это независимые
настройки; ключ классификатора сюда автоматически не передаётся.
Пример URL Fugue из опубликованной документации:
`https://bestapi.fugue.pro/v1/chat/completions`. Оригинальный Jev не подключён.

Режим `GENERATION_RESPONSE_MODE`:

- `prompt` — не отправлять провайдеру параметры принудительного JSON;
  формат запрашивается промптом и проверяется сервисом локально.
- `json_object` — явно включить JSON mode API.
- `json_schema` — явно включить JSON Schema; обязательна передача
  `response_schema=...` в `complete()`.

По умолчанию используется `prompt`, чтобы не предполагать поддержку schema mode
любой моделью или шлюзом. Ошибка неподдерживаемого режима не вызывает молчаливое
переключение. API-клиент не подтверждает фактическую достоверность текста.
Google требует `finishReason=STOP`; Chat Completions — `finish_reason=stop`,
непустой строковый `content`, отсутствие refusal и tool calls.

## Ошибки и повторы

Параметры обоих клиентов имеют префиксы `EXTRACTION_` и `GENERATION_`:
`TIMEOUT` (60 секунд на попытку), `MAX_ATTEMPTS` (3, допустимо 1–5),
`RETRY_DELAY` (1 секунда). Задержка повторов растёт экспоненциально;
`Retry-After` учитывается в секундах и HTTP-date. Пауза ограничена 60 секундами.
Повторяются только HTTP 429/500/502/503/504 и сетевые ошибки. Это повторы
HTTP-вызова, а не повторная генерация при неверном тексте модели.

Redirect отклоняется, чтобы не пересылать ключ. Endpoint должен быть HTTP(S)
без credentials в URL, query и fragment. Отсутствие ключа обнаруживается до
сетевого вызова. Неверные настройки, HTML вместо JSON (включая HTTP 200),
некорректный JSON, объект `error`, незавершённая или заблокированная генерация
прерывают запрос явной ошибкой. Тела ответов и ключи в ошибки не включаются;
ключ исключён из repr и `model_dump()` настроек.

Офлайн-проверка без обращения к API:

```bash
uv run python src/draft_generation/benchmarks/api_clients/run.py --all-revisions
```

Если sandbox запрещает запись в обычный uv cache, можно добавить
`UV_CACHE_DIR=/tmp/codex-uv-cache` перед командой. Проверка не измеряет качество
или задержку модели; её результаты описаны в README бенчмарка.

## Ссылки на протоколы

- [Google Decisions API](https://aistudio.google.com/docs/decisions-api)
- [Google Gemma GenerateContent](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
- [Fugue Chat Completions](https://bestapi.fugue.pro/docs/en/api/chat-completions)
