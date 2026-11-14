# subcategory_extraction

Выделение дополнительных микрокатегорий по `description` и исходной `mcId`.
Контракт: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`.
Выбор в `config.json`: `"subcategory_extraction": "google_decision"`.

## Google Decision

Переписанная реализация использует общий `main.GoogleDecisionClient`.
Один вызов Google Decisions API получает описание и независимые вопросы `noul`
по всем категориям из `main.MICROCATEGORIES`, кроме исходной. Возвращаются
уникальные mcId в числовом порядке с вероятностью не ниже порога.
Пустое или пробельное описание возвращает `[]` без вызова API.

Сервис строго проверяет оболочку `answers`, точное совпадение её ключей с
вопросами, `type=noul`, числовую конечную вероятность от 0 до 1 и confidence,
если он присутствует. Строки, bool и вероятности вне диапазона не принимаются.
Метаданные API игнорируются. Ошибка не превращается в пустой набор категорий.

Промпты находятся в `google_decision/prompts/`. `category_guidance.json`
содержит границы категорий по официальному справочнику; при загрузке набор
идентификаторов сверяется с общим контрактом. Названия категорий берутся
из `main.MICROCATEGORIES`.

`guided` учитывает явно предложенные этапы комплексного ремонта без требования
отдельного заказа. Он исключает отрицания, чужие услуги, поиск работника,
материалы без предложения монтажа и вывод конкретных работ только из фразы
«ремонт под ключ». Описание передаётся как данные, инструкции внутри него
не должны менять задачу модели. `baseline` — более короткая альтернатива.
Поведение модели на этих смысловых случаях требует реального API-бенчмарка;
офлайн-подмена ответа этого не доказывает.

## Настройки

В `.env`:

```dotenv
LLM_API_KEY=your-google-key
EXTRACTION_MODEL_ID=gemini-decision-1
EXTRACTION_PROMPT=guided
EXTRACTION_THRESHOLD=0.5
```

`guided` и порог 0.5 — начальные настройки, а не результат новой оценки.
Промпт можно менять на `baseline`, порог должен быть конечным числом от 0 до 1.
`EXTRACTION_API_KEY` переопределяет общий ключ; при его отсутствии используются
`LLM_API_KEY`, затем `GOOGLE_API_KEY`. Модель при отсутствии отдельной настройки
берётся из `LLM_MODEL_ID`, затем используется `gemini-decision-1`.

`EXTRACTION_API_URL`, `EXTRACTION_TIMEOUT`, `EXTRACTION_MAX_ATTEMPTS`,
`EXTRACTION_RETRY_DELAY` описаны в [общих API-клиентах](../../docs/api-clients.md).
Старые `GOOGLE_DECISION_PROMPT`, `GOOGLE_DECISION_THRESHOLD` и
`GOOGLE_DECISION_ENDPOINT` больше не используются: перенесите значения
в `EXTRACTION_PROMPT`, `EXTRACTION_THRESHOLD`, `EXTRACTION_API_URL`.

Сетевые вызовы происходят только в `run()` / `probabilities()`, не при импорте
или создании экземпляра. Неподдерживаемый API, HTML, ошибки авторизации и
исчерпание ограниченных повторов не вызывают переключение на другую модель.

## Бенчмарки

```bash
# Офлайн: контракт сервиса и сравнение с прежними версиями, без ключа.
uv run python src/subcategory_extraction/benchmarks/contract/run.py --all-revisions

# Настоящие вызовы Google: оба промпта на одних 100 validation-входах.
uv run python src/subcategory_extraction/benchmarks/prompt_comparison/run.py

# Настоящее время run() на 10 одинаковых входах.
uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module google_decision --report src/subcategory_extraction/benchmarks/response_time/rebuild-results.json
```

Сценарии и фактические результаты:
[контракт](benchmarks/contract/README.md),
[качество и промпты](benchmarks/prompt_comparison/README.md),
[задержка](benchmarks/response_time/README.md).
Промпты выбираются по micro-F1 на validation, а итоговый test используется
после фиксации настроек. Исходная категория официального train всегда 101;
качество для остальных исходных категорий отдельно не подтверждено.

## Оставшийся прежний вариант

`tfidf` временно доступен для воспроизводимости существующих сравнений до
отдельного завершения очистки. Он не выбран по умолчанию; обучение и прежние
результаты описаны в [tfidf_quality](benchmarks/tfidf_quality/README.md).
Исходный Jev через Fugue отложен: рабочий маршрут его API не подтверждён.
