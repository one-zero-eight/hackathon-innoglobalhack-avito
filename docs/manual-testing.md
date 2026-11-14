# Ручная проверка всего решения

Рабочая ветка — `zolotarev/feature/avito-pipeline`.
Сервер запускается через `python -m src`; файловый CLI удалён.
Ключи хранятся только в локальном `.env`.

## 1. Подготовка

Из корня проекта:

```bash
git branch --show-current
git status --short
uv sync --locked
```

Ожидаемая ветка — `zolotarev/feature/avito-pipeline`. Незакоммиченные изменения проверяйте через `git status`.

Уже заполненный `.env` сохраните. Если создаёте новый checkout, скопируйте
`.env.example` в `.env` и задайте ключи локально. Для текущего Google-режима:

```dotenv
EXTRACTION_MODEL_ID=gemini-decision-1
EXTRACTION_API_KEY=<ключ Google>
EXTRACTION_PROMPT=guided
EXTRACTION_THRESHOLD=0.5
GENERATION_PROVIDER=google
GENERATION_MODEL_ID=gemma-4-31b-it
GENERATION_API_KEY=<ключ Google>
GENERATION_RESPONSE_MODE=prompt
```

Вместо отдельных ключей можно использовать общий `LLM_API_KEY`.
`config.json` должен выбирать модули, а не модели:

```json
{
  "subcategory_extraction": "google_decision",
  "draft_generation": "llm"
}
```

После изменения `.env` или `config.json` перезапустите HTTP-сервер.

## 2. Офлайн-проверки: без запросов к моделям

```bash
uv run python src/draft_generation/benchmarks/api_clients/run.py --all-revisions
uv run python src/subcategory_extraction/benchmarks/contract/run.py --all-revisions
uv run python src/draft_generation/benchmarks/contract/run.py --all-revisions
uv run python src/draft_generation/benchmarks/http_api/run.py
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --report src/draft_generation/benchmarks/full_pipeline_evaluation/module-server-checks-results.json
```

Ожидается успешный exit code и текущие результаты:

| Проверка | Ожидаемый результат |
| --- | --- |
| API-клиенты | 88/88 |
| Выделение и метрики | 49/49 |
| Генерация и оркестрация | 64/64 |
| HTTP API и модульный запуск | 22/22 |
| Оценщик | 208 записей; 24 нормализованных пересечения с train; 5 проверок checkpoint |
| Формула PDF | 1007 контрольных случаев для двух scorer, все совпадают |

В сравнении Git-версий старые реализации могут иметь failed/unavailable.
Итоговый exit code относится к текущему коду. В частности, старый множитель
60 не проходит новую проверку запрошенного PDF; это не изменение сырого F1.

В `module-server-checks-results.json` проверьте `scoring_audit.status=complete`,
`independent_cases=1007`, `classification_max_points=5`.
Пример PDF: эталон `{102,104}`, ответ `{102,105}` → TP=1, FP=1, FN=1,
micro-F1=0.5, баллы=2.5/5. Проверяется именно глобальный micro-F1:

```text
2 × сумма(TP) / (2 × сумма(TP) + сумма(FP) + сумма(FN))
```

Предсказания берутся из уникальных `mcId` в drafts. Исходная категория,
ошибочно попавшая в ответ, считается FP. Для полностью пустых наборов
используется sklearn `zero_division=0`. Шкала 5 взята из первой страницы
`docs/Критерии авито.pdf`; Markdown-редакция критериев содержит другую шкалу.

Бенчмарки записывают JSON рядом со своими run.py. После новых запусков
таблицы README обновляются вручную фактическими результатами и стоимостью.
Длительность офлайн-проверки не является временем ответа модели.

## 3. Запуск сервера

Из корня проекта в отдельном терминале:

```bash
uv run python -m src
```

Сервер слушает `127.0.0.1:8000`. Для другого порта:

```bash
UVICORN_PORT=8001 uv run python -m src
```

`UVICORN_HOST` и `UVICORN_PORT` можно задать в `.env`; окружение имеет
приоритет. Используйте один экземпляр на порту. Если старый сервер ещё
работает, остановите его перед проверкой новой точки входа. Остановка — Ctrl+C.

## 4. HTTP без внешних вызовов

Откройте [Swagger UI](http://127.0.0.1:8000/docs) и выполните:

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/openapi.json

curl -i http://127.0.0.1:8000/drafts \
  -H 'Content-Type: application/json' \
  -d '{"description":"","mcId":101}'

curl -i http://127.0.0.1:8000/drafts \
  -H 'Content-Type: application/json' \
  -d '{"description":"x","mcId":999}'

curl -i http://127.0.0.1:8000/drafts \
  -H 'Content-Type: application/json' \
  -d '{"description":"x"}'

curl -i http://127.0.0.1:8000/drafts \
  -H 'Content-Type: application/json' \
  -d '{'

curl -i http://127.0.0.1:8000/drafts
```

Ожидаемые результаты по порядку:

1. Health: 200 и `{"status":"ok"}`.
2. OpenAPI: 200, маршруты `/drafts`, `/health`; в схемах есть `mcId`.
3. Пустой источник: 200 и `{"drafts":[]}`.
4. Категория 999: 422.
5. Отсутствующая mcId: 422.
6. Сломанный JSON: 422.
7. GET вместо POST: 405.

Health проверяет приложение, а не доступность Google. Эти запросы к моделям
не обращаются и не подтверждают качество извлечения или генерации.

## 5. Реальный запрос к моделям

Следующий запрос использует модели и их квоты:

```bash
curl -i http://127.0.0.1:8000/drafts \
  -H 'Content-Type: application/json' \
  --data-binary @data/input.json
```

Для предоставленного примера ожидаемая дополнительная категория — 104.
Проверьте, что исходная 101 исключена, нет повторов mcId, title/text непустые,
текст посвящён сантехнике и все услуги/условия подтверждаются источником.
Заголовок и формулировки могут отличаться между вызовами модели.

При последней проверке Google перечислял Gemma 4 как доступную модель,
но генерация возвращала HTTP 500. В таком случае HTTP-адаптер вернёт 502 с полем detail. Это не успешный результат и не F1=0.
Строгая ошибка на неверный JSON модели также отображается как 502.

## 6. Качество и задержка отдельных шагов

Для оценки текущего guided без подбора на итоговом test:

```bash
uv run python src/subcategory_extraction/benchmarks/prompt_comparison/run.py --prompts guided --size 100
uv run python src/subcategory_extraction/benchmarks/response_time/run.py
uv run python src/draft_generation/benchmarks/response_time/run.py
```

Первые два сценария используют только extractor. Последний создаёт 10
черновиков и сохраняет исходники/тексты для ручной фактической проверки.
Среднее, медиана и p95 считаются только для завершённых реальных run();
ожидания квоты и сетевые повторы входят в время.

Если исследуете baseline/guided, сравнивайте их на validation, затем
зафиксируйте настройку. Итоговый test не используется для подбора промпта
или порога. Стоимость API записывается вручную по фактическому тарифу.

## 7. Итоговый F1

Основной сценарий всего решения, включая генерацию:

```bash
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --live --config run.config.json --report src/draft_generation/benchmarks/full_pipeline_evaluation/manual-results.json
```

В конфиге — официальный test из 208 записей, `limit=null`, исходная категория
101. При успешном завершении в отчёте должны быть `status=complete`,
208 элементов `results` и `metrics` с micro_f1, classification_points, TP/FP/FN.
Баллы должны равняться `5 × micro_f1`. Пересечения с train отражаются в
`overlap`: для исходных файлов найдено 24/208; строки не удаляются.

После временного сбоя при неизменных данных, коде и настройках:

```bash
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --live --resume --config run.config.json --report src/draft_generation/benchmarks/full_pipeline_evaluation/manual-results.json
```

При failed/прерывании итоговой метрики нет. Сохранённые старые F1 не
подставляются вместо незавершённого текущего прогона.

Чтобы отдельно измерить **только текущую классификацию**, без Gemma:

```bash
EXTRACTION_RETRY_DELAY=30 EXTRACTION_MAX_ATTEMPTS=5 \
  uv run python src/subcategory_extraction/benchmarks/final_classification/run.py --report src/subcategory_extraction/benchmarks/final_classification/manual-results.json
```

Эти transport-настройки увеличивают ожидание квоты, не меняя модель,
промпт и порог. Повтор с `--resume` требует тех же значений и того же пути
отчёта. Успешный результат: `status=complete`, `completed_rows=208`, `metrics`.
Тексты при этом null, `generator_executed=false`; это classification-only F1,
а не подтверждение работоспособности полного пайплайна или качества текстов.

Прежний classification-only запуск прерван на 78/208 по просьбе пользователя.
Его исходная контрольная точка относится к коду до добавления API; после
изменений кода, включая удаление CLI, используйте новый запуск с отдельным файлом отчёта, как выше.

## 8. Проверка OpenAI-compatible генерации

В `.env` задайте параметры своего провайдера:

```dotenv
GENERATION_PROVIDER=openai_compatible
GENERATION_MODEL_ID=<идентификатор модели провайдера>
GENERATION_API_KEY=<ключ этого провайдера>
GENERATION_API_URL=https://your-provider.example/v1/chat/completions
GENERATION_RESPONSE_MODE=prompt
```

Укажите полный endpoint, включая `/chat/completions`, и перезапустите сервер.
Повторите реальный HTTP-запрос из раздела 5. Экстрактор остаётся Google
Decision 1. При неподдерживаемой модели, авторизации или невалидном JSON
ожидается явная ошибка; автоматического переключения провайдеров нет.

Общие сведения: [HTTP API](http-api.md), [настройки клиентов](api-clients.md),
[архитектура](architecture.md). Изменения и новые отчёты остаются локальными
до вашего явного коммита; push в рамках этой работы не выполняется.
