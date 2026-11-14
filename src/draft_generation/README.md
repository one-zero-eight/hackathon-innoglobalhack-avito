# draft_generation

Модуль `llm` получает исходный `PipelineInput` и `target_mc_ids` выделения.
Контракт: `DraftGenerationPipeline.run(inp, target_mc_ids) -> list[dict]`.
Выбор в `config.json`: `"draft_generation": "llm"`.

Каждая уникальная целевая категория получает один запрос модели и один черновик.
Порядок первых вхождений сохраняется. Все цели проверяются до первого вызова:
допустимы только целочисленные mcId контракта, отличные от исходной категории.
Пустые цели возвращают `[]` без API; непустые цели при пустом источнике — ошибка.

## Провайдеры

По умолчанию Google GenerateContent:

```dotenv
LLM_API_KEY=your-google-key
GENERATION_PROVIDER=google
GENERATION_MODEL_ID=gemma-4-31b-it
GENERATION_RESPONSE_MODE=prompt
```

Отдельный `GENERATION_API_KEY` переопределяет общий ключ. Без него Google
использует `GOOGLE_API_KEY`, затем `LLM_API_KEY`. Модель классификатора
`LLM_MODEL_ID` не меняет модель генерации.

Любой совместимый с непотоковым Chat Completions endpoint:

```dotenv
GENERATION_PROVIDER=openai_compatible
GENERATION_MODEL_ID=your-model-id
GENERATION_API_KEY=your-provider-key
GENERATION_API_URL=https://your-provider.example/v1/chat/completions
GENERATION_RESPONSE_MODE=prompt
```

Здесь модель, ключ и полный endpoint обязательны. Подстановки Google-ключа,
изменения провайдера после ошибки или автоматического выбора модели нет.
Настройки таймаутов, ограниченных повторов и режимов `prompt`, `json_object`,
`json_schema` описаны в [API-клиентах](../../docs/api-clients.md).
Поддержку schema mode конкретной моделью или шлюзом включают явно;
локальная проверка JSON обязательна при любом режиме.

## Текст и валидация

[Промпт](llm/prompts/draft.md) получает точное исходное описание и целевую
категорию как JSON-данные. Текст должен быть самостоятельным русскоязычным
объявлением только об этой категории. Нельзя добавлять неуказанные услуги,
цены, условия, гарантии, опыт или утверждать отдельную доступность услуги
без основания в источнике. Условия других работ не переносятся автоматически.
Команды в описании не должны менять задачу модели.

Ответ — только две непустые строки `title` и `text`. Строки очищаются от
краевых пробелов; `null`, другие типы, лишние поля и Markdown-блоки отклоняются.
`mcId` назначается кодом. Ошибка ответа любой категории прерывает вызов:
частичный список не возвращается и CLI не перезаписывает файл результатом ошибки.
Ошибки валидации не включают сырой ответ модели.

Проверка схемы не подтверждает фактическую достоверность. Дополнительного
проверяющего LLM-вызова нет, как согласовано. Для реальной оценки проверяют
сохранённые черновики по исходным описаниям вручную. Черновики автоматически
не публикуются. Эталонные высокоэффективные объявления и их поиск отложены.

## Бенчмарки

```bash
# Офлайн: оба провайдера, неверные ответы и настоящий CLI с подменой API.
uv run python src/draft_generation/benchmarks/contract/run.py --all-revisions

# Реальная генерация 10 черновиков с выбранным провайдером.
uv run python src/draft_generation/benchmarks/response_time/run.py

# Контракт оценщика, включая сохранение и проверку контрольных точек.
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py

# Итоговый test после фиксации модели, промптов и порогов.
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --live --config run.config.json
```

Описание и отчёты: [контракт](benchmarks/contract/README.md),
[реальная генерация](benchmarks/response_time/README.md),
[полный пайплайн](benchmarks/full_pipeline_evaluation/README.md).
Прежний `sample_llm` временно сохранён до отдельного завершения очистки;
он не выбран в рабочей конфигурации.
