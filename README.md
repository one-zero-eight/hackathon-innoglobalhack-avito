# Услуги из объявлений Авито → черновики

По `description` и текущей `mcId` находим предлагаемые услуги из других
микрокатегорий «Ремонт и отделка» и создаём по одному черновику на категорию.

## Запуск

```bash
uv sync
cp .env.example .env
# Заполните LLM_API_KEY своим ключом Google в локальном .env.
uv run python src/main.py --input data/input.json --output out/output.json
```

`config.json` и `.env` читаются относительно проекта, независимо от cwd.
Пути `--input` и `--output` задаёт вызывающая сторона.

Контракт входа:

```json
{ "description": "Делаю ремонт квартир и устанавливаю раковины.", "mcId": 101 }
```

Контракт выхода:

```json
{ "drafts": [{ "mcId": 104, "title": "...", "text": "..." }] }
```

Разрешены только категории 101–106. Исходная категория исключается.
При отсутствии дополнительных услуг возвращается `{"drafts": []}`.
Ничего не публикуется автоматически. Черновики предназначены для проверки
исполнителем; формат ответа и таблица категорий зафиксированы кейсом.

## Реализации и настройки

```json
{
  "subcategory_extraction": "google_decision",
  "draft_generation": "llm"
}
```

Значения config — имена модулей, не моделей. Оба ключа обязательны;
`null`, неизвестный модуль и неверная конфигурация приводят к явной ошибке.
Интерфейсы сервисов и `PipelineConfig` сохранены; выбор можно передать
программно в `load_pipeline()` / `run_pipeline()`.

- [Выделение](src/subcategory_extraction/README.md): Google Decision 1,
  один запрос с независимыми вопросами `noul` по остальным категориям.
  `EXTRACTION_PROMPT=guided`, `EXTRACTION_THRESHOLD=0.5` — начальные настройки.
- [Генерация](src/draft_generation/README.md): Google Gemma 4 31B по умолчанию,
  один вызов на уникальную целевую категорию, строгие `title` и `text`.
  Опционально `GENERATION_PROVIDER=openai_compatible` с явными моделью,
  ключом и полным URL Chat Completions. Референсные объявления отложены.
- [Общие API-клиенты](docs/api-clients.md): независимые настройки шагов,
  таймауты и ограниченные повторы, явные ошибки на HTML/неверный JSON,
  отсутствие автоматического переключения провайдеров.

Текст генерируется по фактам источника. `mcId` назначает код. Схема не
подтверждает фактическую достоверность; дополнительного проверяющего LLM нет.
API schema mode включается явно; по умолчанию формат задаёт промпт,
локальная проверка обязательна. Ошибка не превращается в пустой или частичный
ответ и не перезаписывает файл результатом неуспешной генерации.

Прежние `tfidf` и `sample_llm` удалены. Их результаты помечены историческими
в соответствующих сценариях, а исходные реализации доступны на `2c65c3f`.
Оригинальный Jev/Fugue отложен; экстрактор использует только Google.

## Бенчмарки

Каждый сценарий: `src/<service>/benchmarks/<name>/run.py`, `README.md`,
JSON-отчёт с датой, commit и отпечатком кода/промптов. Исторические и новые
результаты, разные выборки, офлайн-проверки и live-вызовы разделены.
Стоимость заполняется вручную; скрипты её не вычисляют.

| Проверка | Команда / описание |
| --- | --- |
| Общие клиенты, офлайн | [api_clients](src/draft_generation/benchmarks/api_clients/README.md) |
| Выделение, офлайн | [contract](src/subcategory_extraction/benchmarks/contract/README.md) |
| Генерация и CLI, офлайн | [contract](src/draft_generation/benchmarks/contract/README.md) |
| Выделение, качество и промпты | [prompt_comparison](src/subcategory_extraction/benchmarks/prompt_comparison/README.md) |
| Выделение, задержка | [response_time](src/subcategory_extraction/benchmarks/response_time/README.md) |
| Генерация, задержка и тексты для ручной оценки | [response_time](src/draft_generation/benchmarks/response_time/README.md) |
| Полный пайплайн и оценщик | [full_pipeline_evaluation](src/draft_generation/benchmarks/full_pipeline_evaluation/README.md) |

```bash
# Без API: контракты и сравнение Git-версий.
uv run python src/draft_generation/benchmarks/api_clients/run.py --all-revisions
uv run python src/subcategory_extraction/benchmarks/contract/run.py --all-revisions
uv run python src/draft_generation/benchmarks/contract/run.py --all-revisions
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py

# Настоящие вызовы: нужен локально настроенный ключ.
uv run python src/subcategory_extraction/benchmarks/prompt_comparison/run.py
uv run python src/subcategory_extraction/benchmarks/response_time/run.py
uv run python src/draft_generation/benchmarks/response_time/run.py
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --live --config run.config.json
```

Новый Google-пайплайн проверен офлайн. Live-прогоны в текущем окружении
не завершены из-за отсутствия ключей; новой оценки F1/задержки/качества текста
нет. `guided` не объявляется лучшим на основании исторических результатов.

Общие выборки формируются в `benchmark_support.py` по официальному train:
нормализованные точные дубликаты объединяются; идентификаторы и хеш данных
сохраняются. Близкие неидентичные шаблоны отдельно не группируются.
Новая модель здесь не обучается; результаты сравнения промптов отражают
согласие с автоматическими метками.

Итоговые 208 записей test используются после фиксации настроек. Оценщик
сверяет названия категорий, проверяет нормализованное пересечение с train
и считает sklearn micro-F1 по mcId из итоговых drafts; баллы — 60 × micro-F1.
Найденные 24 пересечения отражаются в отчёте, строки не удаляются.
`--resume` принимает только совпадающий отпечаток данных, кода и настроек,
проверяет сохранённые предсказания и не вызывает модель для готовых строк.
Неуспешный прогон не получает итогового F1. Новый отчёт полного пайплайна —
`src/draft_generation/benchmarks/full_pipeline_evaluation/pipeline-results.json`.

## Структура

Весь исполняемый код — в `src/`. `main.py` содержит общий контракт, таблицу
категорий, API-клиенты и оркестрацию; бизнес-логика — в двух сервисах.
Промпты и ресурсы лежат внутри вариантов. Оценщик — `src/run.py`, общая
поддержка бенчмарков — `src/benchmark_support.py`. Аудит и переразметка
данных находятся отдельно в `src/dataset_annotation/` и в этот rebuild не входят.

Конвенции разработки — в [AGENTS.md](AGENTS.md). Секреты, рабочие артефакты
и локальный бэклог не коммитятся. Новые изменения идут через feature-ветку и PR.
