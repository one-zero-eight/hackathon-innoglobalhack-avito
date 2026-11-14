# Проверка rebuild

Это исторический срез до появления HTTP-сервера и удаления CLI.
Команды текущей проверки — в [manual-testing.md](manual-testing.md).

Срез исполняемого кода: `b08f4e4`. Он совпадает с проверенной рабочей копией
на базе `2c65c3f`; отпечаток кода и промптов:
`20eae115e95e6ff2162a08f927a787957d5f3411c5d1bb427bcbe0bcc107088e`.
Даты фактических запусков и сравнения прежних версий находятся в JSON-отчётах.

## Подтверждено офлайн

| Сценарий | Результат | Отчёт |
| --- | --- | --- |
| Общие API-клиенты | 88/88 | [api_clients](../src/draft_generation/benchmarks/api_clients/results.json) |
| Выделение категорий и метрики | 49/49 | [extraction contract](../src/subcategory_extraction/benchmarks/contract/results.json) |
| Генерация через обоих провайдеров и CLI | 64/64 | [generation contract](../src/draft_generation/benchmarks/contract/results.json) |
| Оценщик | 208 записей, 24 пересечения, 4 проверки checkpoint | [evaluator](../src/draft_generation/benchmarks/full_pipeline_evaluation/checks-results.json) |
| Перенос чтения данных | Те же 100 validation group_id; 4854 уникальных нормализованных описания | Зафиксированные группы в [quality report](../src/subcategory_extraction/benchmarks/prompt_comparison/rebuild-results.json) |

Клиенты проверены на протоколах, авторизации, разделении настроек, ограниченных
повторах, HTTP 200 HTML, ошибках JSON, отказах и незавершённых ответах.
Сервисы проверены на исходной категории, повторах, пустых данных, неверных
типах и отсутствии частичного результата. Для обоих провайдеров настоящий
CLI проверен из другого cwd с контролируемыми ответами API; ошибка генерации
не перезаписывает прежний выходной файл.

Дополнительно настоящий subprocess CLI без подмены клиентов вернул
`{"drafts": []}` для пустого описания из другого cwd без API-ключей.
Python-код компилируется, `git diff --check` и проверка локальных Markdown-ссылок
пройдены. Стоимость локальных проверок заполнена вручную: бесплатно.

Сценарии с `--all-revisions` выполняются на версиях соответствующего кода
в текущей истории Git. Отсутствующие модули и несовместимые старые интерфейсы
помечены unavailable. Офлайн-проверки не являются оценкой модели.

## Live-проверки, которые не завершены

В этом checkout нет `.env` или экспортированных API-ключей. Поэтому:

- [Сравнение промптов](../src/subcategory_extraction/benchmarks/prompt_comparison/rebuild-results.json): failed до API, нового F1 и лучшего промпта нет.
- [Задержка выделения](../src/subcategory_extraction/benchmarks/response_time/extraction-results.json): failed до API, завершённых вызовов 0.
- [Генерация](../src/draft_generation/benchmarks/response_time/generation-results.json): failed до API, текст и фактическая точность не оценены.
- [Полный пайплайн](../src/draft_generation/benchmarks/full_pipeline_evaluation/pipeline-results.json): failed на первой строке без ключа, итогового F1 нет.

Старые показатели не подставляются вместо этих результатов. Новая Gemma не
объявляется доступной по данному аккаунту на основании офлайн-проверки.
Стоимость и задержка настоящей генерации неизвестны; цена не вычислялась.

## Продолжение после настройки ключа

Скопируйте `.env.example` в `.env` и заполните локальный `LLM_API_KEY` Google.
Его достаточно для обоих шагов по умолчанию. Для разных аккаунтов используются
`EXTRACTION_API_KEY` и `GENERATION_API_KEY`; ключи не передаются в чат и Git.

```bash
uv run python -m src
```

После запуска сервера в другом терминале:

```bash
curl http://127.0.0.1:8000/drafts -H 'Content-Type: application/json' --data-binary @data/input.json
uv run python src/subcategory_extraction/benchmarks/prompt_comparison/run.py
uv run python src/subcategory_extraction/benchmarks/response_time/run.py
uv run python src/draft_generation/benchmarks/response_time/run.py
```

Последняя команда сохраняет исходники и тексты черновиков для ручной оценки:
проверьте каждую услугу, цену и условие по источнику, самостоятельность
текста и релевантность целевой категории. JSON-схема этого не доказывает.
Режимы API JSON/schema включаются только при поддержке выбранной модели.

После фиксации модели, промптов и порога:

```bash
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --live --config run.config.json
```

Не меняйте промпт по итоговому test. Неполный прогон можно продолжить
через `--resume`, если настройки, данные и код не изменились. После реальных
вызовов обновите README сценариев фактическими метриками, временем и стоимостью.
Полный набор команд и настройки — в [README](../README.md).
