# Qwen decision-model-preview

Модуль `qwen_decision`, сервис `subcategory_extraction`.
Интерфейс: `SubcategoryExtractionPipeline(BasePipeline)`;
`run(PipelineInput) -> list[int]`. Исходная категория исключается.

Использует нативный [Qwen System One API](https://platform.qianwenai.com/docs/api-reference/decision-model-api)
на [Token Plan endpoint](https://platform.qianwenai.com/docs/token-plan/best-practices/decision-model):
`https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1/systemone`.
Chat Completions не вызывается. Для каждой дополнительной категории задаётся
независимый вопрос `noul`, ответ — P(yes). Это многометочная классификация,
не выбор одной категории из списка.

## Настройка

В локальной `.env` заполнить `QIANWEN_TOKEN_PLAN_API_KEY`.
Ключ не передавать в аргументах команд и не коммитить.
Для выбора изменить `subcategory_extraction` в корневом `config.json`
на `qwen_decision`. Генератор черновиков выбирается независимо.

- `QWEN_DECISION_MODEL`: `decision-model-preview`.
- `QWEN_DECISION_BASE_URL`: указанный выше URL **без** `/systemone`.
- `QWEN_DECISION_TIMEOUT`: 60 секунд для сетевых операций, не общий deadline вызова.
- `QWEN_DECISION_MAX_ATTEMPTS`: до 3 попыток при временных сетевых ошибках,
  HTTP 429/500/502/503/504; задержка 1, затем 2 секунды. Попытки и ожидание
  входят во время настоящего `run()`. Ошибки авторизации не повторяются.
- `QWEN_DECISION_PROMPT`: `auto` по умолчанию, либо `guided` / `boundaries`.
  `auto` требует артефакт калибровки; явные варианты используют порог 0.5.
- `QWEN_DECISION_CALIBRATION`: по умолчанию `artifacts/calibration.json`
  относительно `pipeline.py`. Проверяются модель, endpoint и SHA-256 промпта.
- `QWEN_DECISION_CACHE_DIR`: по умолчанию корневой `out/qwen-decision-cache`.
- `QWEN_DECISION_CACHE=0`: запрет чтения и записи кэша для замера API-времени.

Кэш зависит от полного endpoint, модели, текста и вопросов, но не от ключа.
Хранятся только ответы с точной моделью, полным набором вопросов,
правильными типами и конечными вероятностями в [0, 1]. HTTP-ошибки,
невалидные ответы и недоступность сети не заменяются локальными ответами.
Usage передаётся общему `LLMClient` и записывается в отчёты.

## Воспроизведение выбора промпта и порогов

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/decisions/qwen_decision/train.py --fit-size 240 --control-size 120 --workers 4
```

Только `docs/improved_dataset/train.jsonl`. Нормализованные точные и близкие
дубликаты объединяются в компоненты по char_wb TF-IDF (3–5 символов,
cosine ≥0.85). Используется общий существующий алгоритм группировки.
Детерминированно выбираются представители 240 групп для подбора и 120 других
групп для контроля, seed=20261114. Это эвристика близости, а не полный
семантический аудит дубликатов.

Два заранее написанных промпта: `guided`, `boundaries`. Общий существующий
алгоритм сравнивает фиксированный 0.5, общий и покатегорийные пороги со
стягиванием к 0.5 по трёхчастной групповой cross-validation. Промпт выбирается
по pooled crossfit micro-F1. Выбранное решение записывается в
`artifacts/calibration.json` **до** проверки control; control не меняет выбор.
Категория 101 отсутствует среди дополнительных меток train, её порог остаётся
необученным 0.5. Модель провайдера не дообучается, настраиваются промпт и пороги.

Ответы и происхождение подбора сохраняются в существующем сценарии
`../benchmarks/improved_dataset/qwen-training-results.json`; API-кэш локальный.
Артефакт калибровки входит в Git. Повтор команды переиспользует ответы из кэша;
без кэша потребуется действующий API и новая квота. Preview-модель провайдера
не имеет закреплённых публичных весов: её обновление может изменить результат.

## Общие проверки

После фиксации калибровки:

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods decisions.qwen_decision --workers 4 --append
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods decisions.qwen_decision --workers 4 --append
QWEN_DECISION_CACHE=0 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module decisions.qwen_decision --size 10 --append
```

Сценарии F1 сохраняют фактические вероятности, usage, попадания в кэш,
модель, промпт и пороги. `fixed_threshold_metrics` — пересчёт **тех же**
ответов с порогом 0.5, не новый прогон модели. `--append` сохраняет предыдущие
результаты других модулей с их исходной датой и отпечатком кода.
Старые `recorded-results.json` не изменяются.

Результаты и ограничения двух разных test-наборов — в README соответствующих
существующих сценариев. Исправленный test диагностический, с предварительной
разметкой, не независимый ручной эталон. Стоимость заносится вручную в
`../benchmarks/cpu_costs/README.md`; тариф Token Plan нельзя выводить из
публичной цены другого endpoint.

## Результат выбора

2026-11-14, база `45ff38d`, 4851 строка исправленного train объединена
в 2824 группы. Выбран `boundaries` и покатегорийные пороги:
101 — 0.5 (не обучен), 102 — 0.22, 103 — 0.64,
104 — 0.55, 105 — 0.5, 106 — 0.66.

| Часть train | N групп | Фиксированный 0.5 | После подбора |
| --- | ---: | ---: | ---: |
| guided, threshold-only crossfit | 240 | 0.725236 | 0.734055 |
| boundaries, threshold-only crossfit | 240 | 0.748971 | 0.770107 |
| boundaries, control без участия в выборе | 120 | 0.725664 | 0.761905 |

На исходном test выбранный вариант получил micro-F1 **0.629674** (208 строк),
на исправленном диагностическом test — **0.648980** (119 строк).
Свежий learned_keywords дал соответственно 0.745358 / 0.800948;
проверенные исторические ответы Jev — 0.846154 / 0.812665.
Калибровка помогла на control и диагностическом test, но ухудшила исходный test
относительно фиксированного 0.5. Это перенос на другой состав объявлений,
не доказательство универсального улучшения. После оценки test параметры не менялись.

По этим результатам глобальный `config.json` сохраняет прежний выбор
`semantic_encoder_f1`. Qwen можно выбрать вручную для воспроизведения.
Для непустых черновиков существующего `sample_llm` отдельно требуется
`LLM_API_KEY`; `QIANWEN_TOKEN_PLAN_API_KEY` используется только для решений.

Анализ вариантов и выбор промпта: [PROMPTS.md](PROMPTS.md).
