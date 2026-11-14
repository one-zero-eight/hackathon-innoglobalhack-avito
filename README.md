# Услуги из объявлений Авито → черновики

По тексту объявления и его микрокатегории находим услуги из других
микрокатегорий «Ремонт и отделка» (101–106) и генерируем черновики объявлений.

## Структура

| Путь | Что это |
| --- | --- |
| `src/main.py` | CLI + всё общее: контракт, микрокатегории, `BasePipeline`, `LLMClient` |
| `config.json` | выбор реализации каждого шага |
| `src/subcategory_extraction/` | шаг 1: выделение подкатегорий услуг |
| `src/draft_generation/` | шаг 2: генерация черновиков |
| `data/` | пример входа |
| `docs/` | кейс, критерии, справочник, датасеты |
| `backlog/` | локальный бэклог (в `.gitignore`) |

В каждом разделе — несколько вариантов: `<module>/pipeline.py` и, при
необходимости, `<module>/prompts/`. Общие бенчмарки лежат в
`benchmarks/<name>/`: `run.py` для запуска, `README.md` для описания
и таблиц результатов. Экспериментальные решения тоже оформляем
как модули, без отдельной папки `experiments/`.

## Выбор реализации

Корневой `config.json` задаёт имя папки модуля для каждого шага:

```json
{
  "subcategory_extraction": "tfidf",
  "draft_generation": "sample_llm"
}
```

Доступны `subcategory_extraction/google_decision`,
`subcategory_extraction/tfidf`.
Доступен генератор `draft_generation/sample_llm`. `null` означает, что модуль
не выбран. CLI выдаёт явную ошибку.
Варианты (например, `jev`, `embeddings` или `llm`) добавляются в
`src/<service>/<module>/`; затем в конфиге указывается имя папки вместо `null`.
Каждый вариант содержит `__init__.py` и `pipeline.py` с классом и интерфейсом
своего раздела (см. [AGENTS.md](AGENTS.md)). Переключение — только через конфиг,
без изменения оркестрации. Несуществующий модуль вызывает ошибку.

Для сравнения вариантов используем общие бенчмарки раздела: качество,
время ответа и стоимость для конкретного сценария и выборки.

## Пайплайн

```
description + mcId
   │
   ▼
subcategory_extraction    список mcId найденных услуг (101–106)
   │
   ▼
draft_generation          черновик на каждую целевую категорию
   │
   ▼
{"drafts": [{"mcId": 104, "title": "...", "text": "..."}]}
```

Запуск:

```bash
uv sync
cp .env.example .env          # заполнить LLM_API_KEY

uv run python src/main.py --input data/input.json --output out/output.json
```

Для полного запуска выберите реализации обоих шагов и настройте `.env`
(для генератора см. `src/draft_generation/README.md`).
Общий `LLMClient` — абстрактный интерфейс для конкретных клиентов.

## Оценка полного решения

```bash
uv sync
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --live --config run.config.json
```

`run.config.json` задаёт датасет (`.parquet` или официальный `.jsonl`),
путь к конфигу модулей, исходную категорию (101 для официальных данных),
необязательный лимит строк и частоту логирования.
Все относительные пути считаются от файла конфигурации.
`pipeline_config` ссылается на обычный `config.json`: необходимо выбрать
оба рабочих модуля, включая генератор. При `null` оценка завершается
ошибкой, без подмены полного решения классификатором. Параметры моделей
и секреты выбранных модулей по-прежнему берутся из `.env`.

Метрика — `sklearn.metrics.f1_score(average="micro", zero_division=0)`
по множествам `mcId` из **итоговых drafts**, баллы за классификацию —
`60 × micro-F1` (не итоговый балл за все критерии хакатона).
Повторы не учитываются, ошибочный вывод исходной категории считается FP.
Если все эталоны и ответы пусты, micro-F1 равен 0.
Отчёт `src/draft_generation/benchmarks/full_pipeline_evaluation/live-results.json`
содержит метрику, TP/FP/FN, конфигурацию, время, эталоны и полные ответы
для каждой строки. Поле `report` конфига относится к прежнему `src/run.py`;
бенчмарк сохраняет контрольную точку рядом со своим `run.py`. Ошибка обработки
не превращается в пустой ответ: после ограниченных повторов прогон
помечается неуспешным без итоговой метрики. `--resume` продолжает
незавершённый прогон из сохранённых ответов, если код и входы не менялись.

`training_dataset` включает проверку нормализованных точных совпадений
описаний с train; пересечения отражаются в отчёте и предупреждении,
но не удаляются из оценки. Это не поиск всех близких дубликатов
и не гарантия отсутствия утечки. `null` отключает проверку.
Официальный test следует использовать для итоговой оценки после
фиксации модели и порогов, не для их подбора.

## Бенчмарки

Каждый бенчмарк хранится отдельно:

```text
src/<раздел>/benchmarks/<название>/
├── run.py
└── README.md
```

В README — данные, команды запуска и актуальные результаты: качество,
число вызовов, среднее/медиана/p95 времени ответа и стоимость.
Если есть системные промпты, в сравнении показываем лучший по micro-F1
на одной validation-выборке, а в отдельной таблице — каждый промпт.

| Раздел | Бенчмарк | Сценарий |
| --- | --- | --- |
| Выделение | [Качество TF-IDF](src/subcategory_extraction/benchmarks/tfidf_quality/README.md) | Новое обучение; validation и отдельный holdout |
| Выделение | [Сравнение промптов](src/subcategory_extraction/benchmarks/prompt_comparison/README.md) | TF-IDF и реальные baseline/guided Google на одних 32 входах |
| Выделение | [Время ответа](src/subcategory_extraction/benchmarks/response_time/README.md) | Настоящие run() TF-IDF и Google на одних 10 входах |
| Генерация | [Контракт sample_llm](src/draft_generation/benchmarks/sample_llm/README.md) | Офлайн-проверка схемы и транспорта |
| Генерация | [Время ответа](src/draft_generation/benchmarks/response_time/README.md) | 10 настоящих вызовов, по одному черновику |
| Полный пайплайн | [Оценка решения](src/draft_generation/benchmarks/full_pipeline_evaluation/README.md) | Реальные drafts на 208 строках и отдельная проверка оценщика |

Стоимость в таблицах заполняется вручную человеком или агентом.
Для локальных прогонов и подтверждённой бесплатной квоты — **«бесплатно»**.
Скрипты измеряют время, но не вычисляют цену.

Аудит, переразметка и сборка датасета не входят в эти бенчмарки.
Нерабочие эксперименты со старым внешним кодом удалены из рабочей структуры;
их прежние версии остаются в Git.

Запуск из корня:

```bash
uv run python src/subcategory_extraction/benchmarks/tfidf_quality/run.py
uv run python src/subcategory_extraction/benchmarks/prompt_comparison/run.py
uv run python src/subcategory_extraction/benchmarks/response_time/run.py
uv run python src/draft_generation/benchmarks/sample_llm/run.py
uv run python src/draft_generation/benchmarks/response_time/run.py
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py
uv run python src/draft_generation/benchmarks/full_pipeline_evaluation/run.py --live --config run.config.json
```

Сравнение промптов, задержки Google/генерации и полный `--live` обращаются
к реальному API. Для подтверждённого пользователем текущего ключа используется
бесплатная квота. Измерения включают фактические ожидания и повторы запросов.

## Контрибьюшн

1. Ветка на каждую фичу: `<username>/feature/<slug>`. В `main` — только через PR.
2. Бэклог локальный: `backlog/{TODO,PROCESS,REVIEW,DONE}`,
   задача перемещается по папкам по мере работы.
3. Новый бенчмарк — папка `src/<service>/benchmarks/<name>/` с `run.py`
   и `README.md`. Перед коммитом прогнать на всех версиях кода
   и обновить таблицы прогонов.
4. Правки в чужих сервисах — через свою ветку и с согласия автора.
5. Контракт входа/выхода не меняем без согласования со всей командой.

Подробнее — в [AGENTS.md](AGENTS.md).
