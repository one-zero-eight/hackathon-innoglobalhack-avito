# Отчёт: подходы к выделению микрокатегорий (subcategory_extraction)

Сводка по всем реализациям шага 1 пайплайна, объединённым в `main` со всех
веток репозитория. Каждый подход — отдельный модуль
`src/subcategory_extraction/<module>/pipeline.py`, наследует
`BasePipeline` и реализует `run(PipelineInput) -> list[int]`.

Данные сняты на единой матрице конфигураций
(`src/subcategory_extraction/benchmarks/configurations.py`):

- **classifier_methods** — micro-F1 на официальном InnoGlobalHack test
  (`case_test`, 208 строк);
- **improved_dataset** — micro-F1 на ручной смысловой разметке
  (`manual_test`, 139 строк);
- **response_time** — средняя задержка настоящего вызова `run()` на
  10 фиксированных входов (без прогрева, кэш включён);
- **cpu_costs** — ручная запись стоимости токенов.

Детали протоколов, id датасетов и SHA — в `results.json` каждого сценария.

---

## Сводная таблица micro-F1

Сортировка по убыванию F1 на `case_test`. «—» — модуль не прогонялся в этом
сценарии (нет API/весов/runtime) либо результат не валиден.

| Модуль | case_test F1 | manual_test F1 | Среднее время (с) | Источник |
| --- | ---: | ---: | ---: | --- |
| jev (guided) | 0.8462 | — | — | recorded-replay |
| jev_tfidf (guided) | 0.8423 | — | — | recorded-replay |
| google_decision (guided) | 0.8394 | 0.7379 | 0.801 | live |
| qwen_decision | 0.6297 | — | 14.309 | историческое измерение (README ветки) / live-замер времени |
| tfidf::legacy | 0.8000 | 0.6979 | 0.001 | live |
| tfidf | 0.7785 | 0.6811 | 0.001 | live |
| semantic_encoder | 0.7867 | — | — | recorded-replay |
| semantic_encoder_f1 | 0.7694 | 0.7701 | 0.014 | live |
| semantic_evidence | 0.7643 | 0.7747 | 0.033 | live |
| snippet_logistic_regression | 0.7632 | 0.7593 | 0.008 | live |
| regex | 0.7520 | 0.7684 | 0.005 | live |
| learned_keywords | 0.7454 | 0.7307 | 0.002 | live |
| google_decision::baseline | 0.7385 | 0.7877 | 0.822 | live |
| semantic_encoder::threshold-encoder | 0.7284 | 0.7569 | 0.012 | live |
| semantic_encoder_calibrated | 0.7273 | 0.7519 | 0.011 | live |
| local_jev (guided_ru) | 0.0 | — | — | not valid (macOS) |
| tinybert | — | — | — | обучение |
| jev::baseline, jev_boundaries, jev_retrieval, evidence_jev, jev_arbiter | — | — | — | нет авторизованного API |
| qwen_decision::guided / :boundaries | — | — | — | нет ключа |
| semantic_encoder (базовый, live) | — | — | — | артефакт не загружен |
| local_jev | 0.3990* | — | 1.200 | сохранённые CPU-предсказания; live недоступен на macOS |

> Примечание: строки «recorded-replay» — пересчёт исторических сохранённых
> ответов (validated against matching case_test hash), без новых API-вызовов;
> они отражают прежние веса/модели, а не текущие.
> *local_jev: 0.3990 — исторические сохранённые CPU-предсказания (english/0.5)
> из прежних отчётов; на этой машине сервер не запускается.

---

## Описания подходов

### Локальные модели (без платного API)

#### regex
Правила офлайн-аннотатора (`dataset_annotation.annotate_dataset.propose_*`),
без индивидуальных исключений. Ноль токенов API, максимально быстрый.
Хорошая базовая линия на ручных данных (0.7684 на manual_test).

#### tfidf
Локальная логистическая регрессия на TF-IDF признаках; артефакт JSON.
`tfidf::legacy` — старые веса `out/tfidf.json`; `tfidf` — веса, обученные
на `case_train` (`out/classifier-methods/tfidf.json`). Самый быстрый (1 мс).

#### learned_keywords
TF-IDF + бинарные признаки выученных фраз (40 фраз на категорию, C=4,
порог 0.5). Обучение на `manual_train`. Компактный и бесплатный.

#### semantic_encoder
Fine-tuned RuBERT-tiny2; классификатор на представлении объявления
(RuBERT-encoder + линейная голова). Базовый вариант требует артефакт
`out/semantic-encoder`, в этой среде не загружен; исторический результат
0.7867 получен из recorded.

#### semantic_encoder_calibrated
Тот же RuBERT-encoder, но с зафиксированными (validation/calibration)
порогами категорий вместо 0.5. Более сбалансирован на ручных данных.

#### semantic_encoder_f1
CPU-вариант: выбор checkpoint, оконная токенизация, attention-маска и
дистилляция; результат привязан к артефакту с validation-калибровкой.
Лучшая чисто-локальная точность на обоих датасетах (0.7694 / 0.7701).

#### snippet_logistic_regression
Классификация по бинарным n-граммам в окнах вокруг ключевых слов.
Окна до 7 токенов сохраняют границы ключей; веса — gzip-JSON в модуле.
Внутренний 5-fold показатель до 0.93, на открытом test — 0.763.

#### semantic_evidence
Композиция двух локальных моделей: document-encoder (всё объявление) и
evidence-encoder (фрагменты), взвешенная композиция (0.25/0.75).
CPU-only runtime, артефакты проверяются по SHA-256 внутри модуля.

#### tinybert
Huawei TinyBERT 4L-312D (end-to-end fine-tuning). Ветка держит только
`train.py`/`model.py`/`pipeline.py`; обученные веса отсутствуют — их нужно
воспроизвести `python -m subcategory_extraction.tinybert.train`
(5 эпох, загрузка оригинала с HuggingFace).

### LLM-модели (облачные API)

#### google_decision
Промпт-адаптер: независимые вопросы к Gemini-модели для каждой услуги.
Варианты `guided` (примеры-ориентиры) и `baseline`. Guided — один из
лидеров на `case_test` (0.8394). Среднее время вызова ~0.8 с.

#### qwen_decision
Qwen Token Plan System One (`decision-model-preview`): независимые вопросы,
пороги подбираются только по train. На recorded — 0.6297 на case_test.
Для live-прогонов нужен ключ `QIANWEN_TOKEN_PLAN_API_KEY` (не задан).

#### jev
Нативный Jev (модель, возвращающая вероятности Noul) через Selectel.
Приватный guided-промпт. По recorded-реплею — 0.8462, лучший на case_test.
Для live-прогонов нужен `SELECTEL_API_KEY` (из .env берётся `BESTAPI_API_KEY`,
который не авторизует Selectel).

#### jev_calibrated
Jev с зафиксированным guided-промптом и train-калиброванными порогами
(calibration.json). Требует тот же авторизованный Jev API.

#### jev_boundaries / jev_retrieval / evidence_jev / jev_arbiter / jev_tfidf
Варианты на базе Jev: фиксированные границы категорий; добавление пар
ближайших размеченных примеров; выборка цитат сильной LLM с последующей
оценкой Jev; маршрутизация к сильному JSON Schema-арбитру; взвешивание
вероятностей Jev и TF-IDF-голов. Все упираются в неавторизованный Jev API
и в этой среде дают только исторические/отсутствующие значения.

#### local_jev (jevos-v5)
feder-cr/jev локально на CPU (localhost:8017). В этой macOS-среде
автозагрузка runtime не поддерживается, официальный сервер не запущен;
сохранённые CPU-предсказания `guided_ru` невалидны (F1=0).

---

## Недоступно в текущей среде

| Группа | Причина |
| --- | --- |
| jev, jev_calibrated, jev_boundaries, jev_retrieval, evidence_jev, jev_arbiter, jev_tfidf (+baseline) | в `.env` `BESTAPI_API_KEY` не авторизует Selectel/`SELECTEL_API_KEY` отсутствует |
| qwen_decision (+guided/boundaries) | `QIANWEN_TOKEN_PLAN_API_KEY` не задан |
| local_jev (+guided_ru) | runtime поддерживает Linux x86_64; на macOS официальный сервер не запущен |
| semantic_encoder (базовый live) | артефакт `out/semantic-encoder` не загружен |
| tinybert | обученные веса в ветке отсутствуют (нужно обучение) |

Фиктивные значения не подставляются; в `results.json` такие конфигурации
помечены `failed`/`requires_rerun` с явной ошибкой.

---

## Как воспроизвести

```bash
uv sync
cp .env.example .env            # заполнить ключи и DB_*
uv run python src/database_load.py   # наполнить PostgreSQL (все таблицы)

# F1 на официальном test (case_test)
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --all-configurations

# F1 на ручной разметке (manual_test)
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --all-configurations

# Время работы
uv run python src/subcategory_extraction/benchmarks/response_time/run.py --all-configurations
```

Проверка сохранённых исторических ответов без новых вызовов — `--replay`.
(c)