# Бенчмарк: F1 на исходном InnoGlobalHack
Сервис: subcategory_extraction

## Что проверяем

Micro-F1 дополнительных mcId на **всех 208 строках**
`docs/innoglobalhack_avito/public-test-dataset.parquet`,
SHA-256 `65b5a45a98b577d853a928932bbbdf8bd9bdeb469ded3016faace4d11f3eabaa`.
В файле нет исходного mcId, явно принимаем 101. Дубликаты остаются
в метрике; одинаковый вход при live может вычисляться один раз.
Исходный train `docs/innoglobalhack_avito/public-train-annotated.jsonl`;
для исторического локального обучения использованы 4821 отфильтрованных
описаний. Старый test содержит пересечения с train и противоречивую разметку.

## Запуск

Офлайн без повторного обучения/API:

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --replay
```

Проверяется соответствие сохранённых ответов текущему test, результаты
пишутся в `results.json`; `recorded-results.json` не изменяется.
Для нового сравнения выберите модули через `--methods tfidf`
и при необходимости `--offline`. Данные закреплены за сценарием;
параметры моделей и пути к весам задаются через `.env`.
Отсутствие весов или API даёт неуспешный статус без F1.

## Последний локальный прогон

2026-11-14, опубликованные веса Leo `1c2a6f5`, синхронизация с main `3144889`.
Настоящие CPU-вызовы `semantic_encoder_f1`, без переобучения и подбора порогов.
Артефакт и калибровка проверены по SHA-256 из `../../semantic_encoder_f1/artifact-manifest.json`.
Отпечаток исполняемого кода, версии библиотек и хеши файлов — в `results.json`.

| Модуль / модель | Версия весов / дата | Промпт | N | micro-F1 |
| --- | --- | --- | ---: | ---: |
| semantic_encoder_f1 / distilled RuBERT-tiny2 | 1c2a6f5 / 2026-11-14 | — | 208 | 0.769433 |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RUBERT_F1_ARTIFACT_PATH=src/subcategory_extraction/semantic_encoder_f1/artifacts RUBERT_F1_DEVICE=cpu uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods semantic_encoder_f1 --offline
```

Точные нормализованные совпадения с train модели (`improved_dataset/train.jsonl`):
**24/208**; строки из оценки не исключались. Это не поиск всех близких дубликатов.
Веса и пороги зафиксированы до оценки. API-вызовов классификатора нет.
Предыдущий локальный TF-IDF (`e814b58` + рабочие изменения): N=208, micro-F1=0.800000;
полный отчёт доступен в истории Git, это другая модель.
`--replay` заменяет `results.json` проверенным пересчётом исторических ответов.

## Сохранённые результаты

Исходный замер: 2026-11-14; версии исходников и параметры моделей —
в `recorded-results.json`. Офлайн-пересчёт проверяет прежние предсказания,
не качество текущих весов. Использованный guided-промпт:
`../../google_decision/prompts/google_decision_guided.md`.

| Модуль | Промпт | N | micro-F1 |
| --- | --- | ---: | ---: |
| google_decision | guided | 208 | 0.839394 |
| jev | guided | 208 | 0.846154 |
| tfidf | — | 208 | 0.778462 |
| jev_tfidf | guided | 208 | 0.842262 |
| semantic_encoder | — | 208 | 0.786704 |

## Выводы

Это пересчёт исторических ответов, не оценка текущих обученных весов.
Конкретные данные и SHA сохранены в JSON; качество полного генератора
черновиков не измеряется.

## learned_keywords: новый прогон PR #9

2026-11-14, код `3768d76`, база main `e04c97d`. Веса переобучены только
на `docs/improved_dataset/train.jsonl` (4851 строка), combined-признаки,
40 фраз на категорию, C=4 и порог 0.5 зафиксированы до test.

| Модуль / модель | Версия / дата | Промпт | N | micro-F1 |
| --- | --- | --- | ---: | ---: |
| learned_keywords / TF-IDF + бинарные фразы | 3768d76 / 2026-11-14 | — | 208 | 0.745358 |

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods learned_keywords
```

Настоящие локальные run(), без API, переобучения внутри runner и подбора
порогов по test. Контракт меток и пересечения проверены перед финальным прогоном.
С фактическим train модели совпадают 24/208 строк после нормализации;
по эвристике Jaccard множеств слов ≥0.9 — 30/208 (включая точные совпадения).
Эвристика не заменяет полный семантический аудит; строки из метрики не удалены.
Исправленный test — диагностический, с предварительной разметкой.
Предыдущие результаты других моделей сохранены в results.json со своими
исходными датами и отпечатками в measurement_metadata. SHA-256 весов, train,
версии библиотек, предсказания и детали проверки пересечений находятся в JSON.
Исторические результаты обучения на Юле к этим весам не относятся.
