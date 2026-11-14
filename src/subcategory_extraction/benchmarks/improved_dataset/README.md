# Бенчмарк: F1 на исправленном датасете
Сервис: subcategory_extraction

## Что проверяем

Micro-F1 дополнительных mcId на 119 строках
`docs/improved_dataset/reviewed_test.parquet`,
SHA-256 `0af183508c1501425a1330675ceb8923dfe130fff3f86e418ebe3ec4baba34a2`.
Train `docs/improved_dataset/train.jsonl` (4851 строк),
SHA-256 `d365141906cd7e2a1d66792aa2e37661d8bcdcd0a4bbaf3781649a9108238b76`.
Исходный mcId test предполагается равным 101. Разметка предварительная,
не независимый ручной эталон. Порог/промпт по test не выбираются.

## Запуск

Без новых API-вызовов и без весов:

```bash
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --replay
```

Скрипт проверяет SHA датасета, полные метки и сохранённые ответы,
пересчитывает TP/FP/FN и micro-F1 в `results.json`. Исходные ответы
не перезаписываются (`recorded-results.json`). Для новых вызовов (параметры и пути к весам — через `.env`):

```bash
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods tfidf --offline
```

Последняя команда требует обученного артефакта TF-IDF и может закончиться
ошибкой, если он отсутствует. Доступность других моделей/API проверяется
отдельно, фиктивных предсказаний нет.

## Последний локальный прогон

2026-11-14, опубликованные веса Leo `1c2a6f5`, синхронизация с main `3144889`.
Настоящие CPU-вызовы `semantic_encoder_f1`, без переобучения и подбора порогов.
Артефакт и калибровка проверены по SHA-256 из `../../semantic_encoder_f1/artifact-manifest.json`.
Отпечаток исполняемого кода, версии библиотек и хеши файлов — в `results.json`.

| Модуль / модель | Версия весов / дата | Промпт | N | micro-F1 |
| --- | --- | --- | ---: | ---: |
| semantic_encoder_f1 / distilled RuBERT-tiny2 | 1c2a6f5 / 2026-11-14 | — | 119 | 0.857143 |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RUBERT_F1_ARTIFACT_PATH=src/subcategory_extraction/semantic_encoder_f1/artifacts RUBERT_F1_DEVICE=cpu uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods semantic_encoder_f1 --offline
```

Точные нормализованные совпадения с train модели (`improved_dataset/train.jsonl`):
**0/119**; строки из оценки не исключались. Это не поиск всех близких дубликатов.
Веса и пороги зафиксированы до оценки. API-вызовов классификатора нет.
Предыдущий локальный TF-IDF (`e814b58` + рабочие изменения): N=119, micro-F1=0.758808;
полный отчёт доступен в истории Git, это другая модель.
`--replay` заменяет `results.json` проверенным пересчётом исторических ответов.

## Сохранённые результаты

Исходный замер: 2026-11-14; версии исходников и параметры моделей —
в `recorded-results.json`. Офлайн-пересчёт проверяет прежние предсказания,
не качество текущих весов. Использованный guided-промпт:
`../../google_decision/prompts/google_decision_guided.md`.

| Модуль | Промпт | N | micro-F1 |
| --- | --- | ---: | ---: |
| google_decision | guided | 119 | 0.801061 |
| jev | guided | 119 | 0.812665 |
| tfidf | — | 119 | 0.762332 |
| jev_tfidf | guided | 119 | 0.854962 |
| semantic_encoder | — | 119 | 0.811530 |
| semantic_encoder_f1 (distilled) | — | 119 | 0.857143 |

## Выводы

Значения воспроизведены по сохранённым предсказаниям, не по новым весам;
на скрытый test их переносить нельзя. Отпечатки входов и кода — в JSON.
