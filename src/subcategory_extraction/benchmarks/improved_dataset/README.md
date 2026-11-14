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

2026-11-14, `e814b58` + рабочие изменения (отпечаток в `results.json`).
Настоящие вызовы текущего TF-IDF, существующие веса без переобучения:
**N=119, micro-F1=0.758808**.
Параметры взяты из `.env`; расходы API отсутствуют. Пересечения с train
сохранены в `results.json` и не исключены из метрики.
`results.json` содержит последний прогон; `--replay` заменяет его
проверенным пересчётом исходных `recorded-results.json`.

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
