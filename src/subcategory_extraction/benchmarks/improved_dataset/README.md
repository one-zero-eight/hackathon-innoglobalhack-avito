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

## learned_keywords: новый прогон PR #9

2026-11-14, код `3768d76`, база main `e04c97d`. Веса переобучены только
на `docs/improved_dataset/train.jsonl` (4851 строка), combined-признаки,
40 фраз на категорию, C=4 и порог 0.5 зафиксированы до test.

| Модуль / модель | Версия / дата | Промпт | N | micro-F1 |
| --- | --- | --- | ---: | ---: |
| learned_keywords / TF-IDF + бинарные фразы | 3768d76 / 2026-11-14 | — | 119 | 0.800948 |

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods learned_keywords
```

Настоящие локальные run(), без API, переобучения внутри runner и подбора
порогов по test. Контракт меток и пересечения проверены перед финальным прогоном.
С фактическим train модели совпадают 0/119 строк после нормализации;
по эвристике Jaccard множеств слов ≥0.9 — 0/119 (включая точные совпадения).
Эвристика не заменяет полный семантический аудит; строки из метрики не удалены.
Исправленный test — диагностический, с предварительной разметкой.
Предыдущие результаты других моделей сохранены в results.json со своими
исходными датами и отпечатками в measurement_metadata. SHA-256 весов, train,
версии библиотек, предсказания и детали проверки пересечений находятся в JSON.
Исторические результаты обучения на Юле к этим весам не относятся.

## Qwen decision-model-preview: 2026-11-14

База `45ff38d`, ветка `ZolotarevAlexandr/feature/qwen-decision`;
отпечаток кода `3946999259f93a4db45e354a06a12e6cbff64778b66253c27da547e540e90c0f`.
Это отдельный диагностический test, с предварительной разметкой.

| Модуль / модель | Версия / дата | Промпт | N | micro-F1 |
| --- | --- | --- | ---: | ---: |
| qwen_decision / фиксированный порог 0.5, те же ответы | preview / 2026-11-14 | boundaries | 119 | 0.618395 |
| qwen_decision / train-калибровка | preview / 2026-11-14 | boundaries | 119 | 0.648980 |
| learned_keywords / опубликованные веса, свежий run() | 45ff38d / 2026-11-14 | — | 119 | 0.800948 |
| jev / проверенный пересчёт исторических ответов | исходная запись / 2026-11-14 | guided | 119 | 0.812665 |
| semantic_encoder_f1 / проверенный пересчёт исторических ответов | исходная запись / 2026-11-14 | — | 119 | 0.857143 |

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LEARNED_KEYWORDS_ARTIFACT=src/subcategory_extraction/learned_keywords/artifacts/learned-keywords.json uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods learned_keywords qwen_decision --workers 4 --append
```

Те же зафиксированные до test модель, промпт и пороги, что в classifier_methods;
SHA-256 калибровки `93b7828d87663965313dc05a1d6003f0bd2083d38bfe43d038da016f5c68bfe6`.
108 новых успешных API-ответов, 11 попаданий в кэш. Четыре потока с отдельными
клиентами. Фиксированный 0.5 — пересчёт тех же сохранённых вероятностей.
Калибровка улучшила F1 на 0.030584, но итог ниже прежних решений.

Подбор выполнялся только на 240 группах `train.jsonl`, контроль — ещё 120 групп.
Сначала объединены точные/близкие дубликаты, затем выбраны непересекающиеся
группы. Ни test, ни control не использовались для выбора промпта/порогов.
Отчёт подбора: `qwen-training-results.json`, артефакт и воспроизведение —
[qwen_decision](../../qwen_decision/README.md).
На control выбранный промпт: F1 0.725664 при 0.5 и 0.761905 после калибровки.

До оценки проверены категории и нормализованные точные пересечения с фактическим
train: 0/119; точных побайтовых совпадений с 240 описаниями подбора также 0/119.
Это не полный семантический аудит. Параметры после test не менялись.
Свежий RuBERT недоступен из-за отсутствующих LFS-весов; приведённый F1 —
штатный проверенный пересчёт старых ответов. Старые `recorded-results.json`
не менялись. Прежние результаты в `results.json` сохраняют своё происхождение.

## Первый Selectel-прогон: 2026-11-14, неуспешный

База `8c6fe87d` + изменения; SHA-256 кода `2d9d3f604557db6688508e60449f09e3ef1383dac1cf3d0ee536a2c334927a83`.
`typesafe/jev-1.13`, Selectel Decisions, общий guided-промпт и порог 0.5
зафиксированы до прогона. Состав данных и аудит пересечений проверяет runner.

| Модуль / модель | Версия / дата | Промпт | N | micro-F1 |
| --- | --- | --- | ---: | --- |
| jev / Selectel typesafe/jev-1.13 | 8c6fe87d + изменения / 2026-11-14 | google_decision_guided.md, SHA-256 f05534059aa8ab3e29a281d39d65bfc3e5cbc887330787aba267abd8891728b7 | 103/119 | —: failed, сетевой сбой |

```bash
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods jev --workers 4 --append
```

Прогон остановился при ошибке `Network is unreachable`. В results.json
сохранены первые 103 строки в порядке входа; итогового F1 нет. Часть уже
запущенных параллельных запросов успела заполнить дисковый кэш после ошибки,
поэтому оплаченные успешные ответы могут превышать 103. Исторический Jev F1
выше не является оценкой этого Selectel-клиента. Цена записывается отдельно
в cpu_costs по успешным ответам, а не по числу строк неуспешного отчёта.

## Текущий Selectel Jev: 2026-11-14T20:57:50+03:00

База `8c6fe87d` + изменения; SHA-256 кода
`2d9d3f604557db6688508e60449f09e3ef1383dac1cf3d0ee536a2c334927a83`.
Модель `typesafe/jev-1.13`, endpoint `https://api.selectel.ru/aig/v1/decisions`.
Общий guided-промпт (SHA-256 f05534059aa8ab3e29a281d39d65bfc3e5cbc887330787aba267abd8891728b7)
и порог 0.5 зафиксированы до оценки, подбора по test не было.

| Модуль / модель | Версия / дата | Промпт | N | micro-F1 |
| --- | --- | --- | ---: | ---: |
| jev / Selectel typesafe/jev-1.13 | 8c6fe87d + изменения / 2026-11-14 | google_decision_guided.md | 119 | 0.816754 |

TP / FP / FN: 156 / 17 / 53.
Пересечения с train: 0/119; строки не исключены.
Это диагностическая выборка с предварительной разметкой, не независимый ручной эталон.

```bash
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods jev --workers 1 --append
```

Последовательное завершение: 0 новых API-ответов, 119 чтений проверенного
Selectel-кэша от этого же переключения провайдера. Endpoint, payload и model
входят в ключ кэша; старые ответы BestAPI не подставлялись.
Все 119 ответов повторно прочитаны из текущего Selectel-кэша; итоговый проход сам по себе не является новым измерением модели или задержки.

Полный test сначала был отклонён автоматической проверкой одобрений;
после явного разрешения пользователя использован бюджет до 5 RUB.
Завершение обоих F1-сценариев потребовало ровно 22 новых запросов и
0.361691 RUB по provider-reported usage.cost, без автоматического пересчёта тарифов.
Фактические usage и признаки кэша сохранены в results.json. Остальные модули
сохраняют исходные measurement_metadata; recorded-results.json не менялся.
Классификация оценивается отдельно от генерации текстов.
