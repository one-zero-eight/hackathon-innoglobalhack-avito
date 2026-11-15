# Бенчмарк: F1 на исходном InnoGlobalHack

Micro-F1 дополнительных категорий 102–106 на всех 208 строках официального
InnoGlobalHack test из таблицы PostgreSQL `case_test`
(источник — `docs/innoglobalhack_avito/public-test-dataset.parquet`).
Исходный mcId принимается равным 101.
Содержимое таблицы идентично исходному parquet: SHA-256 строк таблицы —
`cd6f432e3841e842d04bb07827b92f9b644c56b2b528cb36098433168890ae15` (из
PostgreSQL), SHA-256 исходного файла —
`65b5a45a98b577d853a928932bbbdf8bd9bdeb469ded3016faace4d11f3eabaa`.

## Результаты

Одна строка на решение. Повторный прогон заменяет строку; остальные
результаты сохраняют собственную дату и происхождение. Таблица соответствует
[results.json](results.json); снимок исходных ответов
[recorded-results.json](recorded-results.json) не перезаписывается.
Значения измерены на данных, идентичных прежнему parquet, поэтому
результаты остаются валидными после перевода источника в PostgreSQL.

### Результаты (все конфигурации)

| Конфигурация | Версия / дата измерения | Промпт | N | micro-F1 | Происхождение |
| --- | --- | --- | ---: | ---: | --- |
| jev | jev-1.13.0 / сохранённые ответы | guided | 208 | 0.846154 | сохранённые ответы, без новых вызовов |
| jev_tfidf | jev-1.13.0 / сохранённые ответы | guided | 208 | 0.842262 | сохранённые ответы, без новых вызовов |
| google_decision | gemini-decision-1 / 2026-11-15 | guided | 208 | 0.839394 | локальные run() |
| tfidf::legacy | локальные веса out/tfidf.json / 2026-11-15 | — | 208 | 0.800000 | локальные run() |
| semantic_encoder (базовый) | rubert-tiny2 / сохранённые ответы | — | 208 | 0.786704 | сохранённые ответы, без новых вызовов |
| tfidf | веса 2ef7ed22c74a / 2026-11-15 | — | 208 | 0.778462 | локальные run() |
| semantic_encoder_f1 | веса 1c2a6f5 / 2026-11-15 | — | 208 | 0.769433 | локальные run() |
| semantic_evidence | веса selection.json / 2026-11-15 | — | 208 | 0.764314 | локальные run() |
| snippet_logistic_regression | model.json.gz / 2026-11-15 | — | 208 | 0.763194 | локальные run() |
| regex | правила annotate_dataset / 2026-11-15 | — | 208 | 0.751979 | локальные run() |
| learned_keywords | веса d06457a8c082 / 2026-11-15 | — | 208 | 0.745358 | локальные run() |
| google_decision::baseline | gemini-decision-1 / 2026-11-15 | baseline | 208 | 0.738462 | локальные run() |
| semantic_encoder::threshold-encoder | out/rubert-thresholds/encoder / 2026-11-15 | — | 208 | 0.728411 | локальные run() |
| semantic_encoder_calibrated | out/rubert-thresholds/encoder / 2026-11-15 | — | 208 | 0.727273 | локальные run() |
| qwen_decision | decision-model-preview / сохранённые ответы | boundaries | 208 | 0.629674 | сохранённые ответы, без новых вызовов |
| local_jev / jevos-v5 | jevos-v5 / сохранённые предсказания | english / 0.5 | 208 | 0.399050 | сохранённые CPU-предсказания |
| local_jev / jevos-v5 | jevos-v5 / сохранённые предсказания | guided_ru / 0.5 | 208 | 0.000000 | сохранённые CPU-предсказания; на macOS runtime недоступен, результат невалиден |

Порядок — убывание micro-F1. «Сохранённые ответы» — пересчёт исторических
предсказаний (`recorded-results.json`) без новых API-вызовов, не измерение
текущих весов. Строки 2026-11-15 — свежие локальные run() в едином
конфигурационном прогоне; `google_decision` и `google_decision::baseline`
перепрогнаны вживую с рабочим `LLM_API_KEY`.

### Недоступно в текущем окружении (без ключа / runtime / артефакта)

Не прогоняются на этой машине, потому что в `.env` нет работающего ключа
API, runtime `local_jev` поддерживает только Linux x86_64, либо отсутствует
артефакт. Фиктивные значения не подставляются:

- `jev`, `jev::baseline`, `jev_boundaries`, `jev_tfidf`(+baseline),
  `jev_retrieval`(+baseline), `evidence_jev`, `jev_arbiter`(+baseline),
  `jev_calibrated` — в `.env` нет ключа, авторизующего выбранного провайдера
  (`SELECTEL_API_KEY` отсутствует; `BESTAPI_API_KEY` отвечает HTTP 401);
  новых вызовов нет, для `jev`/`jev_tfidf` остаются исторические
  сохранённые ответы (выше отмечены как «сохранённые»);
- `qwen_decision`, `qwen_decision::guided`, `qwen_decision::boundaries` —
  не прогоняются по решению команды (Qwen не запускаем);
- `local_jev` (+ guided_ru) — runtime только для Linux x86_64;
- `semantic_encoder` (базовая) — артефакт `out/semantic-encoder/metadata.json`
  не загружен (известен исторический результат 0.786704 по сохранённым ответам);
## Карта результатов

![Подходы: Micro-F1 (case_test) × время](approaches-map.png)

Ось X — среднее время одного вызова (логарифмическая шкала), ось Y —
micro-F1 на исходном test. Источник времени — сценарий `response_time`.
Синие точки — свежие локальные замеры; красные — исторические/сохранённые.

## Запуск

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TFIDF_ARTIFACT_PATH=out/classifier-methods/tfidf.json uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods lexical.regex lexical.tfidf lexical.learned_keywords --offline
```

Для других реализаций замените список `--methods`. Для проверки исходного
снимка ответов используйте `--replay`; это пересчёт старых предсказаний,
а не измерение текущих весов или задержки модели. Флаг `--offline` запрещает
API-модули. Отсутствие весов/API фиксируется как ошибка без F1.
После запуска обновите соответствующие строки этой таблицы по JSON.

## Условия и ограничения

2026-11-14, опубликованные веса Leo `1c2a6f5`, синхронизация с main `3144889`.
Настоящие CPU-вызовы `semantic_encoder_f1`, без переобучения и подбора порогов.
Артефакт и калибровка проверены по SHA-256 из `../../semantic_encoder_f1/artifact-manifest.json`.
Отпечаток исполняемого кода, версии библиотек и хеши файлов — в `results.json`.


```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RUBERT_F1_ARTIFACT_PATH=src/subcategory_extraction/semantic/semantic_encoder_f1/artifacts RUBERT_F1_DEVICE=cpu uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods semantic.semantic_encoder_f1 --offline
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


## Выводы

Это пересчёт исторических ответов, не оценка текущих обученных весов.
Конкретные данные и SHA сохранены в JSON; качество полного генератора
черновиков не измеряется.

## learned_keywords: новый прогон PR #9

2026-11-14, код `3768d76`, база main `e04c97d`. Веса переобучены только
на `docs/improved_dataset/train.jsonl` (4851 строка), combined-признаки,
40 фраз на категорию, C=4 и порог 0.5 зафиксированы до test.


```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods lexical.learned_keywords
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

## Qwen decision-model-preview: 2026-11-14

База `45ff38d`, рабочая ветка `ZolotarevAlexandr/feature/qwen-decision`,
отпечаток кода `3946999259f93a4db45e354a06a12e6cbff64778b66253c27da547e540e90c0f`.
Модель: `decision-model-preview`, Qwen Token Plan System One.


```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LEARNED_KEYWORDS_ARTIFACT=src/subcategory_extraction/lexical/learned_keywords/artifacts/learned-keywords.json uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods lexical.learned_keywords decisions.qwen_decision --workers 4 --append
```

Промпт и пороги выбраны только по 240 группам исправленного train и сохранены
до test: SHA-256 калибровки
`93b7828d87663965313dc05a1d6003f0bd2083d38bfe43d038da016f5c68bfe6`.
Отдельный control из 120 групп не менял решение. Подробности —
[qwen_decision](../../qwen_decision/README.md).

В Qwen-прогоне 105 успешных новых API-ответов и 103 попадания в кэш;
вероятности, usage и их соответствие строкам сохранены в `results.json`.
Четыре потока имеют независимые экземпляры клиента. Фиксированный порог 0.5
пересчитан по тем же вероятностям и не является новым API-прогоном.
Исторические ответы Jev проверены штатным `--replay` с временным отчётом в out;
это не новая оценка текущей версии Jev.

До оценки проверены контракт категорий и нормализованные точные пересечения
с фактическим train Qwen: 24/208 относительно всего исправленного train.
С выбранными 240 описаниями подбора совпадают 10/208 строк по точным байтам
описания. Строки не исключались. Проверка не гарантирует отсутствия всех
семантических дубликатов. Пороги по результатам test не менялись.
Калибровка здесь снизила F1 на 0.007535; Qwen не превзошёл прежние решения.

Свежий RuBERT-запуск недоступен: в checkout только LFS-указатель весов,
попытка загрузки опубликованного файла через GitHub Media вернула HTTP 404.
Его прежний результат сохранён с прежним происхождением; новые веса ему не подставлялись.
`--append` сохраняет результаты других модулей с их датами и отпечатками;
`recorded-results.json` не изменён.

## Локальный jevos-v5

Английский вариант выбран на 12 примерах прежнего официального train;
порог 0.5 не подбирался по test. Сохранённые 208 входов и эталонных меток
побайтово совпали со снимком case_test текущего отчёта, поэтому обе строки
Jevos совместимы с этим сценарием. Новые вызовы модели при merge не выполнялись.
Манифест весов, вероятности и исходная дата измерения сохранены в results.json.
Первый запуск теперь загружает модель и runtime в игнорируемый out/local-jev/;
команды и ограничения — в ../../local_jev/README.md.
