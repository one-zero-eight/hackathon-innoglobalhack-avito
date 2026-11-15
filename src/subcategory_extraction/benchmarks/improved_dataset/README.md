# Бенчмарк: F1 на ручной разметке test (manual_test)

Micro-F1 дополнительных категорий 102–106 на всех 139 строках ручной
смысловой разметки test из таблицы PostgreSQL `manual_test`
(журнал `docs/manual_review/inputs.jsonl` + `reviews/*.jsonl`, сплит test).
Исходный mcId принимается равным 101.
SHA-256 исходных файлов журнала: `11a849f044d7166f5ad515b26db24c6b56f88240589371f01a76396049cbbed1`;
SHA-256 содержимого таблицы: `fae05cba56fdf57a15aa3979afd828b73928efcca64694be5d805d34e002b99d`
(пересчёт через `src/database.py`).

Сценарий переведён с прежнего `docs/improved_dataset/reviewed_test.parquet`
(119 строк) на `manual_test` по указанию пользователя: эталоном считается новая
ручная разметка. Количество строк и метки изменились, поэтому результаты на
старом датасете с новым протоколом несовместимы и пересчитываются заново.

## Результаты

Одна строка на решение. Повторный прогон заменяет строку; остальные
результаты сохраняют собственную дату и происхождение. Таблица соответствует
[results.json](results.json); снимок исходных ответов
[recorded-results.json](recorded-results.json) не перезаписывается.

| Модуль | Версия / дата измерения | Промпт | N | micro-F1 | Происхождение |
| --- | --- | --- | ---: | ---: | --- |
| semantic_evidence | selection.json / 2026-11-15 | — | 139 | 0.774674 | локальные run(), источник — БД |
| semantic_encoder_f1 | локальный прогон из БД / 2026-11-15 | — | 139 | 0.770093 | локальные run(), источник — БД |
| regex | правила / 2026-11-15 | — | 139 | 0.768362 | локальные run(), источник — БД |
| snippet_logistic_regression | model.json.gz / 2026-11-15 | — | 139 | 0.759295 | локальные run(), источник — БД |
| semantic_encoder::threshold-encoder | out/rubert-thresholds/encoder / 2026-11-15 | — | 139 | 0.756856 | локальные run(), источник — БД |
| semantic_encoder_calibrated | локальный прогон из БД / 2026-11-15 | — | 139 | 0.751852 | локальные run(), источник — БД |
| google_decision::baseline | gemini-decision-1 / 2026-11-15 | baseline | 139 | 0.787671 | локальные run(), источник — БД |
| learned_keywords | локальный прогон из БД / 2026-11-15 | — | 139 | 0.730697 | локальные run(), источник — БД |
| google_decision | gemini-decision-1 / 2026-11-15 | guided | 139 | 0.737945 | локальные run(), источник — БД |
| tfidf::legacy | out/tfidf.json / 2026-11-15 | — | 139 | 0.697872 | локальные run(), источник — БД |
| tfidf | локальный прогон из БД / 2026-11-15 | — | 139 | 0.681128 | локальные run(), источник — БД |
| jev | требуется пересчёт | guided | — | — | нет рабочего BESTAPI-ключа (HTTP 401) |
| jev_tfidf | требуется пересчёт | guided | — | — | нет рабочего BESTAPI-ключа (HTTP 401) |
| qwen_decision | не прогоняется | boundaries | — | — | по решению команды Qwen не запускаем |
| local_jev / jevos-v5 | требуется пересчёт | english / 0.5 | — | — | runtime только для Linux x86_64 |
| local_jev / jevos-v5 | требуется пересчёт | guided_ru / 0.5 | — | — | runtime только для Linux x86_64 |

Недоступны в этом окружении: `jev::baseline`, `jev_boundaries`,
`jev_tfidf`(+baseline), `jev_retrieval`(+baseline), `evidence_jev`,
`jev_arbiter`(+baseline) — с текущим `BESTAPI_API_KEY` API `bestapi.fugue.pro`
отвечает HTTP 401; `qwen_decision`(+guided/+boundaries) — не прогоняем по
решению команды; `local_jev`(+guided_ru) — runtime Linux-only;
`semantic_encoder` (базовая) — артефакт не загружен.

## Запуск

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TFIDF_ARTIFACT_PATH=out/classifier-methods/tfidf.json uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods lexical.tfidf lexical.learned_keywords --offline
```

Для других реализаций замените список `--methods`. Для проверки исходного
снимка ответов используйте `--replay`; это пересчёт старых предсказаний,
а не измерение текущих весов или задержки модели. Флаг `--offline` запрещает
API-модули. Отсутствие весов/API фиксируется как ошибка без F1.
После запуска обновите соответствующие строки этой таблицы по JSON.

## Условия и ограничения

Команда требует обученных артефактов и может закончиться ошибкой, если они
отсутствуют. Доступность моделей/API проверяется отдельно, фиктивных
предсказаний нет. Датасет читается из PostgreSQL (см. `src/database.py`);
перед прогоном таблицы наполняются командой:

```bash
uv run python src/database_load.py
```

## Последний локальный прогон

2026-11-15, источники данных — PostgreSQL. Настоящие CPU-вызовы локальных
моделей `tfidf`, `learned_keywords`, `semantic_encoder`,
`semantic_encoder_calibrated`, `semantic_encoder_f1` на `manual_test`;
API-модули (`google_decision`, `jev`, `jev_tfidf`, `qwen_decision`) ещё не
пересчитаны на новой разметке. Отпечаток исполняемого кода, версии библиотек
и хеши — в `results.json`.

Точные нормализованные совпадения с train (`manual_train`): **13/139** —
это группы журнала ручной разметки, присутствующие и в train, и в test
(дедупликация журнала глобальная). Строки из оценки не исключались; метрика
на этих группах может быть завышена. Пересечение `case_test` с `case_train`
для официального сценария — 24/208.
`--replay` заменяет `results.json` проверенным пересчётом исторических ответов.

## Исторические результаты

Результаты, измеренные на прежнем `reviewed_test.parquet` (119 строк,
семантические исправления improved_dataset), несовместимы с новым протоколом
`manual_test` и пересчёту не подлежат как результаты этого сценария. Их
исходные значения и происхождение сохранены в `recorded-results.json`
(без изменений) и в истории Git. Новые прогоны API-модулей (`google_decision`,
`jev`, `jev_tfidf`, `qwen_decision`) на `manual_test` — отдельная задача,
в `results.json` они помечены `requires_rerun`.

Прежний эксперимент local_jev на reviewed_test (119 строк), включая полные
предсказания и вероятности, сохранён в истории Git: commit a2893bf,
файл src/subcategory_extraction/benchmarks/improved_dataset/results.json.
Его F1 не переносится на manual_test. После merge новый прогон не выполнялся.
