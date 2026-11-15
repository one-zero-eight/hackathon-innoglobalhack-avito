# Micro-F1: case_test

N=208; исходная mcId=101, целевые категории 102–106. Данные читаются через read_records() из PostgreSQL.
Точный нормализованный overlap с case_train: 24/208; строки оставлены в оценке.
SHA-256 таблицы: `cd6f432e3841e842d04bb07827b92f9b644c56b2b528cb36098433168890ae15`.

[Единый автономный HTML: все датасеты, графики и подробности](../response_time/benchmark-report.html).

## Результаты

| Конфигурация | Дата измерения | N | micro-F1 | Происхождение / ошибка |
| --- | --- | ---: | ---: | --- |
| local_jev (guided_ru, threshold=0.5) | 2026-11-14T23:14:37+03:00 | 208/208 | 0.000000 | сохранённый вариант; собственное происхождение в JSON |
| jev::baseline | 2026-11-15T01:47:08+03:00 | 0/208 | — | ValueError: Укажите BESTAPI_API_KEY в .env для запросов Jev |
| jev_tfidf::baseline | 2026-11-15T01:47:08+03:00 | 0/208 | — | ValueError: Укажите BESTAPI_API_KEY в .env для запросов Jev |
| jev_retrieval::baseline | 2026-11-15T01:47:08+03:00 | 0/208 | — | FileNotFoundError: [Errno 2] No such file or directory: '/Users/arsenijsekin/Documents/Codex/2026-11-13/https-github-com-one-zero-eight/hackathon-innoglobalhack |
| jev_arbiter::baseline | 2026-11-15T01:47:08+03:00 | 0/208 | — | ValueError: Укажите BESTAPI_API_KEY в .env для запросов Jev |
| qwen_decision::guided | 2026-11-15T01:47:08+03:00 | 0/208 | — | ValueError: Задайте QIANWEN_TOKEN_PLAN_API_KEY в .env |
| qwen_decision::boundaries | 2026-11-15T01:47:08+03:00 | 0/208 | — | ValueError: Задайте QIANWEN_TOKEN_PLAN_API_KEY в .env |
| tfidf | 2026-11-15T02:05:12+03:00 | 208/208 | 0.778462 | сохранённый вариант; собственное происхождение в JSON |
| semantic_encoder::threshold-encoder | 2026-11-15T02:06:08+03:00 | 208/208 | 0.728411 | сохранённый вариант; собственное происхождение в JSON |
| google_decision::baseline | 2026-11-15T02:37:40+03:00 | 208/208 | 0.738462 | сохранённый вариант; собственное происхождение в JSON |
| regex | 2026-11-15T10:59:58+03:00 | 208/208 | 0.751979 | прогон сравнения; собственное происхождение в JSON |
| tfidf::legacy | 2026-11-15T10:59:58+03:00 | 208/208 | 0.795217 | прогон сравнения; собственное происхождение в JSON |
| learned_keywords | 2026-11-15T10:59:58+03:00 | 208/208 | 0.745358 | прогон сравнения; собственное происхождение в JSON |
| snippet_logistic_regression | 2026-11-15T10:59:58+03:00 | 208/208 | 0.763194 | прогон сравнения; собственное происхождение в JSON |
| semantic_encoder | 2026-11-15T10:59:58+03:00 | 0/208 | — | ValueError: Нет локального артефакта semantic_encoder: /home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/semantic-encoder/metadata.json. Обучен |
| semantic_encoder_calibrated | 2026-11-15T10:59:58+03:00 | 0/208 | — | ValueError: Нет локального артефакта semantic_encoder: /home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/semantic-encoder/metadata.json. Обучен |
| semantic_encoder_f1 | 2026-11-15T10:59:58+03:00 | 208/208 | 0.769433 | прогон сравнения; собственное происхождение в JSON |
| semantic_evidence | 2026-11-15T10:59:58+03:00 | 208/208 | 0.764314 | прогон сравнения; собственное происхождение в JSON |
| local_jev | 2026-11-15T11:05:52+03:00 | 208/208 | 0.399050 | прогон сравнения; собственное происхождение в JSON |
| google_decision | 2026-11-15T11:08:58+03:00 | 208/208 | 0.839394 | прогон сравнения; собственное происхождение в JSON |
| jev | 2026-11-15T11:00:21+03:00 | 208/208 | 0.842415 | прогон сравнения; собственное происхождение в JSON |
| jev_calibrated | 2026-11-15T11:00:21+03:00 | 208/208 | 0.844376 | прогон сравнения; собственное происхождение в JSON |
| jev_boundaries | 2026-11-15T11:00:21+03:00 | 208/208 | 0.836141 | прогон сравнения; собственное происхождение в JSON |
| jev_retrieval | 2026-11-15T11:00:21+03:00 | 0/208 | — | FileNotFoundError: [Errno 2] No such file or directory: '/home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/hybrid-training.jsonl' |
| qwen_decision | 2026-11-15T11:00:21+03:00 | 208/208 | 0.629674 | прогон сравнения; собственное происхождение в JSON |
| jev_tfidf | 2026-11-15T11:00:21+03:00 | 208/208 | 0.846266 | прогон сравнения; собственное происхождение в JSON |
| jev_arbiter | 2026-11-15T11:10:24+03:00 | 208/208 | 0.839204 | прогон сравнения; собственное происхождение в JSON |
| evidence_jev | 2026-11-15T11:00:21+03:00 | 208/208 | 0.840708 | прогон сравнения; собственное происхождение в JSON |
| morphology_logistic_regression | 2026-11-15T14:16:15+03:00 | 208/208 | 0.823881 | новый локальный run(); опубликованные веса |
| morphology_logistic_regression::yandex-youla-decision1 | 2026-11-15T14:16:18+03:00 | 208/208 | 0.814706 | новый локальный run(); опубликованные веса |
| order_format_logistic_regression | 2026-11-15T14:16:20+03:00 | 208/208 | 0.698171 | новый локальный run(); опубликованные веса |
| order_format_phrase_logreg | 2026-11-15T14:16:24+03:00 | 208/208 | 0.757576 | новый локальный run(); опубликованные веса |
| order_format_scope | 2026-11-15T14:16:28+03:00 | 208/208 | 0.762689 | новый локальный run(); опубликованные веса |
| order_format_veto | 2026-11-15T14:16:32+03:00 | 208/208 | 0.742446 | новый локальный run(); опубликованные веса |
| order_format_joint | 2026-11-15T14:16:36+03:00 | 208/208 | 0.794658 | новый локальный run(); опубликованные веса |
| order_format_joint::dedup | 2026-11-15T14:16:39+03:00 | 208/208 | 0.779051 | новый локальный run(); опубликованные веса |
| order_format_joint::yandex_youla | 2026-11-15T14:16:43+03:00 | 208/208 | 0.778523 | новый локальный run(); опубликованные веса |

## Протокол и воспроизведение

Запуск 15 ноября 2026: 24 модуля, включая шесть новых logistic/order-format модулей и девять опубликованных конфигураций.
Модели и пороги зафиксированы, обучения и настройки в этом запуске нет. Старые варианты сохраняют свою дату и машину.
У semantic_encoder и semantic_encoder_calibrated нет весов, у jev_retrieval нет подготовленного индекса; F1 не подставлялся.
manual_test имеет общие train-группы. Автор order_format_joint описывает использование примеров official test при разработке;
его результат не следует считать независимой оценкой. Provider-веса — отдельные явно названные конфигурации.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py \
  --configuration morphology_logistic_regression --offline --append
```

Для остальных вариантов изменить --configuration (список в configurations.py). Каждый повторный запуск заменяет свой ключ.
[results.json](results.json) содержит предсказания, настройки, хеши артефактов и происхождение каждой строки.
[recorded-results.json](recorded-results.json) не изменён. Сохранённые API-ответы могут использоваться для F1;
latency всегда измеряется отдельно настоящими вызовами, а не чтением ответов.

## Диагностика экспорта и IGH

Четыре case/manual CSV побайтно сохраняют исходные описания и совпадают с таблицами прежних прогонов по каноническим хешам.
Аудит: [export-comparison.json](export-comparison.json). При CSV-импорте нужно сохранять переводы строк: open(newline="").

IGH: 4966 уникальных описаний, исходная mcId отсутствует (принята 101), 4840 точных совпадений с train.
Это объединённый корпус, не независимый test. В общей выборке 300 (seed=20261115) 289 train-совпадений.
Full: 15 доступных быстрых локальных конфигураций; API/local_jev ограничены sample по выбору пользователя.
Незавершённый full local_jev не получает F1. Диагностические результаты раздельны:
[full JSON](igh-full-results.json), [sample JSON](igh-sample-results.json), [аудит](igh-audit.json).
rows_without_exact_train_match — пересчёт тех же предсказаний на 126 full / 11 sample строках; близкие дубликаты не исключены.

Yandex/Youla: 2961 строка, из них 2960 явно размечены; row_id=532 без разметки исключён.
Общая выборка 300, seed=20261115, 159 Юла + 141 Яндекс. Метки автоматические (Gemini Decision 1).
Считается agreement F1 только по 102–106: loader теряет целевую 101. Source mcId исключается по контракту.
Google F1=1.0 — согласие с собственным разметчиком. Provider-trained варианты могут пересекаться с этой выборкой.
[provider JSON](export-diagnostic-results.json). evidence_jev: невалидная цитата в одной строке, 299/300, полного F1 нет.

```bash
uv run python src/database_load.py --tables case_train case_test manual_train manual_test igh_deduped_no_conflict
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/classifier_methods/export_diagnostic.py \
  --dataset igh_deduped_no_conflict --size 0 --configurations morphology_logistic_regression morphology_logistic_regression::yandex-youla-decision1 order_format_logistic_regression order_format_phrase_logreg order_format_scope order_format_veto order_format_joint order_format_joint::dedup order_format_joint::yandex_youla \
  --report out/logreg-new-full.json
```

Sample: --size 300 --reuse-report out/logreg-new-full.json и отдельный --report. Provider: --dataset yandex_yoola --size 300.
Новые конфигурации сохраняются по отдельным ключам; объединять разрешено только совпадающие rows и table_sha256.
Файл db_export/igh_deduped_no_conflict.jsonl импортируется явно; default loader по-прежнему загружает прежние пять таблиц.
