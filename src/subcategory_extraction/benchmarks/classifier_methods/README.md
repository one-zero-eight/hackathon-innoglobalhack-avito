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
| jev_semantic | 2026-11-15T20:28:13+03:00 | 208/208 | 0.755208 | frozen semantic candidate; private confirmation not used for selection; provenance in JSON |
| jev_semantic_arbiter | 2026-11-15T20:31:42+03:00 | 208/208 | 0.826709 | frozen semantic candidate; private confirmation not used for selection; provenance in JSON |
| jev_semantic_arbiter::kimi-k3.1-pro | 2026-11-15T21:43:15+03:00 | 208/208 | 0.821705 | Kimi K3.1 Pro; fixed semantics, explicit schema; provenance in JSON |

## Протокол и воспроизведение

Kimi K3.1 Pro проверяется фиксированной конфигурацией `jev_semantic_arbiter::kimi-k3.1-pro`:
тот же native Jev, calibration и категорийные правила, другой structured arbiter.
Для совместимости с BestAPI JSON Schema дополнительно передаётся в system message;
одного response_format оказалось недостаточно в синтетических проверках. Строгая
валидация сохраняется. Это сравнение моделей без подбора по private-меткам;
прежние development/confirmation уже просмотрены и не являются новым holdout.
Результаты добавлены в существующие JSON и общие таблицы/графики HTML, GPT-строки сохранены.

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/export_diagnostic.py \
  --dataset private_test --size 0 --skip-latency --workers 8 \
  --configurations jev_semantic_arbiter::kimi-k3.1-pro --report out/kimi-private.json
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py \
  --configuration jev_semantic_arbiter::kimi-k3.1-pro --workers 4 --report out/kimi-official.json
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py \
  --configuration jev_semantic_arbiter::kimi-k3.1-pro --workers 4 --report out/kimi-manual.json
```

F1 переиспользует валидные API-кэши; официальный прогон продолжен после одного
обрыва соединения и двух разных schema-invalid ответов, manual — после одного
schema-invalid ответа. Невалидные строки повторены по одному разу без изменений
параметров, история ошибок сохранена в retry_history. Скорость измеряется отдельно на 10 прежних validation-входах
без кэша и параллельных benchmark-задач. Private использует тот же одобренный снимок262.

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

## Private-test: все шесть категорий без source mcId

[Результаты и график F1 в общем HTML](../response_time/benchmark-report.html#private),
[исходные предсказания, метрики и аудит](private-test-results.json).
Это отдельный протокол; исходная таблица case_test выше сохраняет свои результаты.

262 описания из db_export/private-test-dataset.csv объединяются с метками по testcase.
Пользователь выбрал исходный валидный снимок data/responce.csv, сохранённый в начале задачи:
SHA-256 меток `7474fb50cf5c313ec891279701819d14aecfc7a35079ed644d19979f12abeed2`.
Рабочая копия data/responce.csv менялась во время подготовки и не перезаписывалась.
Снимок находится в out/private-labels-snapshot.csv. SHA-256 входного CSV:
`1ff8928df6d47269346fd98b4f7d2aacb2f6c5d4a7cdbb99604e77340556f25c`.

По решению пользователя source mcId отсутствует: NULL в опциональной таблице private_test,
NoSourceClassificationInput только в benchmark-адаптере. Производственный контракт main.py
не меняется. Никакая категория не назначается искусственно и не исключается из оценки:
101–106 входят в micro-F1. В 114 строках истинная метка включает 101; у части локальных
моделей нет обученного класса 101. Пропуски считаются FN; в JSON есть поклассовые F1
и отдельный диагностический пересчёт тех же ответов на 102–106.

40/262 описаний точно совпадают с case_train/manual_train после нормализации;
222 остальных оцениваются дополнительно. Это не поиск близких дубликатов.
Модели, промпты и пороги заморожены до оценки; переобучения на приватных метках нет.
Shortlist выбран по прежним результатам: regex, tfidf::legacy, morphology_logistic_regression,
snippet_logistic_regression, order_format_scope, order_format_joint::dedup,
semantic_encoder_f1, semantic_evidence, jev, jev_calibrated, jev_tfidf,
google_decision, evidence_jev. Новые веса вместо опубликованных не подставляются.

Google/Selectel/BestAPI получили отдельное разрешение пользователя на private-описания.
Ответы кэшируются штатно; совпавшие ответы Jev переиспользуются hybrid-модулем.
Лимиты и ошибки фиксируются без подстановки пустых ответов и без полной F1 для
незавершённого метода. Common-subset F1 использует одни и те же завершённые строки
для всех выбранных методов; это отдельный, потенциально более простой поднабор.
Полный исходный текст передаётся модели; собственные ограничения модели сохраняются.
Дополнительная latency не измеряется, чтение кэша не выдаётся за скорость внешнего API.

Данные импортируются только явно и только через database.load_private_test().
Для текущего снимка передать labels_path=Path('out/private-labels-snapshot.csv');
без аргумента loader читает текущий data/responce.csv и строго проверяет join.
Рантайм benchmark читает только PostgreSQL через read_records('private_test').

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
RUBERT_F1_ARTIFACT_PATH=src/subcategory_extraction/semantic/semantic_encoder_f1/artifacts \
uv run python src/subcategory_extraction/benchmarks/classifier_methods/export_diagnostic.py \
  --dataset private_test --size 0 --skip-latency --workers 1 \
  --methods lexical.regex lexical.tfidf lexical.morphology_logistic_regression lexical.snippet_logistic_regression lexical.order_format_scope semantic.semantic_encoder_f1 semantic.semantic_evidence \
  --report out/private-local.json
```

order_format_joint::dedup запускается через --configurations с тем же --report.
Для API используются --workers 4 и отдельные out/private-decisions.json / out/private-evidence.json.
Сборка сохранённых ответов без повторных вызовов:

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/export_diagnostic.py \
  --dataset private_test --size 0 --skip-latency \
  --merge-reports out/private-local.json out/private-decisions.json out/private-evidence.json \
  --report src/subcategory_extraction/benchmarks/classifier_methods/private-test-results.json
```

При объединении проверяются table SHA, точные входы/метки, политика source и fingerprint
моделей. Модели не вызываются; таблицы и веса общей БД не менялись.

Исходный shortlist: 12 полных F1 на 262 строках. Google завершил 251/262: девять строк превышают
budget guard клиента, две вернули HTTP 400. Полного F1 Google нет; его common-subset
micro-F1 на общих 251 строках — 0.910037. На том же поднаборе jev_calibrated — 0.929052.
У jev_tfidf и локальных методов нет предсказаний 101 в этом прогоне; 114 пропусков
учтены как FN. Веса и declared class support не исправлялись по приватным меткам.


## Generalization study: jev_semantic / jev_semantic_arbiter

[Фиксация групп](jev-semantic-split.json), [train-only подбор](jev-semantic-training-results.json),
[выбор до confirmation](jev-semantic-selection.json), [итог и ограничения](jev-semantic-study.json).
Результаты private находятся в той же единой HTML-таблице; отдельной таблицы прогонов нет.

96 fit +48 control из manual_train, с исключением групп private и обоих public tests.
Описания всех144 строк побайтно совпадают с публичным case_train; ручные метки не отправлялись.
Выбран task prompt с group-crossfit порогами, подтверждённый на control0.910256 против0.876712.
101 не подбирался (фиксированный0.5). Ни одни private-метки не обучают пороги или веса.

До новых private-прогнозов зафиксировано 211 development /51 confirmation, char-TFIDF
similarity>=0.85, связанные тексты удержаны вместе. Ранее просмотренные первые примеры
и группы train/test оставлены в development. Baseline aggregate уже был известен:
это ограниченное подтверждение, не новый независимый внешний эталон.

Сравнивались заранее объявленные task/strict native prompts и один structured arbiter.
Native prompt выбран только по train crossfit; arbiter выбран по development0.980251
при control не хуже baseline (0.878049 против0.876712). Затем он заморожен до confirmation.
Confirmation0.972222 (105TP/6FP/0FN), full0.978947 (651TP/22FP/6FN),
no-exact-train-match N2220.977561. Group-bootstrap95% interval confirmation:0.928–1.000;
51 строка не доказывает универсальный уровень>0.95. Один сетевой сбой confirmation
восстановлен без изменения модели с переиспользованием50 валидных ответов.

Промпты задают общие границы категорий, текущее предложение, отрицания и ограничения
заказа. В inference нет ID/хешей объявлений, private-файлов, имён или ответов по словарю.
Все262 метки workbook совпадают с исходным approved CSV snapshot; семь длинных текстов
workbook сокращены, поэтому модели получили полный исходный private CSV.

Обнаружен tradeoff: arbiter private лучше, но case_test0.826709 против calibrated0.844376,
manual_test0.715556 против0.723769. Native semantic на manual_test0.790698, но на case_test0.755208.
Baseline/default config не заменён. Новые методы остаются явными experimental selectors.
Arbiter делает Jev+structured-LLM вызовы, свежий validation mean5.320647с против native1.140226с.
Эти числа не являются latency private-набора. Публичные проверки сделаны после заморозки;
137 manual_test текстов совпадают с case_test, ещё2 — с case_train, метки остаются локально.

Воспроизведение:

```bash
PYTHONPATH=src uv run python src/dataset_annotation/prepare_semantic_protocol.py
uv run python src/subcategory_extraction/decisions/jev_semantic/train.py
```

Manifest создаётся один раз и не перерисовывается. Runtime берёт данные только из PostgreSQL.
Для subset-прогонов export_diagnostic.py поддерживает --selection-manifest и --partition
(development/confirmation либо fit/control). Confirmation не используется для новых подборов.
Полный winner-score собран из сохранённых211+51 ответов, без повторных full API-вызовов.
Канонический private JSON содержит разные revision fingerprints по решениям и не служит
checkpoint для нового inference. Для возобновления использовать исходные out/semantic-*.json;
датасет, split и fingerprint проверяются до продолжения.
