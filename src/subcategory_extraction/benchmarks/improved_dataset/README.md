# Micro-F1: manual_test

N=139; исходная mcId=101, целевые категории 102–106. Данные читаются через read_records() из PostgreSQL.
Точный нормализованный overlap с manual_train: 13/139; строки оставлены в оценке.
SHA-256 таблицы: `fae05cba56fdf57a15aa3979afd828b73928efcca64694be5d805d34e002b99d`.

[Единый автономный HTML: все датасеты, графики и подробности](../response_time/benchmark-report.html).

## Результаты

| Конфигурация | Дата измерения | N | micro-F1 | Происхождение / ошибка |
| --- | --- | ---: | ---: | --- |
| local_jev (guided_ru, threshold=0.5) | 2026-11-15T04:36:26+03:00 | 0/139 | — | RuntimeError: Автозагрузка runtime local_jev поддерживает Linux x86_64; на другой ОС запустите официальный сервер отдельно |
| tfidf | 2026-11-15T02:07:12+03:00 | 139/139 | 0.681128 | сохранённый вариант; собственное происхождение в JSON |
| semantic_encoder::threshold-encoder | 2026-11-15T02:07:46+03:00 | 139/139 | 0.756856 | сохранённый вариант; собственное происхождение в JSON |
| google_decision::baseline | 2026-11-15T02:49:26+03:00 | 139/139 | 0.787671 | сохранённый вариант; собственное происхождение в JSON |
| jev::baseline | 2026-11-15T04:15:35+03:00 | 0/139 | — | ValueError: Укажите SELECTEL_API_KEY в .env для запросов Jev |
| jev_tfidf::baseline | 2026-11-15T04:19:31+03:00 | 0/139 | — | ValueError: Укажите SELECTEL_API_KEY в .env для запросов Jev |
| jev_retrieval::baseline | 2026-11-15T04:23:27+03:00 | 0/139 | — | FileNotFoundError: [Errno 2] No such file or directory: '/Users/arsenijsekin/Documents/Codex/2026-11-13/https-github-com-one-zero-eight/hackathon-innoglobalhack |
| jev_arbiter::baseline | 2026-11-15T04:29:01+03:00 | 0/139 | — | ValueError: Укажите SELECTEL_API_KEY в .env для запросов Jev |
| qwen_decision::guided | 2026-11-15T04:30:59+03:00 | 0/139 | — | ValueError: Задайте QIANWEN_TOKEN_PLAN_API_KEY в .env |
| qwen_decision::boundaries | 2026-11-15T04:33:01+03:00 | 0/139 | — | ValueError: Задайте QIANWEN_TOKEN_PLAN_API_KEY в .env |
| regex | 2026-11-15T11:00:42+03:00 | 139/139 | 0.768362 | прогон сравнения; собственное происхождение в JSON |
| tfidf::legacy | 2026-11-15T11:00:42+03:00 | 139/139 | 0.695096 | прогон сравнения; собственное происхождение в JSON |
| learned_keywords | 2026-11-15T11:00:42+03:00 | 139/139 | 0.730697 | прогон сравнения; собственное происхождение в JSON |
| snippet_logistic_regression | 2026-11-15T11:00:42+03:00 | 139/139 | 0.759295 | прогон сравнения; собственное происхождение в JSON |
| semantic_encoder | 2026-11-15T11:00:42+03:00 | 0/139 | — | ValueError: Нет локального артефакта semantic_encoder: /home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/semantic-encoder/metadata.json. Обучен |
| semantic_encoder_calibrated | 2026-11-15T11:00:42+03:00 | 0/139 | — | ValueError: Нет локального артефакта semantic_encoder: /home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/semantic-encoder/metadata.json. Обучен |
| semantic_encoder_f1 | 2026-11-15T11:00:42+03:00 | 139/139 | 0.770093 | прогон сравнения; собственное происхождение в JSON |
| semantic_evidence | 2026-11-15T11:00:42+03:00 | 139/139 | 0.774674 | прогон сравнения; собственное происхождение в JSON |
| jev | 2026-11-15T11:02:23+03:00 | 139/139 | 0.757202 | прогон сравнения; собственное происхождение в JSON |
| jev_calibrated | 2026-11-15T11:02:23+03:00 | 139/139 | 0.723769 | прогон сравнения; собственное происхождение в JSON |
| jev_boundaries | 2026-11-15T11:02:23+03:00 | 139/139 | 0.726496 | прогон сравнения; собственное происхождение в JSON |
| jev_retrieval | 2026-11-15T11:02:23+03:00 | 0/139 | — | FileNotFoundError: [Errno 2] No such file or directory: '/home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/hybrid-training.jsonl' |
| qwen_decision | 2026-11-15T11:02:23+03:00 | 139/139 | 0.668810 | прогон сравнения; собственное происхождение в JSON |
| jev_tfidf | 2026-11-15T11:02:23+03:00 | 139/139 | 0.753623 | прогон сравнения; собственное происхождение в JSON |
| jev_arbiter | 2026-11-15T11:11:16+03:00 | 139/139 | 0.735043 | прогон сравнения; собственное происхождение в JSON |
| evidence_jev | 2026-11-15T11:11:16+03:00 | 139/139 | 0.758763 | прогон сравнения; собственное происхождение в JSON |
| google_decision | 2026-11-15T11:11:15+03:00 | 139/139 | 0.737945 | прогон сравнения; собственное происхождение в JSON |
| local_jev | 2026-11-15T11:09:41+03:00 | 139/139 | 0.232558 | прогон сравнения; собственное происхождение в JSON |
| morphology_logistic_regression | 2026-11-15T14:16:46+03:00 | 139/139 | 0.723849 | новый локальный run(); опубликованные веса |
| morphology_logistic_regression::yandex-youla-decision1 | 2026-11-15T14:16:48+03:00 | 139/139 | 0.731405 | новый локальный run(); опубликованные веса |
| order_format_logistic_regression | 2026-11-15T14:16:50+03:00 | 139/139 | 0.672489 | новый локальный run(); опубликованные веса |
| order_format_phrase_logreg | 2026-11-15T14:16:52+03:00 | 139/139 | 0.755020 | новый локальный run(); опубликованные веса |
| order_format_scope | 2026-11-15T14:16:55+03:00 | 139/139 | 0.762475 | новый локальный run(); опубликованные веса |
| order_format_veto | 2026-11-15T14:16:58+03:00 | 139/139 | 0.754639 | новый локальный run(); опубликованные веса |
| order_format_joint | 2026-11-15T14:17:00+03:00 | 139/139 | 0.668203 | новый локальный run(); опубликованные веса |
| order_format_joint::dedup | 2026-11-15T14:17:03+03:00 | 139/139 | 0.671171 | новый локальный run(); опубликованные веса |
| order_format_joint::yandex_youla | 2026-11-15T14:17:05+03:00 | 139/139 | 0.669725 | новый локальный run(); опубликованные веса |

## Протокол и воспроизведение

Запуск 15 ноября 2026: 24 модуля, включая шесть новых logistic/order-format модулей и девять опубликованных конфигураций.
Модели и пороги зафиксированы, обучения и настройки в этом запуске нет. Старые варианты сохраняют свою дату и машину.
У semantic_encoder и semantic_encoder_calibrated нет весов, у jev_retrieval нет подготовленного индекса; F1 не подставлялся.
manual_test имеет общие train-группы. Автор order_format_joint описывает использование примеров official test при разработке;
его результат не следует считать независимой оценкой. Provider-веса — отдельные явно названные конфигурации.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py \
  --configuration morphology_logistic_regression --offline --append
```

Для остальных вариантов изменить --configuration (список в configurations.py). Каждый повторный запуск заменяет свой ключ.
[results.json](results.json) содержит предсказания, настройки, хеши артефактов и происхождение каждой строки.
[recorded-results.json](recorded-results.json) не изменён. Сохранённые API-ответы могут использоваться для F1;
latency всегда измеряется отдельно настоящими вызовами, а не чтением ответов.
