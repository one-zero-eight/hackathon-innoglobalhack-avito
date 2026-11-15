# Время настоящих extraction-вызовов

[Единый HTML: графики время–F1, все таблицы и источники](benchmark-report.html).

10 одинаковых validation-входов case_train, seed=44. Intel Core i9-12900H, Linux x86_64.
OpenMP/OpenBLAS/MKL/PyTorch: один поток; local_jev: четыре OpenVINO-потока.
Загрузка модели/данных исключена, первый вызов включён. Отдельного прогрева нет.
API-кэши, включая вложенные, пустые; ожидания сети/квоты и повторы включены.
Прежние F1-задачи могли идти параллельно; девять новых logistic-конфигураций измерены последовательно.
p95: линейный percentile NumPy. N=10 не задаёт SLA. Собственные даты/машины прежних вариантов сохранены.

| Конфигурация | Дата измерения | N | Среднее, с | Медиана, с | p95, с | Происхождение / ошибка |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tfidf | 2026-11-15T02:00:50+03:00 | 10 | 0.001197 | 0.000742 | 0.003397 | сохранённый вариант; своя машина/дата |
| semantic_encoder::threshold-encoder | 2026-11-15T02:01:30+03:00 | 10 | 0.012444 | 0.009354 | 0.025328 | сохранённый вариант; своя машина/дата |
| google_decision::baseline | 2026-11-15T02:53:51+03:00 | 10 | 0.821805 | 0.762775 | 1.197284 | сохранённый вариант; своя машина/дата |
| regex | 2026-11-15T11:02:28+03:00 | 10 | 0.004984 | 0.004765 | 0.009641 | run() сравнения; происхождение в JSON |
| tfidf::legacy | 2026-11-15T11:02:30+03:00 | 10 | 0.001243 | 0.001286 | 0.001654 | run() сравнения; происхождение в JSON |
| learned_keywords | 2026-11-15T11:02:32+03:00 | 10 | 0.002048 | 0.002004 | 0.002621 | run() сравнения; происхождение в JSON |
| snippet_logistic_regression | 2026-11-15T11:02:34+03:00 | 10 | 0.008678 | 0.007689 | 0.020155 | run() сравнения; происхождение в JSON |
| semantic_encoder | 2026-11-15T11:02:37+03:00 | 0 | — | — | — | ValueError: Нет локального артефакта semantic_encoder: /home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/semantic-encoder/metadata.json. Обучен |
| semantic_encoder_calibrated | 2026-11-15T11:02:41+03:00 | 0 | — | — | — | ValueError: Нет локального артефакта semantic_encoder: /home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/semantic-encoder/metadata.json. Обучен |
| semantic_encoder_f1 | 2026-11-15T11:02:45+03:00 | 10 | 0.048889 | 0.031275 | 0.115185 | run() сравнения; происхождение в JSON |
| semantic_evidence | 2026-11-15T11:02:50+03:00 | 10 | 0.072009 | 0.063279 | 0.155704 | run() сравнения; происхождение в JSON |
| local_jev | 2026-11-15T11:02:55+03:00 | 10 | 1.355952 | 1.292797 | 2.591015 | run() сравнения; происхождение в JSON |
| google_decision | 2026-11-15T11:08:09+03:00 | 10 | 2.559258 | 0.958264 | 9.795197 | run() сравнения; происхождение в JSON |
| jev | 2026-11-15T11:08:36+03:00 | 10 | 1.139959 | 1.063822 | 1.528407 | run() сравнения; происхождение в JSON |
| jev_calibrated | 2026-11-15T11:08:50+03:00 | 10 | 1.063949 | 1.011052 | 1.514990 | run() сравнения; происхождение в JSON |
| jev_boundaries | 2026-11-15T11:09:03+03:00 | 10 | 1.052811 | 1.001100 | 1.345090 | run() сравнения; происхождение в JSON |
| jev_retrieval | 2026-11-15T11:09:15+03:00 | 0 | — | — | — | FileNotFoundError: [Errno 2] No such file or directory: '/home/zolotarev/PycharmProjects/hackathon-innoglobalhack-avito/out/hybrid-training.jsonl' |
| qwen_decision | 2026-11-15T11:09:17+03:00 | 10 | 3.724876 | 2.076305 | 11.574989 | run() сравнения; происхождение в JSON |
| jev_tfidf | 2026-11-15T11:09:57+03:00 | 10 | 0.974172 | 1.001609 | 1.141514 | run() сравнения; происхождение в JSON |
| jev_arbiter | 2026-11-15T11:10:09+03:00 | 10 | 2.709684 | 1.388549 | 5.868565 | run() сравнения; происхождение в JSON |
| evidence_jev | 2026-11-15T11:10:38+03:00 | 10 | 5.935213 | 6.251561 | 7.149902 | run() сравнения; происхождение в JSON |
| morphology_logistic_regression | 2026-11-15T14:17:08+03:00 | 10 | 0.002344 | 0.001775 | 0.006043 | новый локальный run(); без кэша ответов |
| morphology_logistic_regression::yandex-youla-decision1 | 2026-11-15T14:17:11+03:00 | 10 | 0.001971 | 0.001617 | 0.004520 | новый локальный run(); без кэша ответов |
| order_format_logistic_regression | 2026-11-15T14:17:13+03:00 | 10 | 0.007932 | 0.006122 | 0.020111 | новый локальный run(); без кэша ответов |
| order_format_phrase_logreg | 2026-11-15T14:17:15+03:00 | 10 | 0.009294 | 0.007274 | 0.022917 | новый локальный run(); без кэша ответов |
| order_format_scope | 2026-11-15T14:17:18+03:00 | 10 | 0.007487 | 0.005639 | 0.019572 | новый локальный run(); без кэша ответов |
| order_format_veto | 2026-11-15T14:17:20+03:00 | 10 | 0.007799 | 0.006393 | 0.018719 | новый локальный run(); без кэша ответов |
| order_format_joint | 2026-11-15T14:17:23+03:00 | 10 | 0.007634 | 0.006158 | 0.018676 | новый локальный run(); без кэша ответов |
| order_format_joint::dedup | 2026-11-15T14:17:25+03:00 | 10 | 0.005774 | 0.004420 | 0.014275 | новый локальный run(); без кэша ответов |
| order_format_joint::yandex_youla | 2026-11-15T14:17:27+03:00 | 10 | 0.004832 | 0.004690 | 0.009675 | новый локальный run(); без кэша ответов |
| jev_semantic | 2026-11-15T20:33:28+03:00 | 10 | 1.140226 | 1.119354 | 1.433347 | fresh run(); loading excluded; response-cache hits0 |
| jev_semantic_arbiter | 2026-11-15T20:33:41+03:00 | 10 | 5.320647 | 5.683063 | 5.893055 | fresh run(); loading excluded; response-cache hits0 |

## Воспроизведение

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py \
  --configuration morphology_logistic_regression --size 10 --append

uv run --with matplotlib python src/subcategory_extraction/benchmarks/response_time/report.py
```

HTML строится офлайн из JSON пяти dataset-scope и response_time; БД/API не нужны.
Он содержит SVG, таблицы, настройки и аудит; умеет выгружать график SVG и сводный JSON.
Исходные results.json / recorded-results.json сохранены для пересчёта, отдельные дубли PNG/SVG/CSV удалены.
Для official/manual графиков соединяются только строки с одинаковыми run_id и comparison_configuration_sha256.
Время разных diagnostic scope берётся из собственных latency-прогонов; оно не подменяется validation-временем.
