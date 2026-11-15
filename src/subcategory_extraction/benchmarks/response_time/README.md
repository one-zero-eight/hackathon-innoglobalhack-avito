# Бенчмарк: время работы классификатора

Настоящие последовательные `run()` на 10 validation-группах официального
train, seed=44. SHA-256 train:
`b7118d68086624fbeab43839f22a635d4e325e4ee339f22d4aa0ad3c2e986615`.
Загрузка модели и данных исключена, первый вызов включён; прогрева нет.

## Результаты

Настоящие последовательные `run()` на штатных 10 validation-входах seed=44.
Свежие замеры 2026-11-15 — Apple Silicon CPU, один поток OpenMP/OpenBLAS
и PyTorch, загрузка модели исключена, первый вызов включён, прогрева и кэша
предсказаний нет. p95 — линейный percentile NumPy.

| Конфигурация | Версия / дата | N | Среднее, с | Медиана, с | p95, с | Происхождение |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tfidf::legacy | out/tfidf.json / 2026-11-15 | 10 | 0.000837 | 0.000714 | 0.001605 | локальный run() |
| tfidf | out/classifier-methods/tfidf.json / 2026-11-15 | 10 | 0.001197 | 0.000742 | 0.003397 | локальный run() |
| learned_keywords | веса d06457a8c082 / 2026-11-15 | 10 | 0.001796 | 0.001509 | 0.003364 | локальный run() |
| regex | — / 2026-11-15 | 10 | 0.005204 | 0.005992 | 0.009410 | локальный run() |
| semantic_encoder_calibrated | out/rubert-thresholds/encoder / 2026-11-15 | 10 | 0.011291 | 0.008832 | 0.023364 | локальный CPU run() |
| semantic_encoder::threshold-encoder | out/rubert-thresholds/encoder / 2026-11-15 | 10 | 0.012444 | 0.009354 | 0.025328 | локальный CPU run() |
| semantic_encoder_f1 | веса 1c2a6f5 / 2026-11-15 | 10 | 0.013947 | 0.010074 | 0.027228 | локальный CPU run() |
| snippet_logistic_regression | model.json.gz / 2026-11-15 | 10 | 0.008473 | 0.006652 | 0.021653 | локальный run() |
| semantic_evidence | selection.json / 2026-11-15 | 10 | 0.032729 | 0.024371 | 0.098837 | локальный CPU run() |
| google_decision | gemini-decision-1 / 2026-11-15 | 10 | 0.801300 | — | — | сетевой API run(), свежий |
| google_decision::baseline | gemini-decision-1 / 2026-11-15 | 10 | 0.821800 | — | — | сетевой API run(), свежий |
| qwen_decision | preview / 2026-11-14 | 10 | 14.309403 | 2.499930 | 62.312641 | сохранённый сетевой API-замер (не тот же прогон, что выше) |
| local_jev / jevos-v5 | OpenVINO INT8 / 2026-11-14 | 10 | 1.199780 | 1.126392 | 2.275740 | сохранённый замер готового сервера (Linux), english / 0.5 |

Свежие 9 строк перемерены на этой машине в одном протоколе (seed=44, size=10).
Строки `qwen_decision` и `local_jev` — исторические замеры других орудий
(внешний API / Linux-сервер), сохранены с исходной датой и происхождением,
их не сравниваем напрямую с локальными CPU-строками.

## Карта

График «Micro-F1 (case_test) × время вызова» объединяет этот сценарий
(время по оси X) с F1 на исходном test — см. [карта](../approaches-map.png).

## Запуск

```bash
TFIDF_ARTIFACT_PATH=out/classifier-methods/tfidf.json uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module lexical.tfidf --size 10
```

Для другого решения замените `--module`. Отчёт — [results.json](results.json).
Повторный запуск заменяет строку выбранного решения, сохраняя остальные;
соответствующую строку README обновляем по JSON. Запуск с другим размером
выборки требует отдельного `--report`, чтобы не смешивать протоколы.

Apple Silicon arm64, CPU. У semantic_encoder_f1 один поток PyTorch,
у learned_keywords один поток OpenMP/OpenBLAS. Кэш предсказаний,
ожидание квоты и повторы отсутствуют; p95 у этих двух измерений — линейный
percentile NumPy. Условия и даты записаны отдельно для каждого решения.
Эти числа не являются SLA и не измеряют качество классификации.


## Выводы

Это время работы конкретного локального запуска, не SLA. Для
интерпретации API-цены см. `../cpu_costs/README.md`.

Исторический замер semantic_encoder_f1 сохранён в `recorded-results.json`
со ссылкой на исходный commit/отчёт. Загрузка исключена; первый вызов включён;
p95 посчитан nearest rank. Его не сравниваем напрямую с выборкой из 10 входов.

## Свежий прогон опубликованных весов Leo

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RUBERT_F1_ARTIFACT_PATH=src/subcategory_extraction/semantic/semantic_encoder_f1/artifacts RUBERT_F1_DEVICE=cpu uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module semantic.semantic_encoder_f1 --size 10
```

Apple Silicon arm64, CPU, один поток PyTorch; версии и платформа — в `results.json`.
Загрузка модели и данных исключена, первый вызов включён, прогрева и кэша
предсказаний нет. Квота и повторы отсутствуют. p95 — линейный percentile NumPy.
Это другая выборка, чем исторические 119 диагностических входов.

## learned_keywords: новый прогон PR #9

2026-11-14, код `3768d76`, база main `e04c97d`; веса и конфигурация
совпадают с двумя F1-сценариями.


```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module lexical.learned_keywords --size 10
```

Apple Silicon arm64, CPU; один поток OpenMP/OpenBLAS. Загрузка модели и данных
исключена, первый run() включён; прогрева, кэша предсказаний, квоты и повторов нет.
p95 — линейный percentile NumPy. Выборка: 10 validation-групп официального train
с seed=44; для этой модели это замер задержки, а не оценка на отложенных данных.
Сохранён прежний замер semantic_encoder_f1 с исходным происхождением.

## Qwen decision-model-preview: 2026-11-14

База `45ff38d`, код `3946999259f93a4db45e354a06a12e6cbff64778b66253c27da547e540e90c0f`.
Те же замороженные `boundaries` и train-пороги, что в двух F1-сценариях.


```bash
QWEN_DECISION_CACHE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module decisions.qwen_decision --size 10 --append
```

Настоящие последовательные `run()` на штатных 10 входах seed=44. Чтение и запись
кэша Qwen отключены, прогрева нет, первый вызов включён. Создание экземпляра
и загрузка данных исключены. Сетевое соединение, ожидание API и возможные
повторы входят во время; в этом прогоне 10 HTTP-попыток на 10 успешных вызовов,
программных повторов не было. Отдельного ожидания квоты нет.
Два вызова заняли примерно 62 секунды: это клиентское время по сети,
не чистая задержка вычисления модели. p95 — линейный percentile NumPy.
Вычислительное оборудование провайдера неизвестно.

Локальный baseline заново выполнен в этом Linux-окружении, отдельно от API,
с одним потоком OpenMP/OpenBLAS и без кэша предсказаний. Старые замеры
сохранены со своими датами; времена Apple Silicon выше не являются новым
Linux-замером. У Qwen N=10, поэтому оценку нельзя трактовать как SLA.

## Локальный feder-cr/jev: 2026-11-14

Те же jevos-v5 INT8, `english`, порог 0.5, что в двух F1-сценариях.
Настоящие последовательные `run()` на штатных 10 validation-входах seed=44.
F1 здесь не считается. Оба модуля измерены заново на одной машине, по очереди,
после завершения F1-прогонов, без параллельных запросов к серверу.


```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module decisions.local_jev --size 10 --append
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module lexical.learned_keywords --size 10 --append
```

Linux x86_64, Intel Core i9-12900H. Jevos: четыре потока OpenVINO;
learned_keywords: один поток OpenMP/OpenBLAS. Сервер запущен с
`--threads 4 --warmup 0 --state-cache 0 --batch-tokens 0`, dynamic quantization=128.
Версия runtime, веса и fingerprint — в `model_provenance` отчёта.
Старт сервера/загрузка весов, создание экземпляра клиента и health исключены;
первый измеряемый run() включён. Сервер до замера уже обрабатывал F1-прогоны,
поэтому это замер готового сервера, а не холодного старта.

Кэш ответов клиента и state-cache сервера отключены, попаданий нет.
Время включает localhost HTTP, токенизацию, вычисления и разбор результата.
Квоты, ожидания внешнего API и повторы отсутствуют: 10 HTTP-попыток на 10 входов.
p95 — линейный percentile NumPy. Средний local_jev примерно в 564 раза медленнее
этого локального baseline. N=10 не даёт SLA. Исторические времена Apple Silicon
и внешнего Qwen выше не являются замерами на этом оборудовании.
