# Бенчмарк: девять методов классификации

Сервис: subcategory_extraction  
Дата: 2026-11-14  Автор: leo  
Ветка: `leo/feature/classifier-methods`, база GitHub `cd0f310`.

## Архитектура после синхронизации с main

Перенос выполнен на `origin/main` **0f2b9b7**, merge **8d3b5d7**.
Ниже первоначальный эксперимент сохранён как история, не как новая оценка.
Каждый из девяти вариантов — `src/subcategory_extraction/<module>/pipeline.py`
с классом `SubcategoryExtractionPipeline(BasePipeline)` и методом
`run(inp: PipelineInput) -> list[int]`. Выбор — `config.json`;
`main.py` и внешний контракт не менялись при переносе.

- `run.py` — общий протокол, загрузка модулей через `load_pipeline`,
  аудит пересечений, метрики, статусы и фактические замеры `run()`.
- `publish.py` — проверка и объединение полных отчётов одного протокола;
  требует также локальный `out/semantic-encoder/train-report.json`.
- `results.json` — новый прогон; `recorded-results.json` — неизменённый
  исторический отчёт; `google-guided-public.json` — реальные ответы для replay.
- Обучение: `../../jev_tfidf/train.py`, `../../semantic_encoder/train.py`.
  Подготовка filtered train остаётся частью обучения, не оценки.
- Промпты и описание каждого решения находятся в его модуле.
  Guided-промпт Jev переиспользует `google_decision/prompts/`, без копирования.

## Проверка после переноса — 2026-11-14

```bash
PYTHONPATH=src uv run python -m subcategory_extraction.jev_tfidf.train --artifact out/classifier-methods/tfidf.json --report out/classifier-methods/training-results.json
TFIDF_ARTIFACT_PATH=out/classifier-methods/tfidf.json uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --offline --training-dataset out/classifier-methods/hybrid-training.jsonl
```

Обучение заново дало 4821 описание и пять голов. Test и протокол ниже не менялись.
Отпечаток кода прогона: `81d2603af5e890a81da1d741d9cd26137aa6f5aa6aa9bbc587d247e954b32f2e`.
Полный отчёт: [results.json](results.json). Код выхода 1 означает неполное
сравнение; для недоступных методов F1 не рассчитывается.

| Сценарий | Версия / дата | Модуль / модель | Лучший промпт | micro-F1 | N | Среднее / медиана / p95, с | Стоимость |
| --- | --- | --- | --- | --- | --- | --- | --- |
| public test, replay | 8d3b5d7 + отпечаток выше / 2026-11-14 | google_decision / gemini-decision-1 | guided, исторический; не новый выбор | 0.839394 | 208 строк | — (replay, не время API) | бесплатно, без новых API-вызовов |
| public test, локальный | та же версия / 2026-11-14 | tfidf / LogisticRegression | — | 0.778462 | 139 вызовов / 208 строк | 0.003210 / 0.001724 / 0.004454 | бесплатно |
| public test | та же версия / 2026-11-14 | jev, jev_tfidf, jev_retrieval, evidence_jev, jev_arbiter, jev_boundaries | не оценивался | — | 0 | — | бесплатно, API отключён |
| public test | та же версия / 2026-11-14 | semantic_encoder / RuBERT-tiny2 | — | — | 0 | — | бесплатно, локальные веса отсутствуют |

Инициализация и обучение не входят во время `run()`. 139 уникальных входов
разворачиваются в 208 строк для F1. В этом прогоне API-запросов нет.
Флаг `--offline` явно блокирует API-методы, но не подменяет их replay Jev.

### Промпт — результат нового прогона

| Сценарий | Модуль / модель | Промпт (файл / версия) | Версия кода / дата | micro-F1 | N | Среднее / p95, с | Стоимость |
| --- | --- | --- | --- | --- | --- | --- | --- |
| public test, replay | google_decision / gemini-decision-1 | `google_decision/prompts/google_decision_guided.md`, hash проверяется по записи | 8d3b5d7 + отпечаток / 2026-11-14 | 0.839394 | 208 строк | — | бесплатно |

Новый выбор лучшего промпта не выполнялся. Результат guided исторический,
пересчитанный по сохранённым вероятностям, а не новый вызов модели.

## Статус эксперимента

Все пять новых методов реализованы как рабочие модули с общим интерфейсом
`SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`. Encoder реально
дообучен и измерен. Четыре новых API-метода были запущены, но не завершили оценку:
предоставленный BestAPI-ключ стал отклоняться с HTTP 401. Запрос списка моделей
также вернул `{"detail":"invalid api key"}`; процесс читал тот же ключ, что `.env`.
Новый ключ не предоставлен, повторные бессмысленные запросы остановлены.

**Полное сравнение качества всех девяти вариантов не завершено.** Отсутствующая
метрика — не нулевой F1. Не использованы фиктивные вероятности, молчаливый
baseline fallback или другая модель под именем Jev. В JSON стоит
`comparison_complete: false`, а статусы и ошибки сохранены для каждого метода.

## Методы

| Модуль | Реализация | Подробное описание |
| --- | --- | --- |
| `google_decision` | Нативный Google Decisions API, baseline/guided | [Модуль](../../google_decision/pipeline.py) |
| `jev` | Нативный System One API, guided | [Модуль](../../jev/README.md) |
| `tfidf` | Word TF-IDF + LogisticRegression, JSON-артефакт | [Качество](../tfidf_quality/README.md) |
| `jev_tfidf` | Фиксированное среднее вероятностей Jev и TF-IDF | [Гибрид](../jev_tfidf/README.md) |
| `jev_retrieval` | Jev + ближайшие положительный и отрицательный полные train-примеры для каждой категории | [RAG](../../jev_retrieval/README.md) |
| `evidence_jev` | `gpt-6.1-terra` извлекает дословные цитаты, затем Jev оценивает полный исходник с этими подсказками | [Evidence → Jev](../../evidence_jev/README.md) |
| `jev_arbiter` | Сильная LLM разрешает споры Jev/TF-IDF и неопределённые решения Jev | [Арбитр](../../jev_arbiter/README.md) |
| `semantic_encoder` | Реально дообученный RuBERT-tiny2 с multilabel-головой и тремя окнами текста | [Encoder](../../semantic_encoder/README.md) |
| `jev_boundaries` | Jev с уточнёнными границами категорий по официальному справочнику | [Границы](../../jev_boundaries/README.md) |

Базовые варианты: Google Decisions с существующим guided-промптом, нативный Jev,
word TF-IDF + LogisticRegression и заранее фиксированное усреднение Jev/TF-IDF.
Тексты исходного guided и промптов методов сохранены в соответствующих модулях;
изменения не подбирались по ошибкам текущего test.

## Данные и зафиксированный протокол

```json
{
  "dataset": "docs/innoglobalhack_avito/public-test-dataset.parquet",
  "rows": 208,
  "source_mc_id_assumption": 101,
  "training_dataset": "out/hybrid-training.jsonl",
  "training_unique_descriptions": 4821,
  "threshold": 0.5,
  "jev_tfidf_weight": 0.5,
  "arbiter_uncertainty_interval": [0.30, 0.70],
  "rag_max_reference_chars": 1800,
  "validation_used": false,
  "test_parameter_search_used": false
}
```

SHA-256 test: `65b5a45a98b577d853a928932bbbdf8bd9bdeb469ded3016faace4d11f3eabaa`.
Обучающие данные получены только из официального `public-train-annotated.jsonl`:
после нормализованной дедупликации и удаления train-компонент, близких к test по
char TF-IDF cosine >=0.85, остаётся 4821 описание. Исключение читает только тексты
test, не его метки. На оценке сохранены **все 208 строк**, включая противоречивые
дубликаты. В test нет source-category поля, поэтому 101 — явное допущение.

Для экономии API и постоянства решений одинаковые `(description, mcId)`
обрабатываются один раз: 139 уникальных входов. Ответы разворачиваются обратно
на все исходные строки; это не удаление дубликатов из метрики. Локальный аудит
фактического train не обнаружил нормализованных точных пересечений с test.

Метрика — `run.calculate_metrics`, то есть sklearn `f1_score(average="micro",
zero_division=0)` по всем шести категориям. Баллы классификации равны `60 * F1`.
Генерация черновиков, её качество и итоговые 100 баллов здесь не оцениваются.

Google baseline — **replay реальных ранее полученных native Decisions API
ответов**, не новый вызов и не предсказания, созданные вручную. Перед replay
проверены SHA-256 всего test, guided-промпта, каждого описания и точный набор
категорий. Запись находится в `results/google-guided-public.json`; mixed/Purple
наборы и прежние validation/holdout-выборки не используются.

## Исторические результаты до переноса

| Метод | Статус | micro-F1 | Баллы / 60 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Google Decisions guided | Реальные сохранённые API-ответы | 0.839394 | 50.363636 | 277 | 88 | 18 |
| Jev guided | Реальные ответы из native-кэша | **0.846154** | **50.769231** | 286 | 95 | 9 |
| TF-IDF + LogisticRegression | Локально измерен | 0.778462 | 46.707692 | 253 | 102 | 42 |
| Jev + TF-IDF, среднее | Измерен на тех же ответах Jev | 0.842262 | 50.535714 | 283 | 94 | 12 |
| Jev + RAG | Прерван: HTTP 401 | — | — | — | — | — |
| Evidence → Jev | Прерван: HTTP 401 | — | — | — | — | — |
| Jev + сильный арбитр | Прерван: HTTP 401 | — | — | — | — | — |
| Дообученный semantic encoder | Локально измерен | 0.786704 | 47.202216 | 284 | 143 | 11 |
| Jev + границы категорий | Прерван: HTTP 401 | — | — | — | — | — |

Это лучший из **измеренных** вариантов на данном публичном наборе, а не
утверждение о максимуме F1 или качестве четырёх недоступных методов. В `config.json`
сохранён `jev`; неуспешно измеренный вариант не объявлен победителем.

## Фактическое обучение encoder

Модель `cointegrated/rubert-tiny2`, фиксированная ревизия
`e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`, MIT, 4821 обучающий документ.
Все transformer-веса и multilabel-голова обучались на Apple M4 Pro/MPS:
три эпохи, seed42, AdamW lr3e-5, weight_decay0.01, batch16 документов.
Для длинных текстов используются начало/середина/конец, до трёх окон512токенов;
masked pooling сохраняет градиенты. Это не frozen embeddings + линейная модель.

Train BCE по эпохам: 0.352606 → 0.180750 → 0.133269. Время команды164.31секунды.
Дообученные веса сохранены в safetensors, tokenizer и metadata — локально в
`out/semantic-encoder/`. Inference работает только с локальным артефактом;
нет автоматического скачивания, обучения или фиктивной модели при ошибке.
Положительных train-примеров категории101 нет — это явно отражено в metadata.

## Проверки и ограничение доступа

- Реальный JSON Schema probe общего клиента успешно прошёл до отказа ключа:
  API сообщил `gpt-6.1-terra` и вернул точную цитату о сантехнических работах.
  Это проверка транспорта, не F1 новых методов.
- Локально проверены отказ при выдуманной цитате, исходной категории, bool вместо
  mcId, неполном/неверном наборе routed-категорий. Модельные ответы не мокались.
- RAG индекс реально построен на 2975 полных коротких train-текстах; для
  контрольного запроса найдены пять пар с правильными положительной/отрицательной
  метками, без normalized-exact совпадения с запросом. Это проверка поиска, не F1.
- Отсутствие ключа даёт явную ошибку, а не пустое успешное предсказание.
  При первой ошибке отменяются ещё не запущенные задания очереди.
- Общая публикация независимо пересчитала каждую доступную метрику из per-row
  предсказаний и проверила идентичность протоколов/эталонных строк всех запусков.
- Четыре API-метода нельзя считать проверенными end-to-end на 208 строках:
  предоставленный ключ отклонён также независимым endpoint списка моделей.

Pytest, `tests/` и тестовый фреймворк не добавлены. Проверки выполнены реальными
бенчмарками/контрольными скриптами, в соответствии с актуальным AGENTS.md.

## Воспроизведение

```bash
uv sync
PYTHONPATH=src uv run python -m subcategory_extraction.jev_tfidf.train
HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src uv run python -m subcategory_extraction.semantic_encoder.train --dataset out/hybrid-training.jsonl --device auto
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods google_decision tfidf semantic_encoder
```

Последняя команда работает без BestAPI после обучения локальных моделей. Для
новых входов Jev/RAG/evidence/арбитра/границ нужен действующий `BESTAPI_API_KEY`
или полный совпадающий кэш фактических ответов. Повторить отдельный API-метод:

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods jev_retrieval --report src/subcategory_extraction/benchmarks/classifier_methods/retrieval-results.json
```

Каждый метод запускается отдельно, чтобы его ошибка не скрыла остальные.
Сборка общего отчёта из шести фактических файлов, без новых вызовов моделей:

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/publish.py --reports out/methods-baselines.json out/methods-retrieval.json out/methods-evidence.json out/methods-arbiter.json out/methods-encoder.json out/methods-boundaries.json
```

Машиночитаемый отчёт с предсказаниями, model IDs, source/data hashes, ошибками,
cache/API-счётчиками и историей реального обучения: [recorded-results.json](recorded-results.json).
Веса, API-кэши и `.env` в Git не включаются.

## Интерпретация

1. TF-IDF в голосовании не улучшил Jev. Encoder немного лучше этого конкретного
   word TF-IDF, но заметно хуже измеренного Jev и даёт много FP.
2. Для RAG, evidence, арбитра и границ нет измеренного преимущества или поражения:
   их нельзя ранжировать по качеству до завершённого инференса.
3. Публичный набор уже просматривался в предыдущих экспериментах и содержит
   противоречивую разметку одинаковых текстов. Это исследовательский benchmark,
   не независимая оценка скрытого соревнования. F1 >=0.98 не достигнут.
4. Модели/пороги/маршрутизация не изменялись после просмотра текущих результатов.
   После отказа доступа другие модельные ответы не подставлялись.
