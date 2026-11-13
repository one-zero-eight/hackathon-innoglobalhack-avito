# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) выбранный модуль выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Варианты: `google_decision` (Google API, промпты baseline/guided),
  `tfidf` (локальная логистическая регрессия, JSON-артефакт),
  `jev` (нативный Jev через BestAPI), `jev_tfidf` (среднее вероятностей),
  `jev_retrieval` (размеченные соседи), `evidence_jev` (цитаты → Jev),
  `jev_arbiter` (сильный арбитр), `semantic_encoder` (fine-tuned RuBERT-tiny2),
  `jev_boundaries` (точные границы категорий).
- Выбор: ключ `subcategory_extraction` в корневом `config.json`
- `null` — модуль не выбран.
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`)
- Google: `LLM_MODEL_ID=gemini-decision-1`, `LLM_API_KEY` или
  `GOOGLE_API_KEY`; `GOOGLE_DECISION_PROMPT=guided`, порог
  `GOOGLE_DECISION_THRESHOLD=0.5`. Запросы отправляются только в `run()`.
- TF-IDF для нового гибрида обучается без validation и поиска порога:
  `PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_jev_hybrid`.
  Веса — `out/tfidf.json`; другой путь задаётся `TFIDF_ARTIFACT_PATH`.
  Артефакт — JSON, не исполняемый pickle. Из обучения по текстам исключены
  компоненты точных и близких совпадений с public test; метки test не используются.
  Порог фиксирован 0.5. Старый `train_tfidf` с validation/holdout не вызывается
  новым workflow. Train содержит только исходную категорию 101.
- Jev: `BESTAPI_API_KEY`, `JEV_MODEL=jev-1.13.0`,
  `JEV_ENDPOINT=https://bestapi.fugue.pro/v1/systemone`, `JEV_THRESHOLD=0.5`.
  По умолчанию переиспользуется существующий Google guided-промпт;
  `JEV_PROMPT_PATH` может указать другой файл с `{category}` и `{examples}`.
  Фактические ответы API кэшируются в `out/jev-cache`; `JEV_CACHE_DIR`
  меняет путь. Cache key включает endpoint, модель, текст и вопросы.
- Гибрид: `JEV_TFIDF_WEIGHT=0.5`, `JEV_TFIDF_THRESHOLD=0.5`.
  Категории без обученной локальной головы оцениваются только Jev.
  Вес и пороги фиксируются до оценки и не подбираются на public test.
- Общий бенчмарк:
  `PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.compare_jev_hybrid`.
  Он загружает реальные модули через `load_pipeline`, сохраняет все 208 строк,
  считает официальную sklearn micro-F1 и `60 × F1` через `src/run.py`.
  Оценка классификаторов не включает качество генерации и полный балл хакатона.
- Офлайн-проверка переноса Google на сохранённых ответах:
  `PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.replay_google`.
- Исторические сравнения с внешним SplitShift и их исходные результаты:
  `benchmarks/historical/`. Они не являются
  готовыми автономными модулями SplitShift: его код и модели не входят в репозиторий.
- В `config.json` выбран `jev` и существующий `sample_llm`: в текущем
  сравнении фиксированное усреднение `jev_tfidf` не превзошло чистый Jev.
  Гибрид остаётся доступен как отдельный модуль, а не выдаётся за улучшение.
  Полный CLI требует также ключ Google в `LLM_API_KEY` для генератора.

Пять новых методов и их ограничения описаны в [общем отчёте](benchmarks/classifier-methods.md).
Общий запуск: `PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.compare_methods`.
Для реального API-инференса нужен рабочий BestAPI-ключ; в текущем эксперименте
четыре новых API-метода остановились на HTTP 401, без фиктивной метрики.
Encoder реально обучен; три эпохи запускаются отдельным `train_semantic_encoder`.
