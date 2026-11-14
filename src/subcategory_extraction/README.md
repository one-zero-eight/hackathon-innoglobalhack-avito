# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) выбранный модуль выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Варианты: `google_decision` (Google API, промпты baseline/guided),
  `tfidf` (локальная логистическая регрессия, JSON-артефакт),
  `jev` (нативный Jev через BestAPI), `jev_tfidf` (среднее вероятностей),
  `jev_retrieval` (размеченные соседи), `evidence_jev` (цитаты → Jev),
  `jev_arbiter` (сильный арбитр), `semantic_encoder` (fine-tuned RuBERT-tiny2),
  `semantic_encoder_calibrated` (validation-пороги),
  `semantic_encoder_f1` (выбор checkpoint, окна/attention и дистилляция; CPU),
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
- В `config.json` выбран локальный CPU-классификатор `semantic_encoder_f1`
  и существующий внешний генератор `sample_llm`. Ключ для классификации
  не нужен; генерация непустых черновиков требует отдельного рабочего API-ключа.
  Исторический `jev_tfidf` остаётся отдельным доступным вариантом.

Пять новых методов и их ограничения описаны в [общем отчёте](benchmarks/classifier-methods.md).
Общий запуск: `PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.compare_methods`.
Для реального API-инференса нужен рабочий BestAPI-ключ; в текущем эксперименте
четыре новых API-метода остановились на HTTP 401, без фиктивной метрики.
Encoder реально обучен; три эпохи запускаются отдельным `train_semantic_encoder`.

[CPU и стоимость API](benchmarks/cpu-costs.md): encoder измерен на CPU без
переобучения; все пять базовых наборов предсказаний совпали с предыдущим отчётом.
`compare_methods` сохраняет latency/loading и `costs`; в API-цене различаются
нулевой расход текущего кэша и неизвестная стоимость нового инференса.
Учёт в `main.py` общий с генератором черновиков.

## Полный план улучшения локального RuBERT

Новый вариант `semantic_encoder_f1` сохраняет публичный контракт и загружает
только одного компактного студента на CPU; учитель нужен лишь при обучении.
`RUBERT_F1_ARTIFACT_PATH` задаёт путь артефакта, по умолчанию
`out/rubert-f1-plan/selected`. `RUBERT_F1_DEVICE=cpu` обязателен; GPU inference
не является скрытым fallback. Без весов или привязанной к ним calibration
загрузка завершается явной ошибкой.

Обучение и единая оценка:

```bash
HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.improve_rubert_f1
```

Нужны прежнее воспроизводимое разбиение из `tune_semantic_thresholds`,
baseline-веса по `improved-dataset.md` и сохранённая train-only смысловая
проверка. Все параметры и выбор сохраняются до доступа к контрольной оценке.
[Протокол, результаты и ограничения](benchmarks/rubert-f1-plan.md).
