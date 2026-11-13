# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) выбранный модуль выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Варианты: `google_decision` (Google API, промпты baseline/guided),
  `tfidf` (локальная логистическая регрессия, JSON-артефакт).
- Выбор: ключ `subcategory_extraction` в корневом `config.json`
- `null` — модуль не выбран.
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`)
- Google: `LLM_MODEL_ID=gemini-decision-1`, `LLM_API_KEY` или
  `GOOGLE_API_KEY`; `GOOGLE_DECISION_PROMPT=guided`, порог
  `GOOGLE_DECISION_THRESHOLD=0.5`. Запросы отправляются только в `run()`.
- TF-IDF: сначала обучите артефакт на официальном train:
  `PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.train_tfidf`.
  По умолчанию веса в `out/tfidf.json`; другой путь задаётся
  `TFIDF_ARTIFACT_PATH`. Артефакт — JSON, не исполняемый pickle.
  Все описания с одинаковой NFKC/casefold/ё→е/whitespace-нормализацией
  объединены до train/validation/holdout-разбиения. Порог выбирается только
  на validation. Близкие, но не идентичные шаблоны отдельно не группируются.
  Train имеет только исходную категорию 101; качество для других исходных
  категорий и распознавание 101 этим прогоном не подтверждены.
- Офлайн-проверка переноса Google на сохранённых ответах:
  `PYTHONPATH=src uv run python -m subcategory_extraction.benchmarks.replay_google`.
- Исторические сравнения с внешним SplitShift и их исходные результаты:
  `benchmarks/historical/`. Они не являются
  готовыми автономными модулями SplitShift: его код и модели не входят в репозиторий.
- `config.json` автоматически не меняется: генератора черновиков пока нет,
  поэтому полный CLI всё ещё требует реализации третьего шага.
