# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) выбранный модуль выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Варианты: `google_decision` (Google API, промпты baseline/guided),
  `tfidf` (локальная логистическая регрессия, JSON-артефакт).
- Выбор: ключ `subcategory_extraction` в корневом `config.json`
- `null` — модуль не выбран.
- Бенчмарки: `benchmarks/<name>/` (`run.py` + `README.md`)
- Google: `LLM_MODEL_ID=gemini-decision-1`, `LLM_API_KEY` или
  `GOOGLE_API_KEY`; `GOOGLE_DECISION_PROMPT=guided`, порог
  `GOOGLE_DECISION_THRESHOLD=0.5`. Запросы отправляются только в `run()`.
- TF-IDF: сначала обучите артефакт на официальном train:
  `uv run python src/subcategory_extraction/benchmarks/tfidf_quality/run.py`.
  По умолчанию веса в `out/tfidf.json`; другой путь задаётся
  `TFIDF_ARTIFACT_PATH`. Артефакт — JSON, не исполняемый pickle.
  Все описания с одинаковой NFKC/casefold/ё→е/whitespace-нормализацией
  объединены до train/validation/holdout-разбиения. Порог выбирается только
  на validation. Близкие, но не идентичные шаблоны отдельно не группируются.
  Train имеет только исходную категорию 101; качество для других исходных
  категорий и распознавание 101 этим прогоном не подтверждены.
- [Сравнение промптов](benchmarks/prompt_comparison/README.md): TF-IDF и оба
  текущих промпта Google на одной validation-выборке, с реальными API-вызовами.
- [Время ответа](benchmarks/response_time/README.md): 10 одинаковых входов,
  среднее, медиана, p95; стоимость в README заполняется вручную.
- Для полного CLI выберите оба модуля в `config.json` и настройте генератор
  по [draft_generation/README.md](../draft_generation/README.md).
