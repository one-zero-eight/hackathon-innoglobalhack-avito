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
- Бенчмарки: `benchmarks/<name>/` (`run.py` + `README.md`)
- Google: `LLM_MODEL_ID=gemini-decision-1`, `LLM_API_KEY` или
  `GOOGLE_API_KEY`; `GOOGLE_DECISION_PROMPT=guided`, порог
  `GOOGLE_DECISION_THRESHOLD=0.5`. Запросы отправляются только в `run()`.
- TF-IDF для нового гибрида обучается без validation и поиска порога:
  `PYTHONPATH=src uv run python -m subcategory_extraction.jev_tfidf.train`.
  Веса — `out/tfidf.json`; другой путь задаётся `TFIDF_ARTIFACT_PATH`.
  Артефакт — JSON, не исполняемый pickle. Из обучения по текстам исключены
  компоненты точных и близких совпадений с public test; метки test не используются.
  Порог фиксирован 0.5. Сценарий `tfidf_quality` с validation/holdout не вызывается
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
  `uv run python src/subcategory_extraction/benchmarks/jev_tfidf/run.py`.
  Он загружает реальные модули через `load_pipeline`, сохраняет все 208 строк,
  считает официальную sklearn micro-F1 и `60 × F1` через `src/run.py`.
  Оценка классификаторов не включает качество генерации и полный балл хакатона.
- В `config.json` выбран `jev` и существующий `sample_llm`: в текущем
  сравнении фиксированное усреднение `jev_tfidf` не превзошло чистый Jev.
  Гибрид остаётся доступен как отдельный модуль, а не выдаётся за улучшение.
  Полный CLI требует также ключ Google в `LLM_API_KEY` для генератора.

Все девять методов и их ограничения описаны в [общем отчёте](benchmarks/classifier_methods/README.md).
Общий запуск: `uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py`.
Для реального API-инференса нужен рабочий BestAPI-ключ; в текущем эксперименте
четыре новых API-метода остановились на HTTP 401, без фиктивной метрики.
Encoder обучается командой `PYTHONPATH=src uv run python -m subcategory_extraction.semantic_encoder.train`.
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

## Карта решений

| Модуль | Решение / ресурсы | Общая оценка |
| --- | --- | --- |
| `google_decision` | Нативный Decisions API; baseline/guided в `prompts/` | `prompt_comparison`, `response_time`, `classifier_methods` |
| `tfidf` | Word TF-IDF + LogisticRegression, JSON-веса | `tfidf_quality`, `classifier_methods`, `jev_tfidf` |
| [jev](jev/README.md) | Нативный Jev, проверка ответа и кэш | `classifier_methods`, `jev_tfidf` |
| `jev_tfidf` | Среднее вероятностей, обучение в `jev_tfidf/train.py` | `classifier_methods`, `jev_tfidf` |
| [jev_retrieval](jev_retrieval/README.md) | Положительный/отрицательный train-сосед | `classifier_methods` |
| [evidence_jev](evidence_jev/README.md) | Проверенные дословные цитаты → Jev | `classifier_methods` |
| [jev_arbiter](jev_arbiter/README.md) | LLM разрешает споры Jev / TF-IDF | `classifier_methods` |
| [semantic_encoder](semantic_encoder/README.md) | RuBERT-tiny2, обучение в `semantic_encoder/train.py` | `classifier_methods` |
| [jev_boundaries](jev_boundaries/README.md) | Jev с промптом границ категорий | `classifier_methods` |

У каждого сценария: `benchmarks/<name>/run.py`, `README.md`, отчёты рядом.
Сохранённые ответы не означают новый вызов API. Недоступные методы не получают
метрику другого модуля; новый частичный прогон описан в общем отчёте.
