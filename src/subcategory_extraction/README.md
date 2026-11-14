# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) выбранный модуль выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Варианты: `google_decision` (Google API, промпты baseline/guided),
  `regex` (существующие правила офлайн-аннотатора без индивидуальных исключений),
  `tfidf` (локальная логистическая регрессия, JSON-артефакт),
  `learned_keywords` (TF-IDF и бинарные признаки выученных фраз),
  `jev` (нативный Jev через BestAPI), `jev_tfidf` (среднее вероятностей),
  `jev_retrieval` (размеченные соседи), `evidence_jev` (цитаты → Jev),
  `jev_arbiter` (сильный арбитр), `semantic_encoder` (fine-tuned RuBERT-tiny2),
  `semantic_encoder_calibrated` (validation-пороги),
  `semantic_encoder_f1` (выбор checkpoint, окна/attention и дистилляция; CPU),
  `jev_boundaries` (точные границы категорий),
  [qwen_decision](qwen_decision/README.md) (Qwen Token Plan System One,
  выбор промпта и порогов только по train).
- Выбор: ключ `subcategory_extraction` в корневом `config.json`
- `null` — модуль не выбран.
- Результат содержит только дополнительные mcId, исходная категория исключена.

## Четыре проверки

[Общий список и команды](benchmarks/README.md). Новые бенчмарки и папки запрещены.

- [Стоимость токенов](benchmarks/cpu_costs/README.md) — ручная запись.
- [F1 improved_dataset](benchmarks/improved_dataset/README.md) — 119 строк.
- [F1 InnoGlobalHack](benchmarks/classifier_methods/README.md) — 208 строк.
- [Время работы](benchmarks/response_time/README.md) — настоящие вызовы `run()`.

Оба F1-runner по умолчанию проверяют сохранённые ответы без API и обучения.
Для оценки текущего модуля передайте `--methods <имя>`; параметры и пути
к весам берутся из `.env`. Подготовка весов описана в README модулей,
код обучения хранится внутри модуля. Старые эксперименты доступны в истории Git.

[dataset_annotation: время и шесть примеров с mcId](../dataset_annotation/rubert_review/README.md).
