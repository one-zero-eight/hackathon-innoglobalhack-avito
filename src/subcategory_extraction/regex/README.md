# regex

Адаптер существующей функции `dataset_annotation.annotate_dataset.propose()`
к интерфейсу `SubcategoryExtractionPipeline.run(PipelineInput)`.
Сами правила и исходные датасеты не изменяются. Вход — только description;
исходная категория исключается. Веса, внешние API и обучение не нужны.

Не используются сохранённые annotations, пользовательские исключения,
идентификаторы строк или эталонные метки. Сравнение с исправленным test
проверяет в основном согласованность с правилами его автоматической разметки:
это не независимая оценка обобщения. Правила ранее исправлялись по всему
корпусу, включая test; результат на исходном test тоже не является holdout.

Для выбора укажите `"subcategory_extraction": "regex"` в config.json.
Оба существующих F1-сценария:

```bash
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods regex --offline
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods regex --offline
```

При mcId, отличном от 101, этот адаптер также не предлагает 101: существующий
аннотатор определяет только направления 102–106. Оба F1-сценария используют 101.
