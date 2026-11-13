# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) выбранный модуль выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Реализаций пока нет. Варианты добавляются в `<module>/pipeline.py`,
  промпты — в `<module>/prompts/`.
- Выбор: ключ `subcategory_extraction` в корневом `config.json`
- `null` — модуль не выбран.
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`)
- Папка бенчмарков пока пуста; прогоны добавим вместе с рабочими реализациями.
