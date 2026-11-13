# subcategory_extraction

Шаг 1 пайплайна: по тексту объявления (`description`) jev-модель выделяет услуги
и определяет микрокатегории (101–106), к которым они относятся.

- Интерфейс: `SubcategoryExtractionPipeline.run(PipelineInput) -> list[int]`
- Промпты: `prompts/`
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`, прогон — `tests/`)
