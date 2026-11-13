# category_validation

Шаг 2 пайплайна: проверка микрокатегорий, выбранных на шаге 1 — остаются только
категории из таблицы кейса (101–106), без исходной `mcId` и дублей.

- Интерфейс: `CategoryValidationPipeline.run(candidates, current_mc_id) -> list[int]`
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`, прогон — `tests/`)
