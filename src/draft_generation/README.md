# draft_generation

Шаг 3 пайплайна: под каждую валидную целевую микрокатегорию создаётся черновик
объявления (`title` + `text`), написанный только по фактам исходного объявления.

- Интерфейс: `DraftGenerationPipeline.run(PipelineInput, target_mc_ids) -> list[dict]`
  (dict контракта кейса, валидация — в `main.PipelineOutput`)
- Реализаций пока нет. Варианты добавляются в `<module>/pipeline.py`,
  промпты — в `<module>/prompts/`.
- Выбор: ключ `draft_generation` в корневом `config.json`
- `null` — модуль не выбран, бенчмарки пропускаются.
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`, прогон — `tests/`)
