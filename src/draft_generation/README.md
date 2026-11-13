# draft_generation

Шаг 3 пайплайна: под каждую валидную целевую микрокатегорию создаётся черновик
объления (`title` + `text`), написанный только по фактам исходного объявления.

- Интерфейс: `DraftGenerationPipeline.run(PipelineInput, target_mc_ids) -> list[dict]`
  (dict контракта кейса, валидация — в `main.PipelineOutput`)
- Промпты: `prompts/`
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`, прогон — `tests/`)
