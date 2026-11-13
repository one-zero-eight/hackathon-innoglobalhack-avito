# category_validation

Шаг 2 пайплайна: проверка микрокатегорий, выбранных на шаге 1 — остаются только
категории из таблицы кейса (101–106), без исходной `mcId` и дублей.

- Интерфейс: `CategoryValidationPipeline.run(candidates, current_mc_id) -> list[int]`
- Вариант `strict`: отбрасывает неизвестные ID, bool/строки/дробные значения,
  исходную категорию и повторы с сохранением порядка.
  Семантическую классификацию выполняет первый шаг: интерфейс второго
  шага не принимает текст объявления.
- Выбор: ключ `category_validation` в корневом `config.json`
- `null` — модуль не выбран.
- Бенчмарки: `benchmarks/` (каждый бенчмарк — отдельный `.md`)
- Офлайн-прогон:
  `PYTHONPATH=src uv run python -m category_validation.benchmarks.run_strict`.
- `benchmarks/historical/` — перенесённые результаты и скрипты
  сравнений с SplitShift. Для исторического повторения нужны локальные
  `local-notes/joined-datasets/train.csv`, companion JSONL и внешний
  `../IGH cases/splitshift`. Новые оценки делаем только на InnoGlobalHack.
