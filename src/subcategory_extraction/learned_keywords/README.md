# learned_keywords

Локальная модель: TF-IDF (униграммы и биграммы) плюс бинарные совпадения
с 40 фразами на категорию. Фразы выбираются по χ² и большей частоте
в положительном классе; логистическая регрессия использует оба блока.
Это combined-вариант из PR #9, переобученный по актуальным правилам.

## Использование

В `config.json` задайте `"subcategory_extraction": "learned_keywords"`.
Класс создаётся без аргументов, `run(PipelineInput)` возвращает дополнительные
mcId и исключает текущую категорию. Входной и выходной контракт не меняется.

Веса и IDF загружаются из `artifacts/learned-keywords.json` относительно
`pipeline.py`; скачивать их отдельно не требуется. Можно указать другой файл
через `LEARNED_KEYWORDS_ARTIFACT`. Pickle и внешние API не используются.
Метаданные, SHA-256 train и весов — в `artifacts/metadata.json`.

## Воспроизведение весов

```bash
uv sync
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src uv run python -m subcategory_extraction.learned_keywords.train
```

Обучение только на `docs/improved_dataset/train.jsonl`, 4851 строка после
проверки и дедупликации. Используется весь train; validation-разбиения нет,
поэтому близкие дубликаты внутри train не попадают в разные части.
Параметры исходного combined-варианта зафиксированы до оценки: `C=4`,
`min_df=2`, 40 фраз на категорию, порог 0.5; параметры по test не подбирались.
Категория 101 не обучалась (в train это исходная категория), поэтому модуль
не умеет предлагать «Ремонт под ключ» для объявлений других категорий.

## Бенчмарки

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --methods learned_keywords
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --methods learned_keywords
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python src/subcategory_extraction/benchmarks/response_time/run.py --module learned_keywords --size 10
```

Все вызовы локальные. F1-runner также поддерживает `--offline` для этого
модуля. Официальный test содержит пересечения с train, исправленный
является диагностическим с предварительной разметкой. Исторические веса,
обученные на Юле, и прежние метрики нельзя сравнивать как один прогон.
