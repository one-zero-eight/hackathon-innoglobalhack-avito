# Jev

`pipeline.py`: `SubcategoryExtractionPipeline(BasePipeline)`,
`run(inp: PipelineInput) -> list[int]`; `probabilities()` используется составными
модулями. `client.py` реализует нативный System One API с проверкой ответа и кэшем.
Исходная категория исключается, порог по умолчанию 0.5.

Настройки: `BESTAPI_API_KEY`, `JEV_MODEL`, `JEV_ENDPOINT`, `JEV_TIMEOUT`,
`JEV_THRESHOLD`, `JEV_PROMPT_PATH`, `JEV_CACHE_DIR`.
По умолчанию используется общий Google guided-промпт, не его копия.
Ключи и кэши не коммитятся; недоступность API не заменяется локальным ответом.

Сравнение: [classifier_methods](../benchmarks/classifier_methods/README.md).
Гибрид: [jev_tfidf](../benchmarks/classifier_methods/README.md).
