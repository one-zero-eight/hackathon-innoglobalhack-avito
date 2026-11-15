# Jev

`pipeline.py`: `SubcategoryExtractionPipeline(BasePipeline)`,
`run(inp: PipelineInput) -> list[int]`; `probabilities()` используется составными
модулями. `client.py` вызывает Selectel Decisions API с проверкой ответа и кэшем.
Исходная категория исключается, порог по умолчанию 0.5.

Настройки: `JEV_PROVIDER`, `SELECTEL_API_KEY`, `JEV_MODEL`, `JEV_ENDPOINT`, `JEV_TIMEOUT`,
`JEV_THRESHOLD`, `JEV_PROMPT_PATH`, `JEV_CACHE_DIR`.
По умолчанию используется общий Google guided-промпт, не его копия.
Ключи и кэши не коммитятся; недоступность API не заменяется локальным ответом.

## Selectel

```dotenv
JEV_PROVIDER=selectel
JEV_MODEL=typesafe/jev-1.13
JEV_ENDPOINT=https://api.selectel.ru/aig/v1/decisions
SELECTEL_API_KEY=
JEV_THRESHOLD=0.5
```

Модель и endpoint перенесены из `.env.example` ветки `projacktor/main`.
Сам `.env` и секреты в той ветке не отслеживаются. Ключ Selectel задаётся
локально отдельно от `BESTAPI_API_KEY`.

Запрос: `Authorization: Bearer`, JSON с `model`, `state: {"text": "..."}`
и `questions` типа `noul`. Ответ проверяется по точному model, набору
вопросов и конечным вероятностям в [0, 1]. Порог остаётся 0.5;
модель, промпт и порог не подбираются на test. HTML не принимается за JSON.

Endpoint и payload входят в ключ кэша: ответы старого BestAPI не используются
как ответы Selectel. Бенчмарки сохраняют provider/endpoint/model, хеш промпта,
порог, вероятности, usage и факт чтения кэша; API-ключ туда не попадает.

Для явного выбора прежнего транспорта задайте `JEV_PROVIDER=bestapi`,
`JEV_MODEL=jev-1.13.0`, `JEV_ENDPOINT=https://bestapi.fugue.pro/v1/systemone`
и отдельный `BESTAPI_API_KEY`. Автоматического переключения между API нет;
ключ одного провайдера не подставляется вместо другого. Endpoint должен
принадлежать выбранному провайдеру и использовать HTTPS.

Сравнение: [classifier_methods](../../benchmarks/classifier_methods/README.md).
Гибрид: [jev_tfidf](../../benchmarks/classifier_methods/README.md).

Анализ вариантов и выбор промпта: [PROMPTS.md](PROMPTS.md).
