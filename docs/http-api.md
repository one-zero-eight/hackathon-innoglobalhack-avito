# HTTP API

CLI по-прежнему читает и записывает JSON-файлы:

```bash
uv run python src/main.py --input data/input.json --output out/output.json
```

`src/api.py` добавляет FastAPI-адаптер той же функции `run_pipeline()`.
В HTTP-режиме вход — JSON-тело запроса, выход — JSON-тело ответа; файлы
для каждого запроса не создаются. Схемы `PipelineInput` и `PipelineOutput`
импортируются из `main.py`, формат кейса не меняется.

## Запуск

```bash
uv sync
uv run uvicorn --app-dir src api:app --host 127.0.0.1 --port 8000
```

Документация и форма запроса: [Swagger UI](http://127.0.0.1:8000/docs).
Схема API: [OpenAPI JSON](http://127.0.0.1:8000/openapi.json).

`config.json` и `.env` читаются при старте относительно проекта. После их
изменения перезапустите сервер. Модули проверяются при старте без вызовов
внешних моделей. Синхронная обработка запросов выполняется в worker thread.

## POST /drafts

```bash
curl http://127.0.0.1:8000/drafts \
  -H 'Content-Type: application/json' \
  --data-binary @data/input.json
```

Тело запроса:

```json
{
  "description": "Устанавливаю раковины и меняю смесители.",
  "mcId": 101
}
```

Ответ при успешной обработке:

```json
{
  "drafts": [
    {
      "mcId": 104,
      "title": "Сантехнические работы",
      "text": "Устанавливаю раковины и меняю смесители."
    }
  ]
}
```

Текст выше — пример формата, а не сохранённая реальная генерация.
Если дополнительных услуг нет, возвращается `{"drafts": []}`.

| Статус | Значение |
| --- | --- |
| 200 | Успешный `PipelineOutput`, поля категорий называются `mcId` |
| 422 | Невалидное тело запроса; модели не вызываются |
| 502 | Ошибка внешней модели или невалидный результат; частичных drafts нет |

## GET /health

```json
{ "status": "ok" }
```

Это проверка доступности приложения, без запросов к моделям. Она не подтверждает
доступность Google/Gemma. В последней реальной проверке генератор возвращал
HTTP 500 на стороне Google; API передаст такую ошибку как 502. Сам HTTP-адаптер
не переключает модели и не подменяет генерацию заглушкой.

Проверка HTTP-контракта: [бенчмарк](../src/draft_generation/benchmarks/http_api/README.md).
Основа реализации: [JSON body](https://fastapi.tiangolo.com/tutorial/body/),
[lifespan](https://fastapi.tiangolo.com/advanced/events/),
[sync endpoints](https://fastapi.tiangolo.com/async/).
