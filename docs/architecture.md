# Архитектура проекта

```mermaid
flowchart TD
    INPUT["POST /drafts<br/>description + mcId"] --> MAIN["api.py + main.py<br/>Проверка входа и запуск пайплайна"]
    CONFIG["config.json + .env<br/>Модули, модели и параметры"] -.-> MAIN

    subgraph EXTRACTION["Выделение категорий: google_decision"]
        DECISION["Google Decisions API<br/>Gemini Decision 1"] --> TARGETS["Проверка вероятностей и порог<br/>Исключение исходной категории"]
    end
    MAIN --> DECISION

    subgraph GENERATION["Генерация черновиков: llm"]
        PROMPT["Исходные факты и целевая категория<br/>Один запрос на уникальный mcId"] --> PROVIDER{"Провайдер генерации"}
        PROVIDER -->|google| GEMMA["Google API<br/>Gemma 4"]
        PROVIDER -->|openai_compatible| CHAT["OpenAI-compatible API<br/>Настроенная модель"]
        GEMMA --> VALIDATE["Проверка title и text<br/>Сборка черновиков с mcId из кода"]
        CHAT --> VALIDATE
    end
    TARGETS -->|целевые mcId| PROMPT
    MAIN -.->|исходный текст| PROMPT
    VALIDATE --> OUTPUT["Выходной JSON<br/>drafts: mcId, title, text"]
```

Сплошные стрелки показывают основной поток обработки, пунктир — конфигурацию
и передачу исходного текста непосредственно генератору.

`main.py` выбирает модули через `config.json`; модели, провайдер генерации
и ключи задаются в локальном `.env`. Внутренние классы клиентов, транспорт,
повторы запросов и устройство бенчмарков на схеме опущены.

Сервер запускается через `uv run python -m src` (`src/__main__.py`).
Входной JSON поступает через `POST /drafts`.
HTTP-адаптер `src/api.py` вызывает `run_pipeline()`;
[описание API](http-api.md) включает запуск и примеры запросов.

Выделение возвращает категории, отличные от исходной. Генератор использует
этот список и факты исходного объявления, создавая один черновик на категорию.
При отсутствии целей результат — `{"drafts": []}` без вызова API генерации.
Ответ проверяется перед отправкой JSON. Ошибка одной категории прерывает
обработку; частичный результат не возвращается. Проверка схемы не заменяет
фактическую проверку текста исполнителем.

Реализации: [выделение](../src/subcategory_extraction/README.md) и
[генерация](../src/draft_generation/README.md).
Бенчмарки и статус live-проверок: [validation.md](validation.md).
