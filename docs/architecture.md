# Архитектура репозитория

По тексту объявления и его текущей микрокатегории сервис находит дополнительные
услуги из категории «Ремонт и отделка» и создаёт для них черновики объявлений.
Исходная микрокатегория исключается из результата.

В репозитории есть два пути выполнения: CLI с реализациями из `config.json` и
веб-сервис с классификацией и генерацией через Selectel. Оба используют общие
схемы данных и таблицу микрокатегорий из `src/main.py`, но веб-сервис не вызывает
CLI-пайплайн.

## Общая архитектура

```mermaid
flowchart TB
    Input["Объявление:<br/>description + mcId"]

    subgraph Offline["CLI и офлайн-оценка"]
        Main["src/main.py<br/>валидация и оркестрация"]
        Config["config.json<br/>выбор реализаций"]
        Extract["subcategory_extraction<br/>TF-IDF или Decisions"]
        Generate["draft_generation/sample_llm<br/>Gemini API"]

        Main --> Config
        Main --> Extract
        Extract -->|"target_mc_ids"| Generate
    end

    subgraph Web["Веб-сервис"]
        UI["React + Mantine<br/>src/web"]
        API["FastAPI<br/>src/api"]
        Service["DraftsService<br/>классификация и генерация"]
        DB[("PostgreSQL<br/>категории и датасет")]
        Decisions["Selectel Decisions<br/>jev-1.13"]
        Chat["Selectel Chat Completions<br/>Llama"]

        UI -->|"POST /api/drafts"| API
        API --> Service
        Service --> DB
        Service --> Decisions
        Service --> Chat
    end

    Input --> Main
    Input --> UI
    Generate --> Output["Результат:<br/>drafts[]"]
    Chat --> Output
```

## CLI

`src/main.py` читает входной JSON, проверяет его через Pydantic, загружает
выбранные модули и последовательно выполняет выделение категорий и генерацию
черновиков. Текущая конфигурация: `tfidf` + `sample_llm`.

```mermaid
sequenceDiagram
    participant User as Пользователь
    participant CLI as src/main.py
    participant Config as config.json
    participant TFIDF as TF-IDF pipeline
    participant Draft as sample_llm pipeline
    participant Gemini as Gemini API

    User->>CLI: --input input.json --output output.json
    CLI->>CLI: Загрузить .env и проверить PipelineInput
    CLI->>Config: load_config()
    Config-->>CLI: tfidf + sample_llm

    CLI->>TFIDF: run(inp)
    TFIDF->>TFIDF: Загрузить out/tfidf.json
    TFIDF->>TFIDF: TF-IDF → логистические веса → вероятности
    TFIDF-->>CLI: mcId с p >= threshold,<br/>кроме исходной категории

    CLI->>Draft: run(inp, target_mc_ids)

    loop Каждая целевая категория
        Draft->>Gemini: Промпт + исходное описание + JSON Schema
        Gemini-->>Draft: {"title": "...", "text": "..."}
        Draft->>Draft: Проверить ответ через Pydantic
    end

    Draft-->>CLI: Список drafts
    CLI->>User: Записать output.json
```

## Веб-сервис

React-клиент получает справочник микрокатегорий и отправляет исходное объявление
в FastAPI. `DraftsService` получает вероятности дополнительных категорий через
Selectel Decisions, применяет порог и параллельно генерирует черновики через
Selectel Chat Completions.

```mermaid
sequenceDiagram
    participant Browser as React-клиент
    participant API as FastAPI
    participant DB as PostgreSQL
    participant Service as DraftsService
    participant Decisions as Selectel Decisions
    participant Chat as Selectel Chat

    Browser->>API: GET /api/microcategories
    API->>DB: Прочитать справочник
    DB-->>Browser: Категории 101–106

    Browser->>API: POST /api/drafts<br/>description + mcId
    API->>Service: run(inp)
    Service->>DB: Получить каталог микрокатегорий

    Service->>Decisions: Текст + noul-вопросы<br/>для всех категорий, кроме исходной
    Decisions-->>Service: Вероятность p для каждой категории

    Service->>Service: Оставить p >= порога модуля<br/>(JEV_THRESHOLD, 0.5 по умолчанию)

    alt Подходящих категорий нет
        Service-->>Browser: {"drafts": []}
    else Категории найдены
        par Параллельная генерация черновиков
            Service->>Chat: system/user prompt<br/>для целевой категории
            Chat-->>Service: Текст с JSON title + text
            Service->>Service: Извлечь JSON и провалидировать
        end
        Service-->>API: PipelineOutput
        API-->>Browser: {"drafts": [...]}
    end
```

## Docker и база данных

`docker-compose.yml` поднимает PostgreSQL, FastAPI и nginx с собранным
React-клиентом. nginx раздаёт статику и проксирует `/api/*` в контейнер `api`.

```mermaid
flowchart LR
    Browser["Браузер"] -->|"localhost:8081"| Web["web<br/>nginx + статика React"]
    Browser -.->|"localhost:8000<br/>прямой доступ"| API

    Web -->|"прокси /api/*<br/>api:8000"| API["api<br/>FastAPI + Uvicorn"]
    API --> DB[("db<br/>PostgreSQL 18")]
    API --> Selectel["Selectel<br/>Decisions + Chat"]

    Init["deploy/db/init.sql"] -->|"первый запуск"| DB
    Data["./data<br/>read-only volume"] --> API
    Env[".env"] --> API
```

В PostgreSQL хранятся справочник `microcategories` и объявления `case_ads`.
`src/api/dataset_loader.py` идемпотентно загружает официальные train/test-данные.

## Оценка решения

`src/run.py` прогоняет полный CLI-пайплайн на датасете, извлекает категории из
итоговых черновиков и сравнивает их с эталонными множествами по micro-F1.

```mermaid
flowchart LR
    RunConfig["run.config.json"] --> Eval["src/run.py"]
    Dataset["Parquet или JSONL"] --> Eval
    Train["Train-датасет"] --> Overlap["Проверка точных<br/>пересечений описаний"]
    Overlap --> Eval

    Eval --> Pipeline["run_pipeline()<br/>классификация + генерация"]
    Pipeline --> Predicted["Множества mcId<br/>из итоговых drafts"]
    Truth["Эталонные категории"] --> Metrics["micro-F1<br/>TP / FP / FN"]
    Predicted --> Metrics
    Metrics --> Report["out/evaluation.json"]
```

Классификационные баллы рассчитываются как `60 × micro-F1`. В отчёт сохраняются
метрики, конфигурация, время выполнения и полный ответ для каждой строки.
