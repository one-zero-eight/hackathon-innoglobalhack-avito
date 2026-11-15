# Услуги из объявлений Авито → черновики

По тексту объявления и его микрокатегории находим услуги из других
микрокатегорий «Ремонт и отделка» (101–106) и генерируем черновики объявлений.

## Структура

| Путь | Что это |
| --- | --- |
| `src/main.py` | CLI + всё общее: контракт, микрокатегории, `BasePipeline`, `LLMClient` |
| `src/api.py` | HTTP API полного пайплайна и селекторы модулей в Swagger |
| `config.json` | выбор реализации каждого шага |
| `src/subcategory_extraction/` | шаг 1: выделение подкатегорий услуг |
| `src/draft_generation/` | шаг 2: генерация черновиков |
| `data/` | пример входа |
| `docs/` | кейс, критерии, справочник, датасеты |
| `backlog/` | локальный бэклог (в `.gitignore`) |
| `src/api/` | веб-сервис: FastAPI-бэкенд, клиенты LLM, промпты |
| `src/web/` | веб-клиент: Vite + React + Mantine |
| `docker-compose.yml`, `deploy/` | инфраструктура веб-сервиса: db + api + web |

В каждом разделе — несколько вариантов: `<module>/pipeline.py` и, при
необходимости, `<module>/prompts/`. Общие `benchmarks/<name>/` с `run.py`,
`README.md` и результатами остаются в корне раздела. Экспериментальные решения тоже оформляем
как модули, без отдельной папки `experiments/`.

Обученные модели храним внутри своего решения:

```text
src/subcategory_extraction/<group>/<module>/  # остальные сервисы: src/<service>/<module>/
├── __init__.py
├── pipeline.py
├── prompts/       # обязательно для decisions, иначе при необходимости
├── PROMPTS.md     # анализ различий промптов для каждого decision-модуля
└── artifacts/     # обученные веса, токенизатор, метаданные и калибровка
```

`artifacts/` включаем в Git и загружаем на GitHub вместе с решением;
для файлов сверх лимита обычного GitHub Git используем Git LFS.
Итоговые веса обучения и путь загрузки модуля по умолчанию должны указывать
сюда; способ загрузки и воспроизведения весов описываем в README модуля.
`semantic_encoder_f1` по умолчанию загружает модель из своей папки `artifacts/`.
Временные checkpoint и отчёты полного обучения остаются в `out/`;
выбранная итоговая модель копируется в модуль.

## Выбор реализации

Корневой `config.json` задаёт модуль каждого шага; для extraction — `<group>.<module>`:

```json
{
  "subcategory_extraction": "decisions.jev",
  "draft_generation": "sample_llm"
}
```

Модули выделения сгруппированы по подходам. Внутри каждой группы — папки модулей:

```text
src/subcategory_extraction/
├── lexical/       # regex/, tfidf/, learned_keywords/, snippet_logistic_regression/
├── semantic/      # semantic_encoder/, semantic_encoder_calibrated/, semantic_encoder_f1/, semantic_evidence/
├── decisions/     # google_decision/, qwen_decision/, jev/, local_jev/, jev_calibrated/, jev_boundaries/, jev_retrieval/
├── hybrid/        # jev_tfidf/, evidence_jev/, jev_arbiter/
└── benchmarks/    # четыре общих сценария для всех подходов
```

TinyBERT исключён из поддерживаемых модулей.
Указывайте полный путь, например `lexical.tfidf` или `decisions.jev`.
[Список методов](src/subcategory_extraction/README.md),
[веса и подготовленные ресурсы](src/subcategory_extraction/ARTIFACTS.md).
Доступен генератор `draft_generation/sample_llm`. `null` означает, что модуль
не выбран. CLI выдаёт явную ошибку.

Текущий Jev использует Selectel Decisions: `JEV_PROVIDER=selectel`,
`JEV_MODEL=typesafe/jev-1.13`,
`JEV_ENDPOINT=https://api.selectel.ru/aig/v1/decisions`.
Задайте отдельный `SELECTEL_API_KEY` в `.env`; детали — в
[README Jev](src/subcategory_extraction/decisions/jev/README.md).
Доступен экспериментальный [jev_calibrated](src/subcategory_extraction/decisions/jev_calibrated/README.md):
минимальный промпт и порог 0.635 выбраны по train-группам; control micro-F1
0.926829 → 0.939759. На официальном test: 0.842415 → 0.844376,
на диагностическом: 0.816754 → 0.775068. Из-за этого компромисса по умолчанию
сохранён прежний `jev`; эксперимент включается явно через config.json.
Новый Selectel-прогон: micro-F1 0.842415 на исходных 208 строках и
0.816754 на 119 диагностических; среднее время run() на 10 входах — 1.026307 с.
Данные и ограничения — в бенчмарках ниже.
Варианты (например, `jev`, `embeddings` или `llm`) добавляются в
`src/<service>/<module>/`; затем в конфиге указывается имя папки вместо `null`.
Каждый вариант содержит `__init__.py` и `pipeline.py` с классом и интерфейсом
своего раздела (см. [AGENTS.md](AGENTS.md)). В CLI выбор задаётся конфигом,
в HTTP — query-параметрами extractor/generator с defaults из конфига.
Несуществующий модуль вызывает ошибку.

Для сравнения вариантов используем общие бенчмарки раздела и фиксируем
модуль в результатах. Скрипты переноса и обучения описаны в
[subcategory_extraction/README.md](src/subcategory_extraction/README.md);
исторические результаты с внешним SplitShift сохранены в
истории Git; невоспроизводимые старые скрипты доступны в Git.

## Бенчмарки выделения категорий

Ровно четыре проверки: [ручная стоимость токенов](src/subcategory_extraction/benchmarks/cpu_costs/README.md),
[F1 improved_dataset](src/subcategory_extraction/benchmarks/improved_dataset/README.md),
[F1 InnoGlobalHack](src/subcategory_extraction/benchmarks/classifier_methods/README.md)
и [время работы](src/subcategory_extraction/benchmarks/response_time/README.md).
Новые бенчмарки и папки не создаём.

```bash
uv run python src/subcategory_extraction/benchmarks/improved_dataset/run.py --replay
uv run python src/subcategory_extraction/benchmarks/classifier_methods/run.py --replay
```

Эти команды проверяют сохранённые ответы без API и обучения. Для оценки
текущих весов передайте `--methods <модуль>`. Исторический F1 модуля
`semantic_encoder_f1` на 119 диагностических строках — 0.857143;
это предварительная разметка, не независимый ручной эталон.
Опубликованные веса Leo включены в модуль через Git LFS (`git lfs pull`).
Загрузка и повторное обучение описаны в
[инструкции модуля](src/subcategory_extraction/semantic/semantic_encoder_f1/README.md).

Для [dataset_annotation](src/dataset_annotation/rubert_review/README.md) —
только время применения сохранённых решений и шесть примеров с mcId.

## HTTP API и Swagger

```bash
uv sync
uv run python -m src
```

Swagger: http://127.0.0.1:8000/docs. В `POST /drafts` выберите extractor и generator
из выпадающих списков; без выбора используются значения `config.json`.
`GET /modules` возвращает все обнаруженные модули и defaults, `GET /health` — состояние HTTP-сервера.

```bash
curl -X POST 'http://127.0.0.1:8000/drafts?extractor=lexical.regex&generator=sample_llm' \
  -H 'Content-Type: application/json' \
  -d '{"description":"Устанавливаю розетки и смесители.","mcId":101}'
```

Тело и результат соответствуют контракту кейса; имена модулей передаются отдельно в URL.
Один общий loader обслуживает оба шага. Сервер последовательно выполняет inference и
переиспользует активные экземпляры. Пустой список целей возвращает `drafts: []` без генерации.
API-ключи и веса нужны только при использовании соответствующего модуля.
Подробности, ошибки и офлайн-проверка: [docs/http-api.md](docs/http-api.md).

## Пайплайн

```
description + mcId
   │
   ▼
subcategory_extraction    список mcId найденных услуг (101–106)
   │
   ▼
draft_generation          черновик на каждую целевую категорию
   │
   ▼
{"drafts": [{"mcId": 104, "title": "...", "text": "..."}]}
```

Запуск:

```bash
uv sync
cp -n .env.example .env       # заполнить SELECTEL_API_KEY и LLM_API_KEY

uv run python src/main.py --input data/input.json --output out/output.json
```

Для полного запуска выберите реализации обоих шагов и настройте `.env`
(для генератора см. `src/draft_generation/README.md`).
Общий `LLMClient` — абстрактный интерфейс для конкретных клиентов.

## Оценка полного решения

```bash
uv sync
uv run python src/run.py --config run.config.json
```

`run.config.json` задаёт датасет (`.parquet` или официальный `.jsonl`),
путь к конфигу модулей, исходную категорию (101 для официальных данных),
путь отчёта, необязательный лимит строк и частоту логирования.
Все относительные пути считаются от файла конфигурации.
`pipeline_config` ссылается на обычный `config.json`: необходимо выбрать
оба рабочих модуля, включая генератор. При `null` оценка завершается
ошибкой, без подмены полного решения классификатором. Параметры моделей
и секреты выбранных модулей по-прежнему берутся из `.env`.

Метрика — `sklearn.metrics.f1_score(average="micro", zero_division=0)`
по множествам `mcId` из **итоговых drafts**, баллы за классификацию —
`60 × micro-F1` (не итоговый балл за все критерии хакатона).
Повторы не учитываются, ошибочный вывод исходной категории считается FP.
Если все эталоны и ответы пусты, micro-F1 равен 0.
Отчёт `out/evaluation.json` содержит метрику, TP/FP/FN, конфигурацию,
время, эталоны и полные ответы для каждой строки. Ошибка обработки
не превращается в пустой ответ: прогон прекращается без новой метрики.

`training_dataset` включает проверку нормализованных точных совпадений
описаний с train; пересечения отражаются в отчёте и предупреждении,
но не удаляются из оценки. Это не поиск всех близких дубликатов
и не гарантия отсутствия утечки. `null` отключает проверку.
Официальный test следует использовать для итоговой оценки после
фиксации модели и порогов, не для их подбора.

## Веб-сервис

Кроме CLI, проект поднимает веб-сервис черновиков: FastAPI-бэкенд
(`src/api/`), одностраничный клиент (`src/web/`, подробнее —
[src/web/README.md](src/web/README.md)) и справочник микрокатегорий
в PostgreSQL. Инфраструктура — `docker-compose.yml` и `deploy/`.

### Продакшен-запуск

```bash
cp .env.example .env            # заполнить SELECTEL_API_KEY
docker compose up -d --build
```

Веб-клиент: http://localhost:8081, API напрямую: http://localhost:8000.
nginx внутри контейнера `web` раздаёт статику и проксирует `/api/` в
сервис `api`. Хост-порт клиента задаётся переменной `WEB_PORT` (`.env`
или окружение compose) и по умолчанию равен 8081: порт 8080 на этой
машине занят другим сервисом.

### Эндпоинты

| Эндпоинт | Что делает |
| --- | --- |
| `POST /api/drafts` | контракт кейса: вход `{"description": "…", "mcId": 101}`, выход `{"drafts": [{"mcId": 104, "title": "…", "text": "…"}]}` |
| `GET /api/microcategories` | справочник из БД: `[{"mcId": 101, "title": "Ремонт под ключ"}, …]` |
| `GET /api/health` | статус API и БД (`ok`/`degraded`) и модель классификатора |
| `GET /api/dataset/samples` | страница объявлений датасета: `?split=train|test&limit=N≤100&offset=K` |
| `GET /api/dataset/random` | один случайный сэмпл: `?split=train|test` (404, если сплит пуст) |
| `GET /api/dataset/stats` | число объявлений по сплитам: `{"train": 10000, "test": 208}` |

Сбой шлюза LLM отвечает 502, недоступная база — 503; `detail` — русское
сообщение, его показывает интерфейс.

### Датасет кейса в БД

Официальные данные InnoGlobalHack × Avito (`public-train-annotated.jsonl`
— 10 000 объявлений с разметкой, `public-test-dataset.parquet` — 208
объявлений с названиями категорий) загружаются в таблицу `case_ads`
PostgreSQL и доступны через эндпоинты `/api/dataset/*`. Исходная категория
всех объявлений — 101; для train хранится `foundMcIds` (массив mcId из
`found_microcategories`), для test — `categories` (массив названий).
Схема создаётся автоматически (`CREATE TABLE IF NOT EXISTS` в лоадере и
в `deploy/db/init.sql`), загрузка идемпотентна: UNIQUE-ключ `row_key` и
`ON CONFLICT DO NOTHING`, повторный запуск не дублирует строки
(в test есть полные дубликаты текстов, поэтому ключ test-строк —
`test:{номер}:{хэш текста}`, а ключ train-строк — `train:{item_id}`).

Локально (БД из compose, код из репозитория):

```bash
docker compose up -d db
PYTHONPATH=src uv run python -m api.dataset_loader      # или: uv run python src/api/dataset_loader.py
# флаги: --data-dir (по умолчанию data/innoglobalhack_avito/innoglobalhack_avito)
#         --database-url (по умолчанию DATABASE_URL из окружения/.env)
```

Внутри compose (каталог `./data` смонтирован в контейнер `api` на read-only):

```bash
docker compose up -d --build          # пересоздать api с монтированием data/
docker compose exec api uv run --no-dev python src/api/dataset_loader.py
```

После загрузки — сэмпл из датасета можно прогнать через основной контракт
`POST /api/drafts`:

```bash
curl -s "http://localhost:8000/api/dataset/random?split=train" \
  | jq -c '{description, mcId: .sourceMcId}' \
  | curl -s -X POST http://localhost:8000/api/drafts \
      -H 'Content-Type: application/json' -d @-
```

### Автозагрузка датасета при старте compose

Сервис `db-load` при каждом `docker compose up` исполняет
`src/api/dataset_loader.py` (см. «Датасет кейса в БД») и завершается:
`restart: "no"`, `depends_on: db: service_healthy`, каталог
`docs/innoglobalhack_avito` смонтирован в лоадер только на чтение.
Повторный запуск идемпотентен (UNIQUE `row_key` + `ON CONFLICT DO
NOTHING`), на `api` и `web` сервис не влияет. Образ лоадера — отдельный
лёгкий `deploy/db-load.Dockerfile` без torch/CUDA: полный
`uv sync --frozen` для одноразовой загрузки не нужен. Повтор вручную:

```bash
docker compose run --rm db-load
docker compose exec db psql -U avito -d avito \
  -c "SELECT split, count(*) FROM case_ads GROUP BY split"
```

### Dev-запуск

```bash
docker compose up -d db
uv run uvicorn api.app:app --app-dir src --port 8000
cd src/web && npm install && npm run dev   # http://localhost:5173
```

Vite проксирует `/api/*` на `http://localhost:8000` (см.
`src/web/vite.config.ts`).

### Переменные окружения

Шаблон — `.env.example`, сервис читает корневой `.env`:

| Переменная | Назначение |
| --- | --- |
| `DATABASE_URL` | подключение к PostgreSQL (compose переопределяет на `db:5432`) |
| `SELECTEL_DECISIONS_URL`, `SELECTEL_DECISIONS_MODEL` | endpoint и модель классификатора (Selectel Decisions API, `typesafe/jev-1.13`, ключ — тот же `SELECTEL_API_KEY`) |
| `CLASSIFIER_THRESHOLD` | порог `p`: категории ниже порога не черновим |
| `SELECTEL_API_URL`, `SELECTEL_API_KEY`, `SELECTEL_MODEL` | генерация черновиков |
| `CLASSIFIER_TIMEOUT`, `CLASSIFIER_RETRIES` | таймаут и повторы запросов классификатора |
| `GENERATION_TIMEOUT`, `GENERATION_RETRIES` | таймаут и повторы генерации |
| `API_PORT`, `WEB_PORT` | хост-порты API и веб-клиента |

> **Примечание.** Классификатор — модель `jev` (`typesafe/jev-1.13`) через
> Selectel Decisions API (`POST /v1/decisions` с вопросами `noul`): она
> возвращает вероятности по микрокатегориям напрямую, без chat-формата.
> Текущая модель видна в `GET /api/health` в поле `classifier_model`.
> Генерация черновиков — `meta-llama/llama-3.2-3b-instruct` через Selectel
> Chat Completions, ключ используется тот же.

## Контрибьюшн

1. Ветка на каждую фичу: `<username>/feature/<slug>`. В `main` — только через PR.
2. Бэклог локальный: `backlog/{TODO,PROCESS,REVIEW,DONE}`,
   задача перемещается по папкам по мере работы.
3. Новые бенчмарки и папки для них не создаём. Используем существующие
   сценарии и обновляем их таблицы результатов; набор закреплён в AGENTS.md.
4. Правки в чужих сервисах — через свою ветку и с согласия автора.
5. Контракт входа/выхода не меняем без согласования со всей командой.

Подробнее — в [AGENTS.md](AGENTS.md).

## CPU и стоимость внешних моделей

Исторический замер на исходных 208 строках: RuBERT-tiny2 на CPU дал micro-F1
0.786704, медиану 3.58 мс на уникальный вход на Apple M4 Pro. Это дообученная
компактная модель, не TinyBERT. Замер не относится к новым весам `improved_dataset`.

Все API-клиенты теперь учитывают usage, повторы и локальный/provider cache.
Обычный CLI дополнительно сохраняет `<output>.costs.json`, включая ошибки;
`run.py` сохраняет расходы; в бенчмарках стоимость токенов вносится вручную. Тарифы задаются через
`LLM_PRICING_FILE` по точному endpoint/model; неизвестная цена — `null`, не ноль.

[Ручной учёт стоимости токенов](src/subcategory_extraction/benchmarks/cpu_costs/README.md).
