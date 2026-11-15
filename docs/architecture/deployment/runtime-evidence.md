# Runtime evidence inventory

Скилл: `deployment-topology-analyzer` (+ Mermaid для диаграмм).
Срез: ветка `projacktor/main`, последний коммит `8d90028`, 2026-11-14.
Состояние: **current state по исходникам, конфигам и документации**. Runtime-телеметрии,
инвентаря облака и логов развёртывания в репозитории нет.

## 1. Границы анализа

Вопрос: где и как исполняется решение, какие внешние зависимости и локальные
артефакты ему нужны, как код попадает к запуску.

В репозитории **не найдено** ни одного артефакта развёртывания:

- `Dockerfile`, `docker-compose*.yml`, Kubernetes/Helm-манифестов;
- Terraform, Ansible, CloudFormation, Pulumi;
- `.github/workflows`, `.gitlab-ci.yml`, Jenkinsfile, иного CI;
- systemd/unit-файлов, runbook'ов, конфигов облака или CDN.

Проверка: полный список файлов (`find . -path ./.git -prune -o -type f -print`)
и каталогов; скрытые каталоги верхнего уровня отсутствуют, кроме `.git`.
Следовательно, «развёртывание» здесь — это локальный запуск CLI на машине
разработчика/жюри, а поставка — исходники через Git.

## 2. Runtime-единицы

| Единица | Тип | Как запускается | Статус |
| --- | --- | --- | --- |
| `src/main.py` | CLI-процесс | `uv run python src/main.py --input … --output …` | активен |
| `subcategory_extraction/jev` | in-process модуль | импортируется оркестратором по `config.json` | активен (`config.json:2`) |
| `draft_generation/sample_llm` | in-process модуль | импортируется оркестратором | активен (`config.json:3`) |
| 8 альтернативных классификаторов | in-process модули | выбор через `config.json` | доступны, не выбраны |
| `src/run.py` | процесс оценки | `--config run.config.json` | вспомогательный |
| `jev_tfidf/train.py`, `semantic_encoder/train.py`, `benchmarks/tfidf_quality/run.py` | процессы обучения | `python -m …` | вспомогательные |
| `src/**/benchmarks/*/run.py` | процессы бенчмарков | `uv run python …` | вспомогательные |
| `src/dataset_annotation/*` | офлайн-скрипты данных | `python …` | вспомогательные |

Оркестрация: `src/main.py:227-284` (`DEFAULT_CONFIG_PATH`, `load_config`,
`load_pipeline`, `run_pipeline`); CLI — `src/main.py:290-315`.

## 3. Внешние зависимости (HTTPS egress)

| Сервис | Endpoint по умолчанию | Аутентификация | Таймаут / повторы | Доказательство |
| --- | --- | --- | --- | --- |
| BestAPI, Jev | `POST https://bestapi.fugue.pro/v1/systemone` | `Authorization: Bearer BESTAPI_API_KEY` | 60 с, повторов нет | `src/subcategory_extraction/jev/client.py:13-30,93-105` |
| BestAPI, structured LLM | `POST https://bestapi.fugue.pro/v1/chat/completions` | `Bearer BESTAPI_API_KEY` | 90 с, повторов нет | `src/main.py:135-151,189-196` |
| Google Decision | `POST https://generativelanguage.googleapis.com/v1beta/decisions` | `x-goog-api-key` (`LLM_API_KEY` или `GOOGLE_API_KEY`) | 30 с; повтор 429/503 через 60 с, 3 попытки; бюджет 45k входных токенов/мин | `src/subcategory_extraction/google_decision/client.py:12-13,19-41,44-86` |
| Google GenerateContent (черновики) | `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` | `x-goog-api-key` (`LLM_API_KEY`) | 60 с; повтор 429/500/502/503/504, 4 попытки, пауза 10 с | `src/draft_generation/sample_llm/client.py:12-31,33-73` |
| Hugging Face Hub | `cointegrated/rubert-tiny2`, revision `e8ed3b0c…` | не задана в коде | только при обучении | `semantic_encoder/model.py:18-19`; `semantic_encoder/train.py:77-80` |
| PyPI | через `uv sync` по `uv.lock` | — | только при установке | `pyproject.toml:9-18`, `uv.lock` |

Активная конфигурация (`jev` + `sample_llm`) требует сеть минимум к BestAPI и Google.
Локального/офлайн-классификатора в активной конфигурации нет.

## 4. Локальное состояние и артефакты

| Объект | Путь | Роль | Коммитится |
| --- | --- | --- | --- |
| Выбор модулей | `config.json` | единственный переключатель шагов | да |
| Конфиг оценки | `run.config.json` | датасет, лимит, путь отчёта | да |
| Секреты | `.env` | ключи API | нет (`.gitignore:1-5`) |
| Датасеты | `docs/innoglobalhack_avito/public-*.{jsonl,parquet}` | train/test | да |
| TF-IDF веса | `out/tfidf.json` | нужны модулям `tfidf`, `jev_tfidf`, `jev_arbiter` | нет |
| Filtered train | `out/hybrid-training.jsonl` | нужен `jev_retrieval`, обучению encoder | нет |
| Encoder | `out/semantic-encoder/` | нужен `semantic_encoder`; inference только `local_files_only=True` | нет |
| Кэш Jev | `out/jev-cache/` | ответы API по SHA-256 ключу | нет |
| Кэш structured LLM | `out/structured-llm-cache/` | ответы API по SHA-256 ключу | нет |
| Отчёт оценки | `out/evaluation.json` | micro-F1 и все ответы | нет |

`out/` целиком в `.gitignore`. После чистого клона активная конфигурация
(`jev` + `sample_llm`) работает без локальных артефактов, но `tfidf`,
`jev_tfidf`, `jev_arbiter`, `jev_retrieval`, `semantic_encoder` падают
с явной ошибкой до первого запроса (`tfidf/pipeline.py:22-27`,
`semantic_encoder/model.py:185-203`).

## 5. Переменные окружения

Читаются кодом; в `.env.example` перечислены не все.

| Группа | Переменные |
| --- | --- |
| Общие | `LLM_MODEL_ID`, `LLM_API_KEY` |
| BestAPI / Jev | `BESTAPI_API_KEY`, `JEV_MODEL`, `JEV_ENDPOINT`, `JEV_THRESHOLD`, `JEV_TIMEOUT`, `JEV_CACHE_DIR`, `JEV_PROMPT_PATH` |
| Гибрид | `JEV_TFIDF_WEIGHT`, `JEV_TFIDF_THRESHOLD` |
| Retrieval | `JEV_REFERENCE_DATASET`, `JEV_REFERENCE_MAX_CHARS` |
| Structured LLM | `STRUCTURED_LLM_MODEL`, `STRUCTURED_LLM_ENDPOINT`, `STRUCTURED_LLM_TIMEOUT`, `STRUCTURED_LLM_CACHE_DIR` |
| Google classifier | `GOOGLE_API_KEY`, `GOOGLE_DECISION_PROMPT`, `GOOGLE_DECISION_THRESHOLD`, `GOOGLE_DECISION_ENDPOINT` |
| Генератор | `SAMPLE_LLM_MODEL_ID`, `SAMPLE_LLM_API_URL`, `SAMPLE_LLM_TIMEOUT`, `SAMPLE_LLM_MAX_ATTEMPTS`, `SAMPLE_LLM_RETRY_DELAY` |
| Локальные модели | `TFIDF_ARTIFACT_PATH`, `SEMANTIC_ENCODER_ARTIFACT_PATH`, `SEMANTIC_ENCODER_DEVICE` |

Не описаны в `.env.example`, но читаются кодом: `GOOGLE_API_KEY`,
`GOOGLE_DECISION_ENDPOINT`, `JEV_TIMEOUT`, `JEV_CACHE_DIR`, `JEV_PROMPT_PATH`,
`STRUCTURED_LLM_TIMEOUT`, `STRUCTURED_LLM_CACHE_DIR`, `TFIDF_ARTIFACT_PATH`,
`SAMPLE_LLM_MODEL_ID`, `SAMPLE_LLM_API_URL`, `SAMPLE_LLM_TIMEOUT`,
`SAMPLE_LLM_MAX_ATTEMPTS`, `SAMPLE_LLM_RETRY_DELAY`.
Это не блокер: у всех есть значения по умолчанию.

## 6. Платформа и зависимости

- Python 3.13: `.python-version:1`; `requires-python = ">=3.13"` (`pyproject.toml:9`).
- Менеджер зависимостей: `uv` + зафиксированный `uv.lock` (`README.md:123-127`).
- Ключевые библиотеки: `torch`, `transformers`, `safetensors`, `scikit-learn`,
  `pyarrow`, `pydantic`, `python-dotenv` (`pyproject.toml:10-18`).
- Устройство вычислений: `cpu` или `mps`; явный запрос недоступного `mps`
  завершается ошибкой, `auto` откатывается на CPU (`semantic_encoder/model.py:50-57`).
  Ветки CUDA в коде нет. Обучение encoder в README заявлено на MPS (`README.md:88`).
- `scipy.sparse` импортируется напрямую (`jev_retrieval/references.py:9`),
  но прямая зависимость в `pyproject.toml` не объявлена; сейчас доступна
  транзитивно через `scikit-learn`.

## 7. Поставка и оценка

- Ветвление и PR: `AGENTS.md` §5, §7. Автоматических проверок в CI нет;
  бенчмарки запускаются вручную.
- Контракт кейса — файловый CLI (`{"description","mcId"}` → `{"drafts":[…]}`),
  HTTP-сервера в репозитории нет (`AGENTS.md` §1-3, `src/main.py:290-315`).
- Live-демо и работоспособность — до 25 баллов; нужно назвать модели, время
  и стоимость (`docs/innoglobalhack_avito/hackathon-evaluation.md:45-55`).
- Отказ предоставленного BestAPI-ключа (HTTP 401) уже зафиксирован в
  `README.md:85-88`; это операционный, а не архитектурный риск.

## 8. Как перепроверить

```bash
git rev-parse HEAD
uv sync --frozen
uv run python src/main.py --input data/input.json --output out/smoke.json

# артефакты не в Git
git check-ignore -v out/tfidf.json out/semantic-encoder out/jev-cache
ls -d out/tfidf.json out/hybrid-training.jsonl out/semantic-encoder 2>/dev/null

# отсутствие средств развёртывания
find . -path ./.git -prune -o \( -iname 'Dockerfile*' -o -iname 'docker-compose*' \
  -o -iname '*.tf' -o -path '*/.github/workflows/*' -o -iname '.gitlab-ci.yml' \) -print

# диаграммы (нужен mermaid-cli или любое Mermaid-превью)
npx -y @mermaid-js/mermaid-cli \
  -i docs/architecture/deployment/deployment-topology.mmd -o /tmp/topology.svg
npx -y @mermaid-js/mermaid-cli \
  -i docs/architecture/deployment/release-path.mmd -o /tmp/release.svg
```
