# Лёгкий образ FastAPI-бэкенда только под классификатор RuModernBERT (lexical.remont_mc_v04).
# Намеренно НЕ переиспользует deploy/api.Dockerfile: полный `uv sync --frozen`
# ставит torch/CUDA/scikit-learn (~8 ГБ), а инференс remont_mc_v04 идёт на CPU
# через onnxruntime + tokenizers + numpy (см. docstring student.py). Здесь только
# зависимости рантайма api и выбранного в config.json класса, и копируется только
# нужный код: другие модули extraction, их веса, benchmarks и docs в образ не идут.
FROM python:3.13-slim

# uv берём из официального образа astral-sh (как в deploy/db-load.Dockerfile)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Зависимости: веб-слой src/api (fastapi/uvicorn/asyncpg/pydantic/dotenv) и CPU-инференс
# remont_mc_v04 (onnxruntime/tokenizers/numpy). БЕЗ torch, transformers, scikit-learn,
# safetensors, pyarrow. Версии — в границах pyproject.toml и uv.lock (numpy 2.x).
# httpx в pyproject есть, но api-слой его не импортирует (клиент local_llm — urllib),
# поэтому в лёгкий образ не входит.
RUN uv pip install --system \
    "fastapi>=0.163.1" \
    "uvicorn>=0.54.0" \
    "asyncpg>=0.32.0" \
    "pydantic>=2" \
    "pydantic-settings>=2.16.3" \
    "python-dotenv>=1" \
    "onnxruntime>=1.17" \
    "tokenizers>=0.21,<0.24" \
    "numpy>=2,<3"

# Ядро (контракт, микрокатегории, load_pipeline/discover_modules) и веб-слой.
COPY src/main.py ./src/main.py
COPY src/api ./src/api

# Выбранный классификатор: код модуля + артефакты, включая model_int8.onnx (~150 МБ,
# gitignored, но присутствует в build-context) и tokenizer.json. __init__.py пакетов
# нужны, чтобы import subcategory_extraction.lexical.remont_mc_v04.pipeline работал.
COPY src/subcategory_extraction/__init__.py ./src/subcategory_extraction/__init__.py
COPY src/subcategory_extraction/lexical/__init__.py ./src/subcategory_extraction/lexical/__init__.py
COPY src/subcategory_extraction/lexical/remont_mc_v04 ./src/subcategory_extraction/lexical/remont_mc_v04

# Выбранный генератор: HTTP-клиент к локальному Chat Completions серверу + промпты.
COPY src/draft_generation/__init__.py ./src/draft_generation/__init__.py
COPY src/draft_generation/local_llm ./src/draft_generation/local_llm

# Конфигурация выбора модулей (extractor=lexical.remont_mc_v04, generator=local_llm):
# load_config() ищет config.json относительно src/main.py, то есть в корне /app.
COPY config.json ./

# Справочник микрокатегорий читает local_llm при инициализации (GUIDE_PATH =
# parents[3] от pipeline.py, то есть /app/docs/innoglobalhack_avito/...).
COPY docs/innoglobalhack_avito/microcategories-guide.md ./docs/innoglobalhack_avito/microcategories-guide.md

# Системный Python: venv/uv sync здесь нет, PATH не переопределяем.
EXPOSE 8000

# LOCAL_LLM_API_URL: по умолчанию compose задаёт http://host.docker.internal:8080/...
# (генератор ходит во внешний локальный LLM-сервер, в этом образе его нет).
CMD ["python", "-m", "uvicorn", "api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
