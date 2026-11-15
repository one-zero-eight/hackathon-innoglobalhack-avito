# Образ одноразового лоадера db-load (сервис db-load в docker-compose.yml).
# Намеренно НЕ переиспользует deploy/api.Dockerfile: лоадеру (dataset_loader)
# нужны только pyarrow/asyncpg/pydantic/dotenv, а полный `uv sync --frozen`
# по общему uv.lock ставит torch + CUDA-колёса (~8 ГБ) — такое не помещается
# на диск сборки и раздувает образ. Версии ограничены теми же границами,
# что в pyproject.toml; копируются только файлы, нужные лоадеру (см. ниже).
FROM python:3.13-slim

# uv берём из официального образа astral-sh (как в deploy/api.Dockerfile)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Зависимости src/api/dataset_loader.py: pyarrow — parquet-файл теста,
# asyncpg — пул подключения (api.db), pydantic и pydantic-settings —
# api.config и src/main.py, python-dotenv — переменные окружения.
RUN uv pip install --system \
    "asyncpg>=0.32.0" \
    "pyarrow>=23" \
    "pydantic>=2" \
    "pydantic-settings>=2.16.3" \
    "python-dotenv>=1"

# Копируем только то, что импортирует dataset_loader: пакет src/api/ и
# src/main.py. Всё src/ целиком копировать нельзя — там же веса моделей
# (artifacts/, ~449 МБ Git LFS), которые одноразовому лоадеру не нужны.
COPY src/api ./src/api
COPY src/main.py ./src/main.py

# --data-dir указывает на каталог, который compose монтирует в контейнер.
CMD ["python", "src/api/dataset_loader.py", "--data-dir", "/app/docs/innoglobalhack_avito"]
