# Образ FastAPI-сервиса черновиков.
# Само приложение src/api пишет другой воркер; здесь только инфраструктура.
FROM python:3.13-slim

# uv берём из официального образа astral-sh
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Сначала только манифест зависимостей — слой кэшируется отдельно от кода
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Затем код проекта (src/api появится внутри src/)
COPY src ./src
COPY config.json ./
COPY docs/innoglobalhack_avito/microcategories-guide.md ./docs/innoglobalhack_avito/microcategories-guide.md

# Собственное окружение проекта вместо системного Python
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Код оркестрации (core) остаётся CLI-ядром; веб-слой — api.app в src/
CMD ["uv", "run", "--no-dev", "uvicorn", "api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
