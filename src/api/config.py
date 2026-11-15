"""Параметры веб-сервиса черновиков: переменные окружения и .env.

Модель, порог, таймауты и повторы обращений к LLM принадлежат выбранным
модулям пайплайна и их настройкам в ``.env`` ядра; здесь остаются только
подключение к БД и порт приложения.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень репозитория: src/api/config.py -> ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Конфигурация api-сервиса (ENV: DATABASE_URL, API_PORT и другие)."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # PostgreSQL: строка подключения (в docker-compose host=db).
    database_url: str = "postgresql://avito:avito@localhost:5432/avito"

    # Порт FastAPI-приложения.
    api_port: int = 8000
