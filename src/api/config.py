"""Параметры веб-сервиса черновиков: переменные окружения и .env.

Значения по умолчанию позволяют поднять приложение без ключей:
ошибки доступа к LLM возникают на запросах, а не на старте.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Корень репозитория: src/api/config.py -> ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Конфигурация api-сервиса (ENV: DATABASE_URL, SELECTEL_*, CLASSIFIER_* и другие)."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # PostgreSQL: строка подключения (в docker-compose host=db).
    database_url: str = "postgresql://avito:avito@localhost:5432/avito"

    # Selectel Decisions API: классификатор микрокатегорий (модель jev,
    # вопросы типа noul). Ключ общий с генерацией — selectel_api_key.
    selectel_decisions_url: str = "https://api.selectel.ru/aig/v1/decisions"
    selectel_decisions_model: str = "typesafe/jev-1.13"

    # Порог отнесения услуги к микрокатегории: p ниже порога не черновим.
    classifier_threshold: float = 0.85

    # Selectel: генерация черновиков.
    selectel_api_url: str = "https://api.selectel.ru/aig/v1/chat/completions"
    selectel_api_key: str = ""
    selectel_model: str = "meta-llama/llama-3.2-3b-instruct"

    # Таймауты HTTP-запросов (сек) и число повторов к LLM-шлюзам.
    classifier_timeout: float = 60.0
    classifier_retries: int = 2
    generation_timeout: float = 120.0
    generation_retries: int = 2

    # Порт FastAPI-приложения.
    api_port: int = 8000
