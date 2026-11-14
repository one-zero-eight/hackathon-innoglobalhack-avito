"""Запуск HTTP-сервера командой python -m src из корня проекта."""

import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main() -> None:
    """Загрузить окружение проекта и запустить существующий FastAPI-адаптер."""
    load_dotenv(SOURCE_DIR.parent / ".env")
    uvicorn.run(
        "api:app",
        app_dir=str(SOURCE_DIR),
        host=os.getenv("UVICORN_HOST", DEFAULT_HOST),
        port=int(os.getenv("UVICORN_PORT", str(DEFAULT_PORT))),
    )


if __name__ == "__main__":
    main()
