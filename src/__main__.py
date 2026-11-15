"""Запуск локального HTTP API: uv run python -m src."""

import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

SOURCE_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Запустить один процесс сервера с настройками host/port из окружения."""
    load_dotenv(SOURCE_DIR.parent / ".env")
    uvicorn.run(
        "api:create_app", factory=True, app_dir=str(SOURCE_DIR), workers=1,
        host=os.getenv("UVICORN_HOST", "127.0.0.1"), port=int(os.getenv("UVICORN_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
