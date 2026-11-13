"""Заглушка клиента jev-модели.

Точный model id и ключ берём из окружения, не хардкодим:
    LLM_MODEL_ID — идентификатор модели (по умолчанию "jev"),
    LLM_API_KEY  — ключ доступа (в репозиторий не коммитим).
"""

import logging
import os

logger = logging.getLogger(__name__)


class LLMClient:
    """Минимальный клиент LLM. TODO: реализовать вызов jev-модели."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "jev")
        self.api_key = os.getenv("LLM_API_KEY")

    def complete(self, prompt: str, **kwargs) -> str:
        """TODO: заменить заглушку на реальный вызов jev-модели."""
        raise NotImplementedError("LLMClient.complete: вызов jev-модели не реализован")
