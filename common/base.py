"""Базовый класс пайплайна.

Один сервис — один класс с методом run(). Класс не делает ничего при импорте.
"""

from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """Оркестрируемый шаг пайплайна."""

    name: str = "base"

    @abstractmethod
    def run(self, *args, **kwargs):
        """Выполнить шаг пайплайна и вернуть результат."""
