"""Ошибки обращения к LLM-шлюзу для веб-сервиса черновиков."""


class LLMGatewayError(Exception):
    """Сбой обращения к LLM-шлюзу: не-2xx ответ, не-JSON тело или нарушенная структура.

    Атрибуты:
        message — человекочитаемое пояснение ошибки.
        snippet — обрезок тела ответа шлюза (не длиннее SNIPPET_MAX_LEN символов),
            пустая строка, если тела не было.
    """

    SNIPPET_MAX_LEN: int = 200

    def __init__(self, message: str, snippet: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.snippet = (snippet or "")[: self.SNIPPET_MAX_LEN]

    def __str__(self) -> str:
        if self.snippet:
            return f"{self.message} | фрагмент ответа: {self.snippet!r}"
        return self.message
