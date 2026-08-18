"""Loopback-only Ollama adapter for guarded presentation feedback."""

from __future__ import annotations

import json
from typing import Any

import requests


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
DEFAULT_MODEL = "presentcoach-local"
MAX_RESPONSE_BYTES = 1_000_000


class LocalPresentationAIError(RuntimeError):
    """Raised when local generation is unavailable or fails closed."""


class OllamaPresentationLLM:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        connect_timeout_seconds: float = 2.0,
        read_timeout_seconds: float = 150.0,
        session: requests.Session | None = None,
    ) -> None:
        if not model or any(character.isspace() for character in model):
            raise ValueError("Ollama model must be a non-empty token")
        self.model = model
        self._timeout = (connect_timeout_seconds, read_timeout_seconds)
        self._session = session or requests.Session()

    def complete_json(
        self, *, system: str, prompt: str, schema: dict[str, object]
    ) -> dict[str, object]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt + "\n/no_think"},
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        }
        try:
            response = self._session.post(
                OLLAMA_CHAT_URL,
                json=payload,
                timeout=self._timeout,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise LocalPresentationAIError("The local presentation model is unavailable") from error
        if response.is_redirect or response.status_code != 200:
            raise LocalPresentationAIError("The local presentation model rejected the request")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise LocalPresentationAIError("The local model response exceeded the size limit")
        try:
            envelope = response.json()
            document = json.loads(envelope["message"]["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            raise LocalPresentationAIError("The local model returned malformed JSON") from error
        if not isinstance(document, dict):
            raise LocalPresentationAIError("The local model output was not an object")
        return document

    def status(self) -> dict[str, object]:
        try:
            response = self._session.get(
                OLLAMA_TAGS_URL,
                timeout=(self._timeout[0], 5.0),
                allow_redirects=False,
            )
            models = response.json().get("models", []) if response.status_code == 200 and not response.is_redirect else []
        except (requests.RequestException, ValueError, AttributeError):
            models = []
        accepted = {self.model, f"{self.model}:latest"}
        installed = next((item for item in models if str(item.get("name", "")) in accepted), None)
        return {
            "available": installed is not None,
            "installed": installed is not None,
            "model": self.model,
            "digest": str(installed.get("digest", "")) if installed else "",
        }
