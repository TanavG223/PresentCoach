import json

import requests

from stroke_screening.presentation_ai import (
    OLLAMA_CHAT_URL,
    OLLAMA_TAGS_URL,
    OllamaPresentationLLM,
)


class FakeResponse:
    def __init__(self, document, *, content=b"{}"):
        self.status_code = 200
        self.is_redirect = False
        self.content = content
        self._document = document

    def json(self):
        return self._document


def test_production_ollama_session_ignores_hostile_proxy_environment(
    monkeypatch,
):
    proxy_url = "http://proxy.invalid:8080"
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("ALL_PROXY", proxy_url)
    monkeypatch.setenv("NO_PROXY", "")
    requests_seen = []

    def fake_post(session, url, **kwargs):
        requests_seen.append(("POST", session.trust_env, url, kwargs))
        output = {"strengths": [], "improvements": [], "insufficient_data": []}
        return FakeResponse(
            {"message": {"content": json.dumps(output)}},
            content=b"local-response",
        )

    def fake_get(session, url, **kwargs):
        requests_seen.append(("GET", session.trust_env, url, kwargs))
        return FakeResponse({"models": [{"name": "presentcoach-local"}]})

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr(requests.Session, "get", fake_get)
    llm = OllamaPresentationLLM()

    proxy_settings = llm._session.merge_environment_settings(
        OLLAMA_CHAT_URL, {}, None, None, None
    )
    output = llm.complete_json(system="system", prompt="prompt", schema={})
    status = llm.status()

    assert llm._session.trust_env is False
    assert proxy_settings["proxies"] == {}
    assert output == {
        "strengths": [], "improvements": [], "insufficient_data": [],
    }
    assert status["available"] is True
    assert [(method, url) for method, _trust, url, _kwargs in requests_seen] == [
        ("POST", OLLAMA_CHAT_URL),
        ("GET", OLLAMA_TAGS_URL),
    ]
    assert all(trust is False for _method, trust, _url, _kwargs in requests_seen)
    assert all(
        call[3]["allow_redirects"] is False for call in requests_seen
    )
