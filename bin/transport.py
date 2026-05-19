"""Generic transport helpers used by device-facing applications."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class JsonHttpClient:
    """Minimal JSON-over-HTTP client with an injectable request session."""

    def __init__(self, base_url: str, *, timeout: float = 1.0, session: Optional[Any] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        session = self.session
        if session is None:
            import requests

            session = requests
        response = session.post(
            f"{self.base_url}/{path.lstrip('/')}",
            json=dict(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
