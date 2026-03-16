"""Optional API key authentication middleware.

If CTM_AUTH__API_KEY env var is set, all /v1/* requests must include
X-API-Key header matching that value. If not set, auth is disabled (dev mode).
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

_SKIP_PATHS = frozenset(
    [
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/v1/health",
    ]
)

_SKIP_PREFIXES = ("/ui",)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce X-API-Key header on /v1/* paths when auth is configured."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip non-protected paths
        if path in _SKIP_PATHS:
            return await call_next(request)
        for prefix in _SKIP_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        # Only enforce on /v1/* routes
        if not path.startswith("/v1/"):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != self._api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header"},
            )

        return await call_next(request)
