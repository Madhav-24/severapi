from __future__ import annotations

import hmac

from fastapi import Request


def valid_api_key(provided: str | None, expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
