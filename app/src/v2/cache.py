from __future__ import annotations

import json
from typing import Any

from app.src.v2.config import settings


class RedisCache:
    def __init__(self) -> None:
        self.enabled = False
        self._client = None
        if not settings.redis_url:
            return
        try:
            import redis

            self._client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self._client.ping()
            self.enabled = True
        except Exception:
            self._client = None
            self.enabled = False

    def get_json(self, key: str) -> Any | None:
        if not self.enabled or self._client is None:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        if not self.enabled or self._client is None:
            return False
        try:
            payload = json.dumps(value, ensure_ascii=False)
            if ttl_seconds:
                self._client.setex(key, ttl_seconds, payload)
            else:
                self._client.set(key, payload)
            return True
        except Exception:
            return False
