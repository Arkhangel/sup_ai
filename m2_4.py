import hashlib
import json
import time
from typing import Any

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    redis_lib = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


def _make_key(model: str, messages: list, temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMCache:
    def __init__(self, ttl: int = 3600):
        self._ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, model: str, messages: list, temperature: float) -> Any | None:
        key = _make_key(model, messages, temperature)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(self, model: str, messages: list, temperature: float, response: Any) -> None:
        key = _make_key(model, messages, temperature)
        self._store[key] = (response, time.time() + self._ttl)

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total else 0.0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": f"{hit_rate:.1f}%"}


class RedisLLMCache:
    def __init__(self, ttl: int = 3600, host: str = "localhost", port: int = 6379, db: int = 0):
        if not REDIS_AVAILABLE:
            raise ImportError("redis-py is not installed. Run: uv add redis")
        self._ttl = ttl
        self._redis = redis_lib.Redis(host=host, port=port, db=db, decode_responses=True)  # type: ignore[union-attr]
        self._hits = 0
        self._misses = 0

    def get(self, model: str, messages: list, temperature: float) -> Any | None:
        key = _make_key(model, messages, temperature)
        raw = self._redis.get(key)
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        return json.loads(raw)

    def set(self, model: str, messages: list, temperature: float, response: Any) -> None:
        key = _make_key(model, messages, temperature)
        self._redis.setex(key, self._ttl, json.dumps(response))

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total else 0.0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": f"{hit_rate:.1f}%"}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def fake_llm_call(model: str, messages: list, temperature: float) -> dict:
    """Simulates an LLM API call (returns a stub response)."""
    print(f"  [API] calling {model}...")
    return {"model": model, "content": "Paris", "temperature": temperature}


def demo(cache: "LLMCache | RedisLLMCache") -> None:
    model = "gpt-4o"
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    temperature = 0.0

    print("=== Request 1 (cold) ===")
    result = cache.get(model, messages, temperature)
    if result is None:
        result = fake_llm_call(model, messages, temperature)
        cache.set(model, messages, temperature, result)
    print(f"  Response: {result}")

    print("=== Request 2 (same, should hit cache) ===")
    result = cache.get(model, messages, temperature)
    if result is None:
        result = fake_llm_call(model, messages, temperature)
        cache.set(model, messages, temperature, result)
    print(f"  Response: {result}")

    print("=== Request 3 (different temperature) ===")
    result = cache.get(model, messages, temperature=0.9)
    if result is None:
        result = fake_llm_call(model, messages, temperature=0.9)
        cache.set(model, messages, 0.9, result)
    print(f"  Response: {result}")

    print(f"\nStats: {cache.stats()}")


if __name__ == "__main__":
    print("--- In-memory cache ---")
    demo(LLMCache(ttl=3600))

    if REDIS_AVAILABLE:
        print("\n--- Redis cache ---")
        try:
            demo(RedisLLMCache(ttl=3600))
        except Exception as e:
            print(f"  Redis unavailable: {e}")
    else:
        print("\n--- Redis cache skipped (redis-py not installed) ---")