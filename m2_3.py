"""
RobustLLMClient — надёжный клиент для LLM API.

Поддерживает:
- Retry с exponential backoff + jitter (tenacity)
- Fallback-цепочку: OpenAI → OpenRouter → fallback-сообщение
- Circuit breaker: 3 ошибки подряд → пауза 60 сек
- Rate limiter на стороне клиента
- Трекинг usage и стоимости
- Структурированное логирование
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import openai
from openai import OpenAI
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("RobustLLMClient")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    # Pricing per 1k tokens (USD). Rough defaults — override as needed.
    prompt_price_per_1k: float = 0.005
    completion_price_per_1k: float = 0.015

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1000 * self.prompt_price_per_1k
            + self.completion_tokens / 1000 * self.completion_price_per_1k
        )

    def update(self, usage) -> None:
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def __str__(self) -> str:
        return (
            f"prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, "
            f"total_tokens={self.total_tokens}, "
            f"cost≈${self.estimated_cost_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Open after `failure_threshold` consecutive errors; resets after `recovery_timeout` sec."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if datetime.now(timezone.utc) >= self._opened_at + timedelta(seconds=self.recovery_timeout):
            logger.info("Circuit breaker: recovery timeout elapsed, closing circuit.")
            self._failures = 0
            self._opened_at = None
            return False
        return True

    def record_failure(self, provider: str) -> None:
        self._failures += 1
        logger.warning(
            "Circuit breaker [%s]: failure %d/%d",
            provider, self._failures, self.failure_threshold,
        )
        if self._failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = datetime.now(timezone.utc)
            logger.error(
                "Circuit breaker [%s]: OPEN — skipping for %.0f sec.",
                provider, self.recovery_timeout,
            )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None


# ---------------------------------------------------------------------------
# Rate Limiter (token bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter.

    Args:
        max_calls: maximum calls per `period` seconds.
        period: window size in seconds.
    """

    def __init__(self, max_calls: int = 10, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        # Drop timestamps outside the window
        while self._timestamps and now - self._timestamps[0] >= self.period:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_calls:
            sleep_for = self.period - (now - self._timestamps[0])
            logger.info("Rate limiter: sleeping %.2f sec before next call.", sleep_for)
            time.sleep(sleep_for)
            # Re-clean after sleep
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] >= self.period:
                self._timestamps.popleft()

        self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Retryable errors
# ---------------------------------------------------------------------------

_RETRYABLE = (
    openai.RateLimitError,
    openai.APIStatusError,       # catches 500, 502, 503 …
    openai.APITimeoutError,
    openai.APIConnectionError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, _RETRYABLE)


# ---------------------------------------------------------------------------
# Provider wrapper
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    name: str
    client: OpenAI
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


# ---------------------------------------------------------------------------
# RobustLLMClient
# ---------------------------------------------------------------------------

class RobustLLMClient:
    """Robust wrapper over OpenAI-compatible APIs.

    Fallback order: OpenAI → OpenRouter → hardcoded message.

    Args:
        openai_api_key: OpenAI API key (None → skip provider).
        openrouter_api_key: OpenRouter API key (None → skip provider).
        max_retries: max retry attempts per provider (default 5).
        rate_limit_calls: max calls per `rate_limit_period` seconds.
        rate_limit_period: window for rate limiter.
        circuit_failure_threshold: consecutive failures before circuit opens.
        circuit_recovery_timeout: seconds the circuit stays open.
        request_timeout: per-request HTTP timeout in seconds.
    """

    FALLBACK_MESSAGE = "Сервис временно недоступен. Попробуйте позже."

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        openrouter_model: str = "openai/gpt-4o-mini",
        max_retries: int = 5,
        rate_limit_calls: int = 10,
        rate_limit_period: float = 60.0,
        circuit_failure_threshold: int = 3,
        circuit_recovery_timeout: float = 60.0,
        request_timeout: float = 30.0,
    ):
        self.openai_model = openai_model
        self.openrouter_model = openrouter_model
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.usage = UsageStats()

        self._rate_limiter = RateLimiter(max_calls=rate_limit_calls, period=rate_limit_period)

        self._providers: list[Provider] = []
        if openai_api_key:
            self._providers.append(Provider(
                name="OpenAI",
                client=OpenAI(api_key=openai_api_key, timeout=request_timeout),
                circuit_breaker=CircuitBreaker(circuit_failure_threshold, circuit_recovery_timeout),
            ))
        if openrouter_api_key:
            self._providers.append(Provider(
                name="OpenRouter",
                client=OpenAI(
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=request_timeout,
                ),
                circuit_breaker=CircuitBreaker(circuit_failure_threshold, circuit_recovery_timeout),
            ))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat request through the fallback chain.

        Returns the assistant reply text, or FALLBACK_MESSAGE if all providers fail.
        """
        self._rate_limiter.acquire()
        self.usage.total_requests += 1

        for provider in self._providers:
            if provider.circuit_breaker.is_open:
                logger.warning("Skipping provider %s — circuit is OPEN.", provider.name)
                continue

            result = self._call_with_retry(provider, messages, **kwargs)
            if result is not None:
                provider.circuit_breaker.record_success()
                return result

        self.usage.failed_requests += 1
        logger.error("All providers exhausted. Returning fallback message.")
        return self.FALLBACK_MESSAGE

    def print_usage(self) -> None:
        print(f"\n[Usage] {self.usage}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_with_retry(
        self, provider: Provider, messages: list[dict], **kwargs
    ) -> Optional[str]:
        """Try a single provider with exponential backoff. Returns None on total failure."""
        attempt_state = {"n": 0}

        # Build a closure so tenacity can log the provider name per attempt.
        @retry(
            retry=retry_if_exception_type(tuple(_RETRYABLE)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=16) + wait_random(0, 0.5),
            before_sleep=self._make_before_sleep(provider.name, attempt_state),
            reraise=False,
        )
        def _do_call():
            attempt_state["n"] += 1
            return self._single_call(provider, messages, **kwargs)

        try:
            return _do_call()
        except RetryError as exc:
            logger.error(
                "[%s] All %d retry attempts failed. Last error: %s",
                provider.name, self.max_retries, exc.last_attempt.exception(),
            )
            provider.circuit_breaker.record_failure(provider.name)
            return None
        except Exception as exc:  # non-retryable (e.g. 401 AuthError)
            logger.error("[%s] Non-retryable error: %s", provider.name, exc)
            return None

    def _single_call(self, provider: Provider, messages: list[dict], **kwargs) -> str:
        model = self.openrouter_model if provider.name == "OpenRouter" else self.openai_model
        try:
            response = provider.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            self.usage.update(response.usage)
            return response.choices[0].message.content
        except openai.APIStatusError as exc:
            logger.warning(
                "[%s] HTTP %d — %s",
                provider.name, exc.status_code, exc.message,
            )
            if not _is_retryable(exc):
                raise  # will bypass tenacity retry
            raise

    @staticmethod
    def _make_before_sleep(provider_name: str, attempt_state: dict):
        def _before_sleep(retry_state):
            exc = retry_state.outcome.exception()
            code = getattr(exc, "status_code", type(exc).__name__)
            wait = retry_state.next_action.sleep if retry_state.next_action else "?"
            logger.warning(
                "[%s] Attempt %d failed (error=%s). Retrying in %.2fs…",
                provider_name,
                attempt_state["n"],
                code,
                wait if isinstance(wait, float) else 0,
            )
        return _before_sleep


# ---------------------------------------------------------------------------
# Demo script
# ---------------------------------------------------------------------------

def _demo():
    """Demonstrates retry and fallback behaviour with fake/mock keys."""
    import os

    print("=" * 64)
    print("RobustLLMClient — демонстрация retry и fallback")
    print("=" * 64)

    # Use env vars if set, otherwise use dummy keys to trigger failures
    openai_key = os.getenv("OPENAI_API_KEY", "sk-invalid-openai-key")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "sk-or-invalid-openrouter-key")

    client = RobustLLMClient(
        openai_api_key=openai_key,
        openrouter_api_key=openrouter_key,
        max_retries=3,              # keep demo short
        rate_limit_calls=20,
        circuit_failure_threshold=3,
        circuit_recovery_timeout=10,
        request_timeout=10.0,
    )

    messages = [{"role": "user", "content": "Привет! Скажи одно слово."}]

    print("\n--- Запрос 1: обычный вызов ---")
    reply = client.chat(messages)
    print(f"Ответ: {reply}")

    print("\n--- Запрос 2: ещё один вызов (проверка rate limiter) ---")
    reply = client.chat(messages)
    print(f"Ответ: {reply}")

    print("\n--- Запрос 3: принудительный circuit breaker (3 ошибки) ---")
    # Monkey-patch a provider to always raise RateLimitError
    if client._providers:
        original_create = client._providers[0].client.chat.completions.create
        call_count = {"n": 0}

        def _always_fail(*args, **kwargs):
            call_count["n"] += 1
            raise openai.RateLimitError(
                message="Simulated 429",
                response=httpx.Response(429),
                body={"error": {"message": "Simulated 429"}},
            )

        client._providers[0].client.chat.completions.create = _always_fail
        reply = client.chat(messages)
        print(f"Ответ после принудительных ошибок: {reply}")
        # Restore
        client._providers[0].client.chat.completions.create = original_create

    client.print_usage()
    print("\nДемонстрация завершена.")


if __name__ == "__main__":
    _demo()