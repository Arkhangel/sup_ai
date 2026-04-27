#!/usr/bin/env python3
"""
Support Bot LLM Benchmark
=========================
Runs eval tasks against Ollama (local) and Anthropic (cloud) models,
measures TTFT / throughput / accuracy, and prints a cost report.

Usage:
    python benchmark.py                        # run all models
    python benchmark.py --models qwen3:8b      # single model
    python benchmark.py --dry-run              # show tasks only
    python benchmark.py --output results.json  # save raw results
"""

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import requests

# ─── optional deps (graceful degradation) ────────────────────────────────────
try:
    import anthropic as _anthropic_sdk
    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    class _FallbackConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, title=""): print(f"\n{'─'*60} {title}")
    console = _FallbackConsole()


# ══════════════════════════════════════════════════════════════════
#  EVAL TASKS
# ══════════════════════════════════════════════════════════════════

@dataclass
class Task:
    id: int
    category: str
    prompt: str
    check: Callable[[str], bool]
    check_type: str          # "exact" | "llm"
    check_desc: str          # human-readable description of the check


def _c(*words) -> Callable[[str], bool]:
    """All words must appear in response (case-insensitive)."""
    def fn(r: str) -> bool:
        r = r.lower()
        return all(w.lower() in r for w in words)
    return fn

def _any(*words) -> Callable[[str], bool]:
    """At least one word must appear."""
    def fn(r: str) -> bool:
        r = r.lower()
        return any(w.lower() in r for w in words)
    return fn

def _not(check: Callable, *words) -> Callable[[str], bool]:
    """check passes AND none of the excluded words appear."""
    def fn(r: str) -> bool:
        r_low = r.lower()
        return check(r) and not any(w.lower() in r_low for w in words)
    return fn

def _lang_check(keyword: str, lang_sample: str) -> Callable[[str], bool]:
    """keyword present AND response language matches sample."""
    def fn(r: str) -> bool:
        return keyword.lower() in r.lower() and lang_sample.lower() in r.lower()
    return fn

def _len_check(min_sentences: int) -> Callable[[str], bool]:
    """Response contains at least N sentences."""
    def fn(r: str) -> bool:
        sentences = [s.strip() for s in re.split(r'[.!?]+', r) if s.strip()]
        return len(sentences) >= min_sentences
    return fn

def _phone_or_keyword(keyword: str) -> Callable[[str], bool]:
    """Phone number pattern OR keyword present."""
    def fn(r: str) -> bool:
        has_phone = bool(re.search(r'\+?\d[\d\s\-\(\)]{6,}', r))
        has_kw = keyword.lower() in r.lower()
        return has_phone or has_kw
    return fn


TASKS: list[Task] = [
    # ── Сброс пароля ──────────────────────────────────────────────
    Task(1,  "Пароль",    "Я забыл пароль, как мне войти в аккаунт?",
         _any("сброс", "восстановл", "письм", "reset"),
         "exact", 'contains("сброс"/"письм"/"reset")'),

    Task(2,  "Пароль",    "Кнопка «Забыл пароль» не работает, что делать?",
         _c("почт", "ссылк"),
         "exact", 'contains("почт") AND contains("ссылк")'),

    Task(3,  "Пароль",    "Я нажал «Сброс пароля», но письмо так и не пришло.",
         _any("спам", "нежелательн", "папк", "junk"),
         "exact", 'contains("спам"/"нежелательн"/"папк")'),

    # ── Аккаунт ───────────────────────────────────────────────────
    Task(4,  "Аккаунт",  "Как изменить email-адрес в моём профиле?",
         _c("настройк", "профил"),
         "exact", 'contains("настройк") AND contains("профил")'),

    Task(5,  "Аккаунт",  "Мой аккаунт заблокирован после 5 неудачных попыток входа.",
         _len_check(3),
         "llm",   "ответ ≥ 3 предложений с причиной и шагами"),

    Task(6,  "Аккаунт",  "Можно ли зарегистрировать два аккаунта на один email?",
         _any("нет", "невозможно", "один аккаунт", "запрещ"),
         "exact", 'contains("нет"/"невозможно"/"один аккаунт")'),

    # ── Оплата ────────────────────────────────────────────────────
    Task(7,  "Оплата",   "Как оплатить подписку банковской картой?",
         _any("visa", "mastercard", "карт", "card"),
         "exact", 'contains("Visa"/"Mastercard"/"карт")'),

    Task(8,  "Оплата",   "С меня списали деньги дважды за одну и ту же подписку!",
         _any("возврат", "refund", "вернём", "верн"),
         "llm",   "упоминает возврат + срок"),

    Task(9,  "Оплата",   "Я оплатил, но чек на почту не пришёл.",
         _any("заказ", "история платеж", "личн кабинет", "раздел"),
         "exact", 'contains("заказ"/"история"/"личн кабинет")'),

    Task(10, "Оплата",   "Как подключить автоматическое продление подписки?",
         _len_check(3),
         "llm",   "пошаговая инструкция ≥ 3 шагов"),

    # ── Технические ошибки ────────────────────────────────────────
    Task(11, "Тех. ошибка", "Получаю ошибку 500 при попытке зайти на сайт.",
         _not(_any("повтор", "попробуй", "обновит"), "ничего не могу", "не знаю"),
         "exact", 'contains("повторить") NOT contains("ничего не могу")'),

    Task(12, "Тех. ошибка", "Приложение постоянно вылетает на iPhone с iOS 17.",
         _any("обновл", "версию", "переустанов", "update"),
         "llm",   "предлагает обновление/переустановку"),

    Task(13, "Тех. ошибка", "Сайт не открывается в браузере Firefox.",
         _any("кэш", "cache", "другой браузер", "очист"),
         "exact", 'contains("кэш"/"cache"/"другой браузер")'),

    Task(14, "Тех. ошибка", "Не приходят push-уведомления на телефон.",
         _any("разрешен", "настройк", "уведомлен", "permission"),
         "llm",   "проверяет разрешения + настройки профиля"),

    Task(15, "Тех. ошибка", "В интерфейсе нет кнопки «Скачать», куда она пропала?",
         _any("подписк", "тариф", "план", "premium"),
         "exact", 'contains("подписк"/"тариф"/"план")'),

    # ── Контент ───────────────────────────────────────────────────
    Task(16, "Контент",  "Как удалить мой отзыв, который я оставил по ошибке?",
         _any("профил", "отзыв", "удалит", "личн"),
         "llm",   "корректный путь в интерфейсе"),

    Task(17, "Контент",  "Я загрузил не тот файл, как его заменить?",
         _c("удалит", "загрузит"),
         "exact", 'contains("удалить") AND contains("загрузить")'),

    Task(18, "Контент",  "Почему мой пост был удалён без предупреждения?",
         _any("правил", "политик", "условия", "terms"),
         "llm",   "ссылается на правила, не агрессивен"),

    # ── Эскалация ─────────────────────────────────────────────────
    Task(19, "Эскалация", "Я хочу поговорить с живым оператором, а не с ботом.",
         _any("оператор", "специалист", "связаться", "support@", "чат с"),
         "exact", 'contains("оператор"/"специалист"/"связаться")'),

    Task(20, "Эскалация", "Ваш бот полностью бесполезен! Это просто ужас!",
         _not(_any("понимаю", "сожалею", "помогу", "sorry", "извините"),
              "не могу помочь", "сам разберись"),
         "llm",   "не оборонительный тон, предлагает помощь"),

    Task(21, "Эскалация", "Моя проблема не решается уже третий день, это неприемлемо!",
         _any("понимаю", "сожалею", "оператор", "приоритет", "escalat"),
         "llm",   "выражает сочувствие + эскалирует"),

    # ── Английские запросы ────────────────────────────────────────
    Task(22, "Пароль EN", "My password reset link is not working. What should I do?",
         _lang_check("reset", "email"),
         "exact", 'contains("reset") AND contains("email") in English'),

    Task(23, "Оплата EN", "I was charged for the wrong subscription plan. Can I get a refund?",
         _any("refund", "return", "contact", "support"),
         "llm",   "mentions refund + escalation in English"),

    # ── Out-of-scope ──────────────────────────────────────────────
    Task(24, "Out-of-scope", "Как мне позвонить в службу поддержки?",
         _phone_or_keyword("номер"),
         "exact", "phone number OR contains('номер')"),
]


# ══════════════════════════════════════════════════════════════════
#  MODEL ADAPTERS
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Ты — агент техподдержки. Отвечай кратко, вежливо и по делу. "
    "Если вопрос вне твоей компетенции — скажи честно и предложи альтернативу. "
    "Отвечай на том же языке, что и пользователь. "
    "/no_think"          # disables Qwen3 thinking mode
)


@dataclass
class RunResult:
    task_id: int
    model: str
    prompt: str
    response: str
    ttft_s: float
    total_s: float
    output_tokens: int
    passed: bool
    error: Optional[str] = None

    @property
    def throughput(self) -> float:
        if self.total_s > 0 and self.output_tokens > 0:
            return round(self.output_tokens / self.total_s, 1)
        return 0.0


def _detect_backend(host: str) -> str:
    """Return 'ollama' or 'openai' based on host port or explicit marker."""
    if "lmstudio" in host or ":1234" in host:
        return "openai"
    # Ask Ollama's health endpoint; if it replies it's Ollama
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.ok:
            return "ollama"
    except Exception:
        pass
    # Fall back to OpenAI-compat (LM Studio / llama.cpp / vLLM / etc.)
    return "openai"


def _run_ollama(model: str, prompt: str, system: str, host: str) -> RunResult:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "options": {"temperature": 0.1, "seed": 42},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    try:
        with requests.post(url, json=payload, stream=True, timeout=90) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token and ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(token)
                if data.get("done"):
                    break
    except Exception as exc:
        return RunResult(0, model, prompt, "", 0, 0, 0, False, error=str(exc))

    total = time.perf_counter() - t0
    response = "".join(chunks)
    output_tokens = max(1, len(response.split()))
    return RunResult(0, model, prompt, response,
                     round(ttft or total, 3), round(total, 3),
                     output_tokens, False)


def _run_openai_compat(model: str, prompt: str, system: str, host: str) -> RunResult:
    """OpenAI-compatible streaming endpoint — works with LM Studio, llama.cpp, vLLM."""
    url = f"{host}/v1/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "temperature": 0.1,
        "seed": 42,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    output_tokens = 0
    try:
        with requests.post(url, json=payload, stream=True, timeout=90) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                # SSE format: b"data: {...}" or b"data: [DONE]"
                line = raw_line
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    data = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                delta = data.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content") or ""
                if token and ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(token)
                # some servers send usage in the last chunk
                usage = data.get("usage") or {}
                if usage.get("completion_tokens"):
                    output_tokens = usage["completion_tokens"]
    except Exception as exc:
        return RunResult(0, model, prompt, "", 0, 0, 0, False, error=str(exc))

    total = time.perf_counter() - t0
    response = "".join(chunks)
    if not output_tokens:
        output_tokens = max(1, len(response.split()))
    return RunResult(0, model, prompt, response,
                     round(ttft or total, 3), round(total, 3),
                     output_tokens, False)


def _run_local(model: str, prompt: str, system: str, host: str) -> RunResult:
    """Auto-detect Ollama vs OpenAI-compat and dispatch accordingly."""
    backend = _detect_backend(host)
    if backend == "ollama":
        return _run_ollama(model, prompt, system, host)
    return _run_openai_compat(model, prompt, system, host)


def _run_anthropic(model: str, prompt: str, system: str, api_key: Optional[str]) -> RunResult:
    if not HAS_ANTHROPIC_SDK:
        return RunResult(0, model, prompt, "", 0, 0, 0, False,
                         error="anthropic SDK not installed: pip install anthropic")
    client = _anthropic_sdk.Anthropic(api_key=api_key)
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    output_tokens = 0
    try:
        with client.messages.stream(
            model=model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if text and ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(text)
            msg = stream.get_final_message()
            output_tokens = msg.usage.output_tokens
    except Exception as exc:
        return RunResult(0, model, prompt, "", 0, 0, 0, False, error=str(exc))

    total = time.perf_counter() - t0
    return RunResult(0, model, prompt, "".join(chunks),
                     round(ttft or total, 3), round(total, 3),
                     output_tokens, False)


# ══════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    name: str           # display name
    model_id: str       # actual model string
    backend: str        # "ollama" | "lmstudio" | "openai_compat" | "anthropic"
    # pricing (per 1M tokens, USD)
    price_input: float = 0.0
    price_output: float = 0.0
    price_cache_read: float = 0.0
    infra_monthly: float = 0.0   # GPU cost etc.
    local_host: str = "http://localhost:11434"
    anthropic_api_key: Optional[str] = None

    def run(self, prompt: str, system: str) -> RunResult:
        if self.backend in ("ollama", "lmstudio", "openai_compat"):
            return _run_local(self.model_id, prompt, system, self.local_host)
        elif self.backend == "anthropic":
            return _run_anthropic(self.model_id, prompt, system, self.anthropic_api_key)
        raise ValueError(f"Unknown backend: {self.backend}")


DEFAULT_MODELS: list[ModelConfig] = [
    # ModelConfig(
    #     name="Claude Haiku 3.5",
    #     model_id="claude-haiku-4-5-20251001",
    #     backend="anthropic",
    #     price_input=0.80,
    #     price_output=4.00,
    #     price_cache_read=0.08,
    # ),
    ModelConfig(
        name="Qwen3.5 9B",
        model_id="qwen3.5:9b",
        backend="ollama",          # auto-switches to openai_compat if LM Studio detected
        infra_monthly=80.0,
    ),
    ModelConfig(
        name="Gemma 4 E4B",
        model_id="gemma4:e4b",
        backend="ollama",
        infra_monthly=60.0,
    ),
]


# ══════════════════════════════════════════════════════════════════
#  COST CALCULATOR
# ══════════════════════════════════════════════════════════════════

@dataclass
class CostParams:
    requests_per_day: int = 1200
    avg_input_tokens: int = 420
    avg_output_tokens: int = 180
    system_prompt_tokens: int = 240
    cache_hit_rate: float = 0.65
    days_per_month: int = 30

    @property
    def requests_per_month(self) -> int:
        return self.requests_per_day * self.days_per_month


def calculate_cost(model: ModelConfig, params: CostParams) -> dict:
    rpm = params.requests_per_month
    total_input_m  = rpm * params.avg_input_tokens  / 1_000_000
    total_output_m = rpm * params.avg_output_tokens / 1_000_000
    cache_tokens_m = rpm * params.cache_hit_rate * params.system_prompt_tokens / 1_000_000

    cost_no_cache = (
        total_input_m  * model.price_input +
        total_output_m * model.price_output
    )
    cost_with_cache = (
        total_input_m  * model.price_input * (1 - params.cache_hit_rate * 0.5) +
        total_output_m * model.price_output +
        cache_tokens_m * model.price_cache_read
    )
    return {
        "input_tokens_m":  round(total_input_m, 2),
        "output_tokens_m": round(total_output_m, 2),
        "cost_no_cache":   round(cost_no_cache, 2),
        "cost_with_cache": round(cost_with_cache, 2),
        "infra_monthly":   model.infra_monthly,
        "total_no_cache":  round(cost_no_cache + model.infra_monthly, 2),
        "total_with_cache":round(cost_with_cache + model.infra_monthly, 2),
    }


# ══════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════

@dataclass
class ModelReport:
    model: ModelConfig
    results: list[RunResult] = field(default_factory=list)
    cost: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.passed)
        return round(passed / len(self.results) * 100, 1)

    @property
    def median_ttft(self) -> float:
        vals = [r.ttft_s for r in self.results if r.ttft_s > 0]
        return round(statistics.median(vals), 3) if vals else 0.0

    @property
    def median_throughput(self) -> float:
        vals = [r.throughput for r in self.results if r.throughput > 0]
        return round(statistics.median(vals), 1) if vals else 0.0

    @property
    def failed_tasks(self) -> list[RunResult]:
        return [r for r in self.results if not r.passed]


def run_benchmark(
    models: list[ModelConfig],
    tasks: list[Task],
    cost_params: CostParams,
    verbose: bool = False,
) -> list[ModelReport]:
    reports = []

    for model in models:
        console.print(f"\n[bold]▶ {model.name}[/bold]" if HAS_RICH
                      else f"\n▶ {model.name}")
        report = ModelReport(model=model)

        if HAS_RICH:
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            )
            with progress_ctx as progress:
                ptask = progress.add_task("running tasks…", total=len(tasks))
                for task in tasks:
                    result = model.run(task.prompt, SYSTEM_PROMPT)
                    result.task_id = task.id
                    result.model = model.name
                    result.prompt = task.prompt
                    result.passed = task.check(result.response) if not result.error else False
                    report.results.append(result)
                    progress.advance(ptask)
        else:
            for i, task in enumerate(tasks, 1):
                print(f"  [{i}/{len(tasks)}] task #{task.id}…", end=" ", flush=True)
                result = model.run(task.prompt, SYSTEM_PROMPT)
                result.task_id = task.id
                result.model = model.name
                result.prompt = task.prompt
                result.passed = task.check(result.response) if not result.error else False
                report.results.append(result)
                status = "✓" if result.passed else "✗"
                print(f"{status} ({result.ttft_s:.2f}s TTFT)")

        report.cost = calculate_cost(model, cost_params)
        reports.append(report)

        # per-model summary
        console.print(f"  accuracy={report.accuracy}%  "
                      f"TTFT={report.median_ttft}s  "
                      f"throughput={report.median_throughput} tok/s")

        if verbose:
            for r in report.failed_tasks:
                t = next(t for t in tasks if t.id == r.task_id)
                console.print(f"  ✗ task #{r.task_id} [{t.category}]: {t.prompt[:60]}")
                console.print(f"    response: {r.response[:120]}")
                if r.error:
                    console.print(f"    ERROR: {r.error}")

    return reports


# ══════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════

def print_summary(reports: list[ModelReport], tasks: list[Task]) -> None:
    if HAS_RICH:
        _print_rich_summary(reports, tasks)
    else:
        _print_plain_summary(reports, tasks)


def _print_rich_summary(reports: list[ModelReport], tasks: list[Task]) -> None:
    console.rule("BENCHMARK RESULTS")

    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    tbl.add_column("Model",        style="bold", min_width=18)
    tbl.add_column("Accuracy",     justify="right")
    tbl.add_column("Passed",       justify="right")
    tbl.add_column("TTFT (med)",   justify="right")
    tbl.add_column("tok/s (med)",  justify="right")
    tbl.add_column("API $/mo",     justify="right")
    tbl.add_column("Infra $/mo",   justify="right")
    tbl.add_column("Total $/mo",   justify="right")

    best_acc = max(r.accuracy for r in reports)
    for rep in reports:
        acc_str = f"[green]{rep.accuracy}%[/green]" if rep.accuracy == best_acc else f"{rep.accuracy}%"
        passed  = sum(1 for r in rep.results if r.passed)
        c = rep.cost
        tbl.add_row(
            rep.model.name,
            acc_str,
            f"{passed}/{len(rep.results)}",
            f"{rep.median_ttft}s",
            str(rep.median_throughput),
            f"${c['cost_with_cache']}",
            f"${c['infra_monthly']}",
            f"[cyan]${c['total_with_cache']}[/cyan]",
        )
    console.print(tbl)

    # per-category breakdown
    categories = sorted({t.category for t in tasks})
    console.rule("ACCURACY BY CATEGORY")
    cat_tbl = Table(box=box.SIMPLE_HEAD)
    cat_tbl.add_column("Category", min_width=14)
    for rep in reports:
        cat_tbl.add_column(rep.model.name, justify="right")

    for cat in categories:
        cat_tasks = [t for t in tasks if t.category == cat]
        row = [cat]
        for rep in reports:
            cat_results = [r for r in rep.results if r.task_id in {t.id for t in cat_tasks}]
            passed = sum(1 for r in cat_results if r.passed)
            pct = round(passed / len(cat_results) * 100) if cat_results else 0
            color = "green" if pct == 100 else ("yellow" if pct >= 67 else "red")
            row.append(f"[{color}]{passed}/{len(cat_results)} ({pct}%)[/{color}]")
        cat_tbl.add_row(*row)
    console.print(cat_tbl)

    # failures
    console.rule("FAILED TASKS")
    for rep in reports:
        if rep.failed_tasks:
            console.print(f"\n[bold]{rep.model.name}[/bold]")
            for r in rep.failed_tasks:
                t = next(t for t in tasks if t.id == r.task_id)
                console.print(f"  [red]✗[/red] #{t.id} [{t.category}] {t.prompt[:70]}")
                if r.error:
                    console.print(f"    [red]ERROR: {r.error}[/red]")
                else:
                    console.print(f"    check: {t.check_desc}")
                    console.print(f"    response: {r.response[:120]}")


def _print_plain_summary(reports: list[ModelReport], tasks: list[Task]) -> None:
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    header = f"{'Model':<20} {'Acc':>6} {'Passed':>8} {'TTFT':>8} {'tok/s':>7} {'$/mo':>8}"
    print(header)
    print("-"*70)
    for rep in reports:
        passed = sum(1 for r in rep.results if r.passed)
        c = rep.cost
        print(f"{rep.model.name:<20} {rep.accuracy:>5}% "
              f"{passed:>3}/{len(rep.results):<3} "
              f"{rep.median_ttft:>7}s "
              f"{rep.median_throughput:>6} "
              f"${c['total_with_cache']:>7}")

    print("\n" + "="*70)
    print("FAILED TASKS")
    print("="*70)
    for rep in reports:
        if rep.failed_tasks:
            print(f"\n{rep.model.name}")
            for r in rep.failed_tasks:
                t = next(t for t in tasks if t.id == r.task_id)
                status = f"ERROR: {r.error}" if r.error else f"check: {t.check_desc}"
                print(f"  ✗ #{t.id} [{t.category}] {t.prompt[:60]}")
                print(f"    {status}")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Support Bot LLM Benchmark")
    p.add_argument("--models", nargs="+", metavar="MODEL_ID",
                   help="Restrict to these model_ids (e.g. qwen3:8b)")
    p.add_argument("--tasks", nargs="+", type=int, metavar="TASK_ID",
                   help="Run only specific task IDs (e.g. 1 5 20)")

    host_group = p.add_mutually_exclusive_group()
    host_group.add_argument("--local-host", default=None,
                   help="Local inference server URL (auto-detects Ollama vs LM Studio)")
    host_group.add_argument("--lmstudio-host", default=None,
                   metavar="HOST",
                   help="LM Studio URL, e.g. http://localhost:1234  (shortcut, sets OpenAI-compat mode)")
    host_group.add_argument("--ollama-host", default=None,
                   help="Ollama URL, e.g. http://localhost:11434")

    p.add_argument("--anthropic-key", default=None,
                   help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    p.add_argument("--requests-per-day", type=int, default=1200)
    p.add_argument("--avg-input",  type=int, default=420)
    p.add_argument("--avg-output", type=int, default=180)
    p.add_argument("--cache-hit",  type=float, default=0.65)
    p.add_argument("--output", metavar="FILE",
                   help="Save raw results to JSON file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print tasks and exit")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print failed responses")
    return p.parse_args()


def main():
    args = parse_args()

    # ── dry run ─────────────────────────────────────────────────
    if args.dry_run:
        print(f"{'#':<4} {'Category':<14} {'Type':<6} {'Prompt'}")
        print("─" * 80)
        for t in TASKS:
            print(f"{t.id:<4} {t.category:<14} {t.check_type:<6} {t.prompt[:60]}")
        print(f"\nTotal tasks: {len(TASKS)}")
        return

    # ── select models ────────────────────────────────────────────
    import os
    api_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")

    # ── resolve local host ───────────────────────────────────────
    if args.lmstudio_host:
        local_host = args.lmstudio_host
        force_openai = True
    elif args.ollama_host:
        local_host = args.ollama_host
        force_openai = False
    elif args.local_host:
        local_host = args.local_host
        force_openai = False
    else:
        local_host = "http://localhost:11434"
        force_openai = False

    models = list(DEFAULT_MODELS)
    if args.models:
        models = [m for m in models if m.model_id in args.models]
        if not models:
            print(f"No models matched: {args.models}", file=sys.stderr)
            sys.exit(1)

    for m in models:
        m.local_host = local_host
        if force_openai and m.backend != "anthropic":
            m.backend = "lmstudio"
        if m.backend == "anthropic":
            m.anthropic_api_key = api_key

    # ── select tasks ─────────────────────────────────────────────
    tasks = TASKS
    if args.tasks:
        tasks = [t for t in TASKS if t.id in args.tasks]

    # ── cost params ──────────────────────────────────────────────
    cost_params = CostParams(
        requests_per_day=args.requests_per_day,
        avg_input_tokens=args.avg_input,
        avg_output_tokens=args.avg_output,
        cache_hit_rate=args.cache_hit,
    )

    # ── run ──────────────────────────────────────────────────────
    console.print(f"\nRunning {len(tasks)} tasks on {len(models)} model(s)…\n")
    reports = run_benchmark(models, tasks, cost_params, verbose=args.verbose)

    # ── summary ──────────────────────────────────────────────────
    print_summary(reports, tasks)

    # ── save ─────────────────────────────────────────────────────
    if args.output:
        data = []
        for rep in reports:
            data.append({
                "model": rep.model.name,
                "accuracy": rep.accuracy,
                "median_ttft": rep.median_ttft,
                "median_throughput": rep.median_throughput,
                "cost": rep.cost,
                "results": [asdict(r) for r in rep.results],
            })
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()