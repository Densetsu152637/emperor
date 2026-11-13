"""Hosted models over OpenAI-compatible APIs: the teacher (DeepSeek V4 Pro), the tool simulator
(DeepSeek Flash) and the judge (Kimi K3.1 Pro), with spend tracking and budget caps.

API keys are read from the operating system's credential store through `keyring` (service and
username from the config) when a client is made. They are kept only in the request headers:
never printed, logged, written or put into an error message.

Provider behaviour that shapes the requests (checked against the official docs on 2026-11-13;
`configs/hosted.toml` cites the pages):

- DeepSeek (`https://api.deepseek.com`): thinking is switched with `{"thinking": {"type":
  "enabled" | "disabled"}}` and `reasoning_effort` (low / high / max; medium maps to high). In
  thinking mode `temperature` has no effect and `tool_choice: "required"` is a 400 error, so it
  is not sent. The chain of thought comes back in `reasoning_content` and, when the request
  carries tools, must be passed back on earlier assistant turns (generation already does).
  Usage reports `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`. Prices halve
  off-peak; peak is 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday.
- Kimi Open Platform (`https://api.moonshot.ai/v1`, keys from platform.kimi.ai), K3 series:
  always thinks (no `thinking` field;
  `reasoning_effort` low / high / max); `temperature`, `top_p`, `n` and the penalties are fixed
  and passing them is an error. Usage reports `prompt_tokens_details.cached_tokens` and
  `cache_write_tokens`.
- Neither documents a `seed` parameter, so none is sent.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from emperor.data.teacher import (
    _CONTEXT_ERROR,
    TeacherContextError,
    TeacherError,
    TeacherOutputError,
    TeacherSetupError,
)

DEFAULT_CONFIG = Path("configs/hosted.toml")
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class BudgetExceeded(TeacherError):
    """A provider's spend reached its cap: the run stops (resumable) until the cap is raised."""


# -- configuration ----------------------------------------------------------------------------


@dataclass
class Rates:
    """USD per million tokens at the provider's standard (peak) price."""

    input_miss: float
    input_hit: float
    output: float
    cache_write: float | None = None  # Moonshot bills cache writes; None means input_miss


@dataclass
class Provider:
    name: str
    base_url: str
    key_user: str
    key_service: str = "emperor"
    key_env: str | None = None  # an environment variable checked before the keyring entry
    budget_usd: float = 25.0
    max_concurrency: int = 8
    rates: dict[str, Rates] = field(default_factory=dict)
    rates_source: str = ""
    # Off-peak pricing: `offpeak_factor` of the standard price outside the peak windows.
    peak_hours_utc: list[tuple[int, int]] = field(default_factory=list)
    peak_weekdays_only: bool = True
    offpeak_factor: float = 1.0
    offpeak_only: bool = False  # wait for off-peak hours before every request
    # Request shaping (see the module docstring).
    thinking_param: bool = False  # the provider takes {"thinking": {"type": ...}}
    fixed_sampling: bool = False  # temperature and top_p must not be sent
    required_tool_choice_when_thinking: bool = True
    timeout_seconds: float = 900.0
    max_retries: int = 8

    def is_peak(self, when: datetime) -> bool:
        if not self.peak_hours_utc:
            return False
        when = when.astimezone(UTC)
        if self.peak_weekdays_only and when.weekday() >= 5:
            return False
        return any(start <= when.hour < end for start, end in self.peak_hours_utc)

    def price_factor(self, when: datetime) -> float:
        return 1.0 if self.is_peak(when) else self.offpeak_factor


@dataclass
class Role:
    name: str
    provider: str
    model: str
    thinking: bool | None = None  # None: do not send the field
    reasoning_effort: str | None = None
    max_tokens: int | None = None  # overrides the caller's budget (thinking needs room)


@dataclass
class HostedConfig:
    providers: dict[str, Provider]
    roles: dict[str, Role]
    source: Path | None = None
    env_file: Path | None = None  # KEY=VALUE lines, checked after the process environment

    def role(self, name: str) -> tuple[Role, Provider]:
        role = self.roles[name]
        return role, self.providers[role.provider]


def load_config(path: Path = DEFAULT_CONFIG) -> HostedConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    providers = {}
    for name, raw in data.get("providers", {}).items():
        raw = dict(raw)
        rates = {model: Rates(**r) for model, r in raw.pop("rates", {}).items()}
        peaks = [tuple(p) for p in raw.pop("peak_hours_utc", [])]
        providers[name] = Provider(name=name, rates=rates, peak_hours_utc=peaks, **raw)
    roles = {name: Role(name=name, **raw) for name, raw in data.get("roles", {}).items()}
    env_file = data.get("keys", {}).get("env_file")
    for role in roles.values():
        if role.provider not in providers:
            raise ValueError(f"role {role.name}: unknown provider {role.provider}")
        if role.model not in providers[role.provider].rates:
            raise ValueError(f"role {role.name}: no rates for {role.model} in {role.provider}")
    return HostedConfig(
        providers=providers,
        roles=roles,
        source=Path(path),
        env_file=Path(env_file).expanduser() if env_file else None,
    )


# -- spend ------------------------------------------------------------------------------------


def usage_tokens(usage: dict) -> dict:
    """Input split into cache hits, cache writes and misses, plus output and reasoning tokens,
    from either provider's usage block."""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    if "prompt_cache_hit_tokens" in usage or "prompt_cache_miss_tokens" in usage:  # DeepSeek
        hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        write = 0
        miss = int(usage.get("prompt_cache_miss_tokens") or max(prompt - hit, 0))
    else:  # Moonshot and plain OpenAI-style
        hit = int(details.get("cached_tokens") or 0)
        write = int(details.get("cache_write_tokens") or 0)
        miss = max(prompt - hit - write, 0)
    output = int(usage.get("completion_tokens") or 0)
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    return {
        "input_hit": hit,
        "input_write": write,
        "input_miss": miss,
        "output": output,
        "reasoning": reasoning,
    }


def request_cost(provider: Provider, model: str, tokens: dict, when: datetime) -> float:
    rates = provider.rates[model]
    write_rate = rates.cache_write if rates.cache_write is not None else rates.input_miss
    usd = (
        tokens["input_hit"] * rates.input_hit
        + tokens["input_write"] * write_rate
        + tokens["input_miss"] * rates.input_miss
        + tokens["output"] * rates.output
    ) / 1_000_000
    return usd * provider.price_factor(when)


class Ledger:
    """Every hosted request's usage and cost, appended to `path` (JSONL), with running totals
    per provider. Totals are rebuilt from the file on start, so budgets hold across resumes."""

    def __init__(self, path: Path | None, budgets: dict[str, float] | None = None):
        self.path = Path(path) if path else None
        self.budgets = dict(budgets or {})
        self._lock = threading.Lock()
        self.totals: dict[str, dict] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self._add(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def _add(self, row: dict) -> None:
        total = self.totals.setdefault(
            row["provider"],
            {"usd": 0.0, "requests": 0, "input_hit": 0, "input_write": 0, "input_miss": 0,
             "output": 0, "reasoning": 0},
        )  # fmt: skip
        total["usd"] += row["usd"]
        total["requests"] += 1
        for key in ("input_hit", "input_write", "input_miss", "output", "reasoning"):
            total[key] += row["tokens"].get(key, 0)

    def spent(self, provider: str) -> float:
        with self._lock:
            return self.totals.get(provider, {}).get("usd", 0.0)

    def check(self, provider: str) -> None:
        budget = self.budgets.get(provider)
        if budget is not None and self.spent(provider) >= budget:
            raise BudgetExceeded(
                f"{provider}: spend ${self.spent(provider):.2f} reached the ${budget:.2f} cap "
                "(raise budget_usd in the hosted config to continue)"
            )

    def record(self, provider: str, role: str, model: str, tokens: dict, usd: float) -> None:
        row = {
            "t": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": provider,
            "role": role,
            "model": model,
            "tokens": tokens,
            "usd": round(usd, 6),
        }
        with self._lock:
            self._add(row)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(row) + "\n")

    def summary(self) -> dict:
        with self._lock:
            return {
                name: {**total, "usd": round(total["usd"], 4), "budget_usd": self.budgets.get(name)}
                for name, total in self.totals.items()
            }


# -- the client -------------------------------------------------------------------------------


def read_env_file(path: Path | None, name: str) -> str:
    """One variable from a `.env` file (KEY=VALUE lines; `#` comments, an optional `export`
    and surrounding quotes allowed). Read only for this process: nothing is exported to the
    environment, so child processes never inherit the key."""
    if not path or not name or not Path(path).is_file():
        return ""
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        if key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value.strip()
    return ""


def read_key(provider: Provider, env_file: Path | None = None) -> str:
    """The provider's API key, from the first source that has it: the process environment
    variable `key_env`, the same name in `env_file`, then the OS credential store (keyring).
    Never printed, logged or written."""
    if provider.key_env:
        key = os.environ.get(provider.key_env, "").strip()
        if key:
            return key
        key = read_env_file(env_file, provider.key_env)
        if key:
            return key
    try:
        import keyring  # the `hosted` dependency group
    except ImportError:
        keyring = None
    key = keyring.get_password(provider.key_service, provider.key_user) if keyring else None
    if not key:
        raise TeacherSetupError(
            f"no API key for {provider.name}: set {provider.key_env or '(no variable configured)'}"
            f" in the environment or in {env_file or '(no env_file configured)'}, or store it with "
            f"keyring under service '{provider.key_service}', username '{provider.key_user}'"
        )
    return key


def _retry_after(response) -> float | None:
    try:
        value = response.headers.get("retry-after")
        return float(value) if value else None
    except (AttributeError, TypeError, ValueError):
        return None


class HostedClient:
    """`Chat` for one role on a hosted OpenAI-compatible API: shapes the request for the
    provider, retries rate limits and server errors with backoff, records usage and cost, and
    enforces the provider's budget cap before each request."""

    def __init__(
        self,
        role: Role,
        provider: Provider,
        ledger: Ledger,
        *,
        key: str | None = None,
        http=None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        should_stop: Callable[[], bool] = lambda: False,
    ):
        if urlparse(provider.base_url).scheme != "https":
            raise TeacherSetupError(f"{provider.name}: the API must be reached over https")
        self.role = role
        self.provider = provider
        self.ledger = ledger
        self.model = role.model
        self.base_url = provider.base_url.rstrip("/")
        self.template_kwargs = {"enable_thinking": bool(role.thinking or role.reasoning_effort)}
        self.served_model: str | None = None
        self._sleep = sleep
        self._clock = clock
        self._should_stop = should_stop
        self._headers = {"Authorization": f"Bearer {key or read_key(provider)}"}
        if http is None:
            import httpx

            http = httpx.Client(timeout=provider.timeout_seconds)
        self._http = http

    def __repr__(self) -> str:  # never show the headers
        return f"HostedClient({self.provider.name}, {self.model})"

    def close(self) -> None:
        self._http.close()

    def body(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
        tool_choice: str | None,
    ) -> dict:
        role, provider = self.role, self.provider
        body: dict = {
            "model": role.model,
            "messages": messages,
            "max_tokens": role.max_tokens or max_tokens,
        }
        thinking = bool(role.thinking) if role.thinking is not None else None
        if not provider.fixed_sampling and not thinking:
            body["temperature"] = temperature
        if provider.thinking_param and thinking is not None:
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if role.reasoning_effort and thinking is not False:
            body["reasoning_effort"] = role.reasoning_effort
        if tools:
            body["tools"] = tools
            if tool_choice and not (
                tool_choice == "required"
                and thinking is not False
                and not provider.required_tool_choice_when_thinking
            ):
                body["tool_choice"] = tool_choice
        return body

    def _wait_for_offpeak(self) -> None:
        while self.provider.offpeak_only and self.provider.is_peak(self._clock()):
            if self._should_stop():
                raise TeacherError(f"{self.provider.name}: stopped while waiting for off-peak")
            self._sleep(60)

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        seed: int | None = None,  # not sent: neither provider documents it
        max_tokens: int = 4096,
        tool_choice: str | None = None,
    ) -> dict:
        body = self.body(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
        )
        url = f"{self.base_url}/chat/completions"
        last = "no attempt"
        for attempt in range(self.provider.max_retries):
            self.ledger.check(self.provider.name)
            self._wait_for_offpeak()
            sent = self._clock()
            try:
                response = self._http.post(url, json=body, headers=self._headers)
            except Exception as error:  # timeouts, dropped connections: retried
                last = type(error).__name__
                self._sleep(self._backoff(attempt, None))
                continue
            status = response.status_code
            if status == 200:
                try:
                    data = response.json()
                    message = data["choices"][0]["message"]
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    raise TeacherOutputError(f"{self.provider.name}: bad response body") from error
                tokens = usage_tokens(data.get("usage") or {})
                usd = request_cost(self.provider, self.role.model, tokens, sent)
                self.ledger.record(self.provider.name, self.role.name, self.role.model, tokens, usd)
                self.served_model = data.get("model") or self.served_model
                return message
            text = _safe_text(response)
            if status in RETRY_STATUS:
                last = f"HTTP {status}"
                self._sleep(self._backoff(attempt, _retry_after(response)))
                continue
            if status in (401, 403):
                raise TeacherSetupError(f"{self.provider.name}: HTTP {status} (check the API key)")
            if status == 402:
                raise BudgetExceeded(
                    f"{self.provider.name}: HTTP 402, the account balance is empty"
                )
            if _CONTEXT_ERROR.search(text):
                raise TeacherContextError(f"{self.provider.name}: {text[:300]}")
            raise TeacherOutputError(f"{self.provider.name}: HTTP {status}: {text[:300]}")
        raise TeacherError(
            f"{self.provider.name}: no answer after {self.provider.max_retries} tries ({last})"
        )

    @staticmethod
    def _backoff(attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, 300.0)
        return min(2.0**attempt, 60.0) * (0.5 + random.random())


def _safe_text(response) -> str:
    try:
        return response.text or ""
    except Exception:
        return ""


def make_clients(
    config: HostedConfig,
    ledger: Ledger,
    roles: tuple[str, ...] = ("teacher", "simulator", "judge"),
    *,
    should_stop: Callable[[], bool] = lambda: False,
) -> dict[str, HostedClient]:
    """One client per role; keys are read once per provider."""
    keys: dict[str, str] = {}
    clients = {}
    for name in roles:
        role, provider = config.role(name)
        if provider.name not in keys:
            keys[provider.name] = read_key(provider, config.env_file)
        clients[name] = HostedClient(
            role, provider, ledger, key=keys[provider.name], should_stop=should_stop
        )
    return clients
