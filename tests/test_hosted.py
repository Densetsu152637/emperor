"""Hosted clients: request shape per provider, usage and cost, budgets, backoff, key handling."""

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from emperor.data.hosted import (
    BudgetExceeded,
    HostedClient,
    Ledger,
    load_config,
    request_cost,
    usage_tokens,
)
from emperor.data.teacher import (
    TeacherContextError,
    TeacherError,
    TeacherOutputError,
    TeacherSetupError,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "hosted.toml")
KEY = "sk-test-not-a-real-key"
PEAK = datetime(2026, 11, 13, 2, 0, tzinfo=UTC)  # a Friday, 02:00 UTC
OFFPEAK = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)
WEEKEND = datetime(2026, 11, 14, 2, 0, tzinfo=UTC)


class Response:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code = status
        self._body = body
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._body


def ok(message=None, usage=None, model="served-model"):
    return Response(
        200,
        {
            "model": model,
            "choices": [{"message": message or {"content": "hi"}}],
            "usage": usage or {},
        },
    )


class Http:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, json, headers):
        self.posts.append({"url": url, "json": dict(json), "headers": headers})
        return self.responses.pop(0)

    def close(self):
        pass


def client(role, responses, *, ledger=None, clock=lambda: OFFPEAK, sleeps=None, **over):
    r, p = CONFIG.role(role)
    p = dataclasses.replace(p, **over)
    http = Http(responses)
    made = HostedClient(
        r,
        p,
        ledger or Ledger(None),
        key=KEY,
        http=http,
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
        clock=clock,
    )
    return made, http


def test_the_config_names_the_three_roles_and_cites_its_rates():
    assert CONFIG.roles["teacher"].model == "deepseek-v4-pro"
    assert CONFIG.roles["simulator"].model == "deepseek-flash"
    assert CONFIG.roles["judge"].model == "kimi-k3.1-pro"
    for provider in CONFIG.providers.values():
        assert provider.rates_source.startswith("https://")
        assert provider.base_url.startswith("https://")


def test_deepseek_teacher_thinks_and_never_gets_a_required_tool_choice():
    teacher, http = client("teacher", [ok()])
    teacher.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}],
                 tool_choice="required", temperature=0.7, seed=3, max_tokens=4096)  # fmt: skip
    body = http.posts[0]["json"]
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == CONFIG.roles["teacher"].reasoning_effort
    assert "tool_choice" not in body  # a 400 in thinking mode
    assert "temperature" not in body and "seed" not in body
    assert body["max_tokens"] == 16384  # the role's budget covers reasoning
    assert http.posts[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert http.posts[0]["headers"]["Authorization"] == f"Bearer {KEY}"


def test_simulator_does_not_think_and_keeps_its_temperature():
    simulator, http = client("simulator", [ok()])
    simulator.chat([{"role": "user", "content": "x"}], temperature=0.8, max_tokens=1200)
    body = http.posts[0]["json"]
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body
    assert body["temperature"] == 0.8 and body["max_tokens"] == 1200
    assert body["model"] == "deepseek-flash"


def test_kimi_judge_gets_no_sampling_parameters_and_no_thinking_field():
    judge, http = client("judge", [ok()])
    judge.chat([{"role": "user", "content": "x"}], temperature=0.6, max_tokens=8192)
    body = http.posts[0]["json"]
    assert "temperature" not in body and "thinking" not in body
    assert body["reasoning_effort"] == CONFIG.roles["judge"].reasoning_effort
    assert body["model"] == "kimi-k3.1-pro"
    assert http.posts[0]["url"] == "https://api.moonshot.ai/v1/chat/completions"


def test_reasoning_comes_back_in_reasoning_content():
    teacher, _ = client("teacher", [ok({"content": "a", "reasoning_content": "because"})])
    reply = teacher.chat([{"role": "user", "content": "x"}])
    assert reply["reasoning_content"] == "because"
    assert teacher.served_model == "served-model"


def test_usage_splits_cache_hits_for_both_providers():
    deepseek = usage_tokens(
        {"prompt_tokens": 100, "completion_tokens": 50, "prompt_cache_hit_tokens": 60,
         "prompt_cache_miss_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 30}}
    )  # fmt: skip
    assert deepseek == {"input_hit": 60, "input_write": 0, "input_miss": 40, "output": 50,
                        "reasoning": 30}  # fmt: skip
    kimi = usage_tokens(
        {"prompt_tokens": 373, "completion_tokens": 112,
         "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 256}}
    )  # fmt: skip
    assert (kimi["input_hit"], kimi["input_write"], kimi["input_miss"]) == (100, 256, 17)


def test_deepseek_costs_half_off_peak_and_kimi_costs_the_same_at_any_hour():
    deepseek = CONFIG.providers["deepseek"]
    tokens = {"input_hit": 1_000_000, "input_write": 0, "input_miss": 1_000_000,
              "output": 1_000_000, "reasoning": 0}  # fmt: skip
    peak = request_cost(deepseek, "deepseek-v4-pro", tokens, PEAK)
    assert peak == pytest.approx(0.044 + 1.32 + 3.96)
    assert request_cost(deepseek, "deepseek-v4-pro", tokens, OFFPEAK) == pytest.approx(peak / 2)
    assert request_cost(deepseek, "deepseek-v4-pro", tokens, WEEKEND) == pytest.approx(peak / 2)
    kimi = CONFIG.providers["kimi"]
    assert request_cost(kimi, "kimi-k3.1-pro", tokens, PEAK) == pytest.approx(0.2 + 2 + 10)


def test_every_request_is_recorded_and_the_budget_stops_the_run(tmp_path):
    ledger = Ledger(tmp_path / "spend.jsonl", {"deepseek": 0.01})
    usage = {"prompt_tokens": 1000, "completion_tokens": 3000, "prompt_cache_hit_tokens": 0,
             "prompt_cache_miss_tokens": 1000}  # fmt: skip
    teacher, http = client("teacher", [ok(usage=usage)] * 3, ledger=ledger)
    teacher.chat([{"role": "user", "content": "x"}])
    assert ledger.spent("deepseek") == pytest.approx((1000 * 1.32 + 3000 * 3.96) / 1e6 / 2)
    teacher.chat([{"role": "user", "content": "x"}])  # still under the cap before it
    with pytest.raises(BudgetExceeded):
        teacher.chat([{"role": "user", "content": "x"}])
    assert len(http.posts) == 2
    # The totals survive a restart.
    again = Ledger(tmp_path / "spend.jsonl", {"deepseek": 0.01})
    assert again.summary()["deepseek"]["requests"] == 2
    with pytest.raises(BudgetExceeded):
        again.check("deepseek")


def test_rate_limits_and_server_errors_are_retried_with_backoff():
    sleeps = []
    teacher, http = client(
        "teacher",
        [Response(429, headers={"retry-after": "7"}), Response(503), ok()],
        sleeps=sleeps,
    )
    assert teacher.chat([{"role": "user", "content": "x"}])["content"] == "hi"
    assert sleeps[0] == 7.0 and len(sleeps) == 2 and len(http.posts) == 3

    stuck, _ = client("teacher", [Response(500)] * 8)
    with pytest.raises(TeacherError) as error:
        stuck.chat([{"role": "user", "content": "x"}])
    assert not isinstance(error.value, TeacherOutputError)


def test_client_errors_are_classified_without_retrying():
    for status, text, kind in (
        (400, "This model's maximum context length is 1048576 tokens", TeacherContextError),
        (400, "invalid tool schema", TeacherError),  # the setup: stops the phase
        (404, "model not found", TeacherError),
        (400, "Content Exists Risk", TeacherOutputError),  # this sample only
        (401, "bad key", TeacherSetupError),
        (402, "Insufficient Balance", BudgetExceeded),
    ):
        made, http = client("teacher", [Response(status, text=text)])
        with pytest.raises(kind):
            made.chat([{"role": "user", "content": "x"}])
        assert len(http.posts) == 1


def test_offpeak_only_waits_for_the_cheap_hours():
    times = iter([PEAK, PEAK, OFFPEAK, OFFPEAK])
    sleeps = []
    teacher, http = client(
        "teacher", [ok()], clock=lambda: next(times), sleeps=sleeps, offpeak_only=True
    )
    teacher.chat([{"role": "user", "content": "x"}])
    assert sleeps == [60, 60] and len(http.posts) == 1


def test_the_key_never_appears_in_errors_logs_or_repr(tmp_path):
    ledger = Ledger(tmp_path / "spend.jsonl")
    made, _ = client("teacher", [Response(400, text="bad request"), ok()], ledger=ledger)
    with pytest.raises(TeacherError) as error:
        made.chat([{"role": "user", "content": "x"}])
    made.chat([{"role": "user", "content": "x"}])
    assert KEY not in str(error.value) and KEY not in repr(made)
    assert KEY not in (tmp_path / "spend.jsonl").read_text(encoding="utf-8")


def test_a_missing_key_is_a_setup_error(monkeypatch):
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda service, user: None)
    monkeypatch.delenv("Deepseek_API_Key", raising=False)
    role, provider = CONFIG.role("teacher")
    with pytest.raises(TeacherSetupError) as error:
        HostedClient(role, provider, Ledger(None), http=Http([]))
    assert "keyring" in str(error.value) and "deepseek" in str(error.value)


def test_plain_http_endpoints_are_refused():
    role, provider = CONFIG.role("teacher")
    insecure = dataclasses.replace(provider, base_url="http://api.deepseek.com")
    with pytest.raises(TeacherSetupError):
        HostedClient(role, insecure, Ledger(None), key=KEY, http=Http([]))


def test_ledger_rows_are_json(tmp_path):
    ledger = Ledger(tmp_path / "spend.jsonl")
    ledger.record("deepseek", "teacher", "deepseek-v4-pro", usage_tokens({}), 0.0)
    row = json.loads((tmp_path / "spend.jsonl").read_text(encoding="utf-8"))
    assert row["provider"] == "deepseek" and row["role"] == "teacher"


def test_the_key_comes_from_the_environment_first_then_keyring(monkeypatch):
    import keyring

    from emperor.data.hosted import read_key

    provider = CONFIG.providers["kimi"]
    assert provider.key_env == "Kimi_API_Key"
    monkeypatch.setattr(keyring, "get_password", lambda service, user: "from-keyring")
    monkeypatch.setenv("Kimi_API_Key", "from-env")
    assert read_key(provider) == "from-env"
    monkeypatch.delenv("Kimi_API_Key")
    assert read_key(provider) == "from-keyring"
    assert CONFIG.providers["deepseek"].key_env == "Deepseek_API_Key"


def test_kimi_platform_keys_use_the_documented_endpoint():
    assert CONFIG.providers["kimi"].base_url == "https://api.moonshot.ai/v1"


def test_keys_come_from_the_env_file_after_the_environment_and_before_keyring(
    monkeypatch, tmp_path
):
    import os

    import keyring

    from emperor.data.hosted import read_key

    env_file = tmp_path / ".env"
    env_file.write_text(
        '# keys\nexport Deepseek_API_Key="ds-from-file"\nKimi_API_Key = kimi-from-file\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("Deepseek_API_Key", raising=False)
    monkeypatch.delenv("Kimi_API_Key", raising=False)
    monkeypatch.setattr(keyring, "get_password", lambda service, user: "from-keyring")
    assert read_key(CONFIG.providers["deepseek"], env_file) == "ds-from-file"
    assert read_key(CONFIG.providers["kimi"], env_file) == "kimi-from-file"
    assert "Deepseek_API_Key" not in os.environ  # read, never exported to child processes
    monkeypatch.setenv("Deepseek_API_Key", "ds-from-env")
    assert read_key(CONFIG.providers["deepseek"], env_file) == "ds-from-env"
    env_file.write_text("Kimi_API_Key=\n", encoding="utf-8")  # empty: falls through
    assert read_key(CONFIG.providers["kimi"], env_file) == "from-keyring"


def test_a_missing_key_names_the_variable_and_the_file(monkeypatch, tmp_path):
    import keyring

    from emperor.data.hosted import read_key

    monkeypatch.delenv("Kimi_API_Key", raising=False)
    monkeypatch.setattr(keyring, "get_password", lambda service, user: None)
    with pytest.raises(TeacherSetupError) as error:
        read_key(CONFIG.providers["kimi"], tmp_path / "missing.env")
    assert "Kimi_API_Key" in str(error.value) and "missing.env" in str(error.value)


def test_the_config_points_at_the_user_env_file():
    assert CONFIG.env_file is not None and CONFIG.env_file.name == ".env"
    assert CONFIG.env_file.parent.name == ".emperor"


def test_the_peak_window_in_utc_including_weekends_and_other_time_zones():
    from datetime import timedelta, timezone

    deepseek = CONFIG.providers["deepseek"]
    assert deepseek.offpeak_only  # the default: teacher and simulator wait for off-peak
    friday = datetime(2026, 11, 13, tzinfo=UTC)  # a Friday
    peak_hours = [h for h in range(24) if deepseek.is_peak(friday.replace(hour=h))]
    assert peak_hours == [1, 2, 3, 6, 7, 8, 9]
    saturday, sunday = friday + timedelta(days=1), friday + timedelta(days=2)
    assert not any(deepseek.is_peak(day.replace(hour=h)) for day in (saturday, sunday)
                   for h in range(24))  # fmt: skip
    # 12:30 in Sydney (UTC+11) on Friday is 01:30 UTC: peak.
    sydney = timezone(timedelta(hours=11))
    assert deepseek.is_peak(datetime(2026, 11, 13, 12, 30, tzinfo=sydney))
    # 10:30 on Monday in Sydney is 23:30 UTC on Sunday: off-peak.
    assert not deepseek.is_peak(datetime(2026, 11, 16, 10, 30, tzinfo=sydney))
    # The window reopens on the hour.
    assert deepseek.next_offpeak(friday.replace(hour=2, minute=17)) == friday.replace(hour=4)
    assert deepseek.next_offpeak(friday.replace(hour=7)) == friday.replace(hour=10)
    noon = friday.replace(hour=12, minute=5)
    assert deepseek.next_offpeak(noon) == noon
    assert not CONFIG.providers["kimi"].offpeak_only  # the judge has no off-peak price


def test_a_peak_request_waits_and_reports_when_it_resumes():
    times = iter([PEAK, PEAK, PEAK, OFFPEAK, OFFPEAK])
    seen = []

    def sleep(seconds):
        seen.append(teacher.waiting_until)

    r, p = CONFIG.role("teacher")
    teacher = HostedClient(r, p, Ledger(None), key=KEY, http=Http([ok()]), sleep=sleep,
                           clock=lambda: next(times))  # fmt: skip
    teacher.chat([{"role": "user", "content": "x"}])
    assert seen and all(t == PEAK.replace(hour=4) for t in seen)
    assert teacher.waiting_until is None


def test_an_overloaded_judge_falls_back_for_a_cool_down_then_tries_the_primary(tmp_path):
    overloaded = Response(
        429, text='{"error":{"type":"engine_overloaded_error","message":"overloaded"}}'
    )
    ledger = Ledger(tmp_path / "spend.jsonl")
    judge, http = client(
        "judge",
        [overloaded, overloaded, overloaded, ok(model="kimi-k3.1-pro-highspeed"),
         ok(model="kimi-k3.1-pro-highspeed"), ok(model="kimi-k3.1-pro")],
        ledger=ledger,
    )  # fmt: skip
    now = [0.0]
    judge._monotonic = lambda: now[0]
    judge.chat([{"role": "user", "content": "x"}])
    models = [p["json"]["model"] for p in http.posts]
    assert models == ["kimi-k3.1-pro"] * 3 + ["kimi-k3.1-pro-highspeed"]
    assert judge.last_model == "kimi-k3.1-pro-highspeed"
    judge.chat([{"role": "user", "content": "x"}])  # still cooling down
    assert http.posts[-1]["json"]["model"] == "kimi-k3.1-pro-highspeed"
    now[0] = 901.0
    judge.chat([{"role": "user", "content": "x"}])  # the primary again
    assert http.posts[-1]["json"]["model"] == "kimi-k3.1-pro"
    rows = [json.loads(line) for line in (tmp_path / "spend.jsonl").read_text().splitlines()]
    assert [r["model"] for r in rows] == [
        "kimi-k3.1-pro-highspeed", "kimi-k3.1-pro-highspeed", "kimi-k3.1-pro"
    ]  # fmt: skip
