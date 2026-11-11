import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from emperor.eval.cli import main
from emperor.eval.models import ModelError, OpenAICompatModel, ScriptedModel
from emperor.eval.runner import MAX_TURNS, ReplayEnvironment, run_task
from emperor.eval.tasks import Task, load_suite, task_from_dict

TIER = 65_536


def heldout(task_id: str) -> Task:
    return next(t for t in load_suite("heldout") if t.id == task_id)


def test_reference_replay_records_calls_and_final():
    task = heldout("refactor-rename")
    transcript = run_task(task, ScriptedModel(task.reference), TIER)
    assert transcript.error is None
    assert [c.name for c in transcript.calls] == ["spawn_agent", "await_tasks"]
    reports = json.loads(transcript.calls[1].result)["reports"]
    assert "48 passed" in reports[0]["report"]
    assert transcript.final.startswith("Done")


def test_replies_match_briefs_by_keyword():
    task = heldout("research-http-clients")
    environment = ReplayEnvironment(task)
    first = json.loads(environment.handle("spawn_agent", {"brief": "Look into requests"}))
    second = json.loads(environment.handle("spawn_agent", {"brief": "Now aiohttp please"}))
    reports = json.loads(environment.handle("await_tasks", {}))["reports"]
    assert [r["task_id"] for r in reports] == [first["task_id"], second["task_id"]]
    assert reports[0]["report"].startswith("requests:")
    assert reports[1]["report"].startswith("aiohttp:")
    unknown = json.loads(environment.handle("await_tasks", {"task_ids": ["t99"]}))
    assert unknown["reports"][0]["error"] == "unknown task id"


def test_delegate_chunks_answers_every_chunk():
    task = heldout("long-input-log-chunks")
    environment = ReplayEnvironment(task)
    ids = json.loads(
        environment.handle("delegate_chunks", {"artefact_id": "a", "chunk_tokens": 1, "brief": "b"})
    )["task_ids"]
    assert len(ids) == 4


def test_invalid_calls_are_answered_with_an_error_and_recorded():
    task = task_from_dict({"id": "x", "prompt": "p", "tools": ["spawn_agent"]}, "s")
    model = ScriptedModel(
        [
            {"tool_calls": [{"name": "spawn_agent", "arguments": "{not json"}]},
            {"tool_calls": [{"name": "web_fetch", "arguments": {"url": "u"}}]},
            {"content": "gave up"},
        ]
    )
    transcript = run_task(task, model, TIER)
    assert [c.valid for c in transcript.calls] == [False, False]
    assert all(c.result.startswith("Error: invalid tool call") for c in transcript.calls)
    assert transcript.final == "gave up"


def test_overflow_stops_the_task():
    task = load_suite("longctx")[0]
    transcript = run_task(task, ScriptedModel(task.reference), 1024)
    assert transcript.overflowed
    assert "overflow" in transcript.error
    # The tier-sized haystack itself fits its own tier.
    assert not run_task(task, ScriptedModel(task.reference), 32_768).overflowed


def test_turn_limit():
    task = task_from_dict({"id": "x", "prompt": "p"}, "s")
    loop = [{"tool_calls": [{"name": "await_tasks", "arguments": {}}]}] * (MAX_TURNS + 1)
    transcript = run_task(task, ScriptedModel(loop), TIER)
    assert transcript.error == f"no final answer after {MAX_TURNS} turns"


class FakeServer:
    """A tiny OpenAI-compatible endpoint that replays canned responses and records requests."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                body = json.dumps(outer.responses.pop(0)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def completion(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7},
    }


@pytest.fixture
def fake_server():
    servers = []

    def start(responses):
        servers.append(FakeServer(responses))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def test_openai_compatible_backend(fake_server, monkeypatch):
    monkeypatch.setenv("EMPEROR_EVAL_API_KEY", "dummy-key-for-test")
    server = fake_server(
        [
            completion(tool_calls=[("spawn_agent", {"brief": "## Goal\nx\n## Acceptance\ny"})]),
            completion(tool_calls=[("await_tasks", {})]),
            completion(content="All done."),
        ]
    )
    task = heldout("refactor-rename")
    transcript = run_task(task, OpenAICompatModel(server.url, "emperor"), TIER)
    assert transcript.final == "All done."
    assert transcript.prompt_tokens == 300 and transcript.completion_tokens == 21
    first = server.requests[0]
    assert first["path"] == "/v1/chat/completions"
    assert first["body"]["model"] == "emperor"
    assert first["body"]["messages"][0]["content"].rstrip().endswith("context budget: 64K tokens")
    assert {t["function"]["name"] for t in first["body"]["tools"]} >= {"spawn_agent", "ask_user"}
    assert first["headers"]["Authorization"] == "Bearer dummy-key-for-test"
    # The tool result goes back with the call's id.
    assert server.requests[1]["body"]["messages"][-1]["tool_call_id"] == "c0"


def test_cli_replay_through_a_server(fake_server, capsys, tmp_path):
    server = fake_server([completion(content="418 means I'm a teapot.")])
    tasks = tmp_path / "one.json"
    task = {"id": "t", "prompt": "418?", "expect": {"routing": {"mode": "answer"}}}
    tasks.write_text(json.dumps(task), encoding="utf-8")
    code = main(["--model", "llamacpp:emperor", "--base-url", server.url, "--tasks", str(tasks)])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["dry_run"] is False and report["backend"] == "replay"
    assert report["metrics"]["routing"]["score"] == 1.0


def test_unreachable_server_is_a_model_error():
    with pytest.raises(ModelError, match="cannot reach"):
        OpenAICompatModel("http://127.0.0.1:9/v1", "m", timeout=2).complete([], [])


def test_thinking_is_split_from_the_answer_before_parsing(fake_server):
    from emperor.eval.models import split_thinking

    assert split_thinking("<think>\nplan it\n</think>\n\nDone.") == ("plan it", "Done.")
    # The generation prompt opened the block, so only the closing tag is in the reply.
    assert split_thinking("plan it\n</think>\n\nDone.") == ("plan it", "Done.")
    assert split_thinking("<think>\nran out of tokens") == ("ran out of tokens", "")
    assert split_thinking("No thinking here.") == (None, "No thinking here.")

    server = fake_server(
        [
            completion(content="Spawn one agent.\n</think>\n\n"),
            completion(content="All done."),
        ]
    )
    # Reasoning the server separated, and a block it left in the content.
    server.responses[0]["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "c0",
            "type": "function",
            "function": {"name": "update_plan", "arguments": json.dumps({"steps": []})},
        }
    ]
    server.responses[1]["choices"][0]["message"]["reasoning_content"] = "Wrap up."
    model = OpenAICompatModel(server.url, "emperor")
    assert model.max_tokens >= 16384
    transcript = run_task(heldout("refactor-rename"), model, TIER)
    assert transcript.final == "All done."
    assert transcript.replies[0].reasoning == "Spawn one agent."
    assert transcript.replies[0].content == ""
    assert transcript.replies[1].reasoning == "Wrap up."
    assert "</think>" not in transcript.output_text()
    # The reasoning goes back with its turn, as the model was trained to see it.
    history = server.requests[1]["body"]["messages"]
    assert history[2]["reasoning_content"] == "Spawn one agent."


def test_scripted_thinking_does_not_reach_the_final_answer():
    task = heldout("answer-http-418")
    model = ScriptedModel([{"content": "<think>\n418 is a joke code.\n</think>\n\nI'm a teapot."}])
    transcript = run_task(task, model, TIER)
    assert transcript.final == "I'm a teapot."
    assert transcript.replies[0].reasoning == "418 is a joke code."


def test_a_reply_cut_off_mid_thought_is_reasoning_without_an_answer(fake_server):
    from emperor.eval.models import split_thinking

    assert split_thinking("Still weighing the options", truncated=True) == (
        "Still weighing the options",
        "",
    )
    reply = completion(content="Still weighing the options")
    reply["choices"][0]["finish_reason"] = "length"
    served = completion(content="Partial answer")
    served["choices"][0]["finish_reason"] = "length"
    served["choices"][0]["message"]["reasoning_content"] = "Done thinking."
    server = fake_server([reply, served])
    model = OpenAICompatModel(server.url, "emperor")
    first = model.complete([], [])
    assert (first.reasoning, first.content) == ("Still weighing the options", "")
    # The server separated the reasoning, so the cut-off text is a (partial) answer.
    second = model.complete([], [])
    assert (second.reasoning, second.content) == ("Done thinking.", "Partial answer")


def test_reasoning_counts_towards_the_context_budget():
    task = heldout("refactor-rename")
    # All of it is reasoning: the visible content alone would never overflow the budget.
    turn = {
        "content": "<think>" + "word " * 40_000 + "</think>",
        "tool_calls": [{"name": "update_plan", "arguments": {"steps": []}}],
    }
    transcript = run_task(task, ScriptedModel([turn, turn]), 32_768)
    assert transcript.overflowed and len(transcript.replies) == 1
