"""Trace generation: the teacher plays Emperor; a separate "environment" call plays the tools.

The teacher sees `emperor-root.md`, the brief/plan/questions shapes and the style requirements.
The stored sample keeps only the plain system prompt (with the context-budget line), so the
style is what the student learns rather than what it is told.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from emperor.data.prompts import PromptRecord
from emperor.data.schema import LICENCE, validate_sample
from emperor.data.shards import ShardWriter, done_ids, read_jsonl
from emperor.data.style import STYLE
from emperor.data.teacher import TEACHER_MODEL, TeacherError, TeacherOutputError
from emperor.eval.contract import check_call, openai_tools, prompt, system_prompt


class Chat(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = ...,
        temperature: float = ...,
        seed: int | None = ...,
        max_tokens: int = ...,
    ) -> dict: ...


@dataclass
class GenerationConfig:
    max_turns: int = 14
    temperature: float = 0.7
    max_tokens: int = 4096
    teacher_model: str = TEACHER_MODEL


def teacher_system(context_tokens: int) -> str:
    """What the teacher is told: the real prompt, the artefact shapes and the style."""
    shapes = "\n\n".join(prompt(name).strip() for name in ("plan", "questions", "brief"))
    return f"{system_prompt(context_tokens).rstrip()}\n\n{shapes}\n\n{STYLE}"


_ENV_SYSTEM = """\
You simulate the tool results of an agent platform, for training data. You are not the
assistant and never address the user. You receive the user's task and one tool call the
assistant made, and you reply with the tool's result only: plain text, no commentary, no JSON
wrapper unless the tool would return one.

Rules:
- web_search: return 4 to 6 results, each as a title, a plausible https URL and a one-line
  snippet, one per line. Content must be realistic and consistent with what is known.
- web_fetch: return the readable text of the page, 150 to 400 words, with concrete facts.
- A subagent report (await_tasks): write the report the subagent would send for its brief,
  following the brief's "Output format", 80 to 250 words, with concrete evidence for each
  acceptance check. Occasionally (about one time in six) report a genuine problem or a failed
  check, so the assistant has to re-plan.
- ask_user: write the user's answers, one line per question, as a normal person would; accept
  the recommended answer about half the time.
Never include credentials, API keys, tokens or private endpoints in any result.
"""


class Environment:
    """Simulates tool results; ids and bookkeeping are deterministic, content comes from the LLM."""

    def __init__(self, client: Chat, task_text: str, seed: int, max_tokens: int = 1200):
        self.client = client
        self.task_text = task_text
        self.seed = seed
        self.max_tokens = max_tokens
        self.next_task = 1
        self.pending: dict[str, str] = {}  # task id -> brief
        self.calls = 0

    def _ask(self, request: str) -> str:
        self.calls += 1
        reply = self.client.chat(
            [
                {"role": "system", "content": _ENV_SYSTEM},
                {"role": "user", "content": request},
            ],
            temperature=0.8,
            seed=self.seed + self.calls,
            max_tokens=self.max_tokens,
        )
        return (reply.get("content") or "").strip() or "(no output)"

    def run(self, name: str, arguments: dict) -> str:
        if name == "update_plan":
            return json.dumps({"ok": True, "steps": len(arguments.get("steps", []))})
        if name in ("spawn_agent", "delegate_chunks"):
            return self._spawn(name, arguments)
        if name == "await_tasks":
            return self._await(arguments)
        if name == "ask_user":
            questions = [q["question"] for q in arguments["questions"]]
            answers = self._ask(
                f"User task: {self.task_text}\n\nTool: ask_user\nQuestions:\n"
                + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
                + "\n\nReturn the answers, one per line."
            )
            return json.dumps({"answers": answers})
        if name in ("web_search", "web_fetch"):
            return self._ask(
                f"User task: {self.task_text}\n\nTool: {name}\nArguments: "
                f"{json.dumps(arguments, ensure_ascii=False)}"
            )
        return json.dumps({"error": f"unknown tool {name}"})

    def _spawn(self, name: str, arguments: dict) -> str:
        if name == "delegate_chunks":
            tokens = max(int(arguments.get("chunk_tokens", 32000)), 1)
            count = max(2, 300_000 // tokens)
            ids = []
            for _ in range(min(count, 12)):
                task_id = f"t{self.next_task}"
                self.next_task += 1
                self.pending[task_id] = arguments["brief"]
                ids.append(task_id)
            if arguments.get("reduce"):
                task_id = f"t{self.next_task}"
                self.next_task += 1
                self.pending[task_id] = "Merge the chunk reports: " + arguments["reduce"]
                ids.append(task_id)
            return json.dumps({"task_ids": ids})
        task_id = f"t{self.next_task}"
        self.next_task += 1
        self.pending[task_id] = arguments["brief"]
        return json.dumps({"task_id": task_id, "status": "running"})

    def _await(self, arguments: dict) -> str:
        wanted = arguments.get("task_ids") or list(self.pending)
        reports = []
        for task_id in wanted:
            brief = self.pending.pop(task_id, None)
            if brief is None:
                reports.append({"task_id": task_id, "status": "unknown_task"})
                continue
            text = self._ask(
                f"User task: {self.task_text}\n\nTool: await_tasks, subagent {task_id}\n"
                f"The subagent's brief was:\n{brief}\n\nWrite its report."
            )
            reports.append({"task_id": task_id, "status": "done", "report": text})
        if not reports:
            return json.dumps({"reports": [], "note": "no running tasks"})
        return json.dumps({"reports": reports}, ensure_ascii=False)


def _normalise_calls(raw_calls: list[dict], counter: list[int]) -> list[dict]:
    calls = []
    for raw in raw_calls:
        function = raw.get("function") or {}
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        counter[0] += 1
        calls.append(
            {
                "id": f"call_{counter[0]}",
                "type": "function",
                "function": {"name": function.get("name", ""), "arguments": arguments},
            }
        )
    return calls


def generate_trace(
    record: PromptRecord,
    teacher: Chat,
    environment_client: Chat,
    config: GenerationConfig | None = None,
) -> tuple[dict | None, str | None]:
    """Play one prompt to the end; returns (sample, None) or (None, rejection reason)."""
    config = config or GenerationConfig()
    context_tokens = record.context_budget_k * 1024
    tools = openai_tools(record.tools)
    env = Environment(environment_client, record.text, record.seed)
    teacher_messages = [
        {"role": "system", "content": teacher_system(context_tokens)},
        {"role": "user", "content": record.text},
    ]
    stored = [
        {"role": "system", "content": system_prompt(context_tokens)},
        {"role": "user", "content": record.text},
    ]
    counter = [0]
    for turn in range(config.max_turns):
        reply = teacher.chat(
            teacher_messages,
            tools=tools,
            temperature=config.temperature,
            seed=record.seed + turn,
            max_tokens=config.max_tokens,
        )
        content = (reply.get("content") or "").strip()
        calls = _normalise_calls(reply.get("tool_calls") or [], counter)
        if not calls and not content:
            return None, "empty_reply"
        message: dict = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        teacher_messages.append(message)
        stored.append(message)
        if not calls:
            break
        for call in calls:
            name = call["function"]["name"]
            arguments, errors = check_call(name, call["function"]["arguments"], record.tools)
            if errors or arguments is None:
                return None, "tool_call_invalid"
            result = env.run(name, arguments)
            tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
            teacher_messages.append(tool_message)
            stored.append(tool_message)
    else:
        return None, "turn_limit"

    sample = {
        "id": record.id,
        "task_class": record.task_class,
        "specificity": record.specificity,
        "context_budget_k": record.context_budget_k,
        "tools": record.tools,
        "messages": stored,
        "meta": {
            "teacher": config.teacher_model,
            "licence": LICENCE,
            "seed": record.seed,
            "host": "local llama.cpp",
        },
    }
    check = validate_sample(sample)
    if not check.ok:
        return None, check.reasons[0]
    return sample, None


def run_generation(
    prompts_path: Path,
    out_dir: Path,
    teacher: Chat,
    environment_client: Chat | None = None,
    *,
    config: GenerationConfig | None = None,
    limit: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Generate traces for every prompt not yet done; resumable and safe to stop between samples.

    Accepted samples go to `out_dir/traces`, rejections (with reasons) to `out_dir/rejects`; both
    count as done. Returns counts and whether the run stopped early.
    """
    traces = ShardWriter(out_dir / "traces", "traces")
    rejects = ShardWriter(out_dir / "rejects", "rejects")
    finished = done_ids(out_dir / "traces") | done_ids(out_dir / "rejects")
    todo = [PromptRecord(**r) for r in read_jsonl(prompts_path) if r["id"] not in finished]
    if limit is not None:
        todo = todo[:limit]
    counts = {"todo": len(todo), "accepted": 0, "rejected": 0, "stopped": False}
    env_client = environment_client or teacher
    for record in todo:
        if should_stop():
            counts["stopped"] = True
            break
        try:
            sample, reason = generate_trace(record, teacher, env_client, config)
        except TeacherOutputError:
            sample, reason = None, "teacher_output_unparsable"
        except TeacherError as error:
            # The server is gone (paused, crashed): leave the id undone and let the caller retry.
            counts["stopped"] = True
            counts["error"] = str(error)
            break
        except Exception as error:  # one bad sample must not end an overnight run
            sample, reason = None, f"error:{type(error).__name__}"
        if sample:
            traces.write(sample)
            counts["accepted"] += 1
        else:
            rejects.write({"id": record.id, "task_class": record.task_class, "reason": reason})
            counts["rejected"] += 1
        log(f"{record.id} {'ok' if sample else reason}")
    return counts
