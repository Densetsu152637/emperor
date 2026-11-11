"""Replay runner: drives the model under test through a task, answering its tool calls from the
task's recorded subagent replies, user answers, and web results. Deterministic and free."""

import json
from dataclasses import dataclass, field

from emperor.eval.contract import check_call, estimate_tokens, openai_tools, system_prompt
from emperor.eval.models import ChatModel, ModelError, Reply
from emperor.eval.tasks import Task, render_prompt

MAX_TURNS = 16


@dataclass
class CallRecord:
    """One tool call the model made, with its validation result and what it was answered."""

    turn: int
    name: str
    raw_arguments: str
    arguments: dict | None
    errors: list[str]
    result: str

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass
class Transcript:
    task_id: str
    replies: list[Reply] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    final: str = ""
    error: str | None = None
    overflowed: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def valid_calls(self, name: str) -> list[CallRecord]:
        return [c for c in self.calls if c.name == name and c.valid]

    def output_text(self) -> str:
        """Everything the model wrote: message text and raw tool arguments."""
        parts = [r.content or "" for r in self.replies]
        parts += [c.raw_arguments for c in self.calls]
        return "\n".join(parts)


class ReplayEnvironment:
    """Answers tool calls from a task's recordings, standing in for imperiumd and subagents."""

    def __init__(self, task: Task):
        self.task = task
        self.unused = list(range(len(task.replies)))
        self.reports: dict[str, str] = {}
        self.pending: list[str] = []
        self.next_id = 1

    def _new_task(self, report: str) -> str:
        task_id = f"t{self.next_id}"
        self.next_id += 1
        self.reports[task_id] = report
        self.pending.append(task_id)
        return task_id

    def _take_reply(self, tool: str, brief: str) -> str:
        """The first unused recording for `tool` whose `match` words all occur in the brief,
        else the first unused recording for `tool`, else a placeholder report."""
        lowered = brief.lower()
        candidates = [
            i for i in self.unused if self.task.replies[i].get("tool", "spawn_agent") == tool
        ]
        for i in candidates:
            if all(w.lower() in lowered for w in self.task.replies[i].get("match", [])):
                self.unused.remove(i)
                return self.task.replies[i]["report"]
        if candidates:
            self.unused.remove(candidates[0])
            return self.task.replies[candidates[0]]["report"]
        return "The subagent finished but left no report."

    def handle(self, name: str, arguments: dict) -> str:
        if name == "spawn_agent":
            return json.dumps(
                {"task_id": self._new_task(self._take_reply(name, arguments["brief"]))}
            )
        if name == "delegate_chunks":
            chunk_replies = [
                i for i in self.unused if self.task.replies[i].get("tool") == "delegate_chunks"
            ]
            ids = []
            for i in chunk_replies:
                self.unused.remove(i)
                ids.append(self._new_task(self.task.replies[i]["report"]))
            if not ids:
                ids.append(self._new_task("The chunk subagent finished but left no report."))
            return json.dumps({"task_ids": ids})
        if name == "await_tasks":
            wanted = arguments.get("task_ids") or list(self.pending)
            done = []
            for task_id in wanted:
                if task_id in self.reports:
                    done.append({"task_id": task_id, "report": self.reports[task_id]})
                    if task_id in self.pending:
                        self.pending.remove(task_id)
                else:
                    done.append({"task_id": task_id, "error": "unknown task id"})
            return json.dumps({"reports": done})
        if name == "update_plan":
            return "Plan updated."
        if name == "ask_user":
            if self.task.user_answers:
                return json.dumps({"answers": self.task.user_answers})
            return "The user has not answered yet. Continue with work that does not depend on it."
        if name == "web_search":
            results = self.task.web.get("search", {}).get(arguments["query"], [])
            return json.dumps({"results": results})
        if name == "web_fetch":
            page = self.task.web.get("fetch", {}).get(arguments["url"])
            return page if page is not None else f"Error: could not fetch {arguments['url']}"
        return f"Error: tool {name} is not available"


def _assistant_message(reply: Reply) -> dict:
    message: dict = {"role": "assistant", "content": reply.content or ""}
    if reply.reasoning:
        # Sent back like the training data has it: a thinking model's template shows the
        # reasoning of the turns since the user's message.
        message["reasoning_content"] = reply.reasoning
    if reply.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in reply.tool_calls
        ]
    return message


def run_task(task: Task, model: ChatModel, context_tokens: int) -> Transcript:
    transcript = Transcript(task_id=task.id)
    environment = ReplayEnvironment(task)
    tools = openai_tools(task.tools)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt(context_tokens)},
        {"role": "user", "content": render_prompt(task, context_tokens)},
    ]
    for turn in range(MAX_TURNS):
        used = sum(
            estimate_tokens(m.get("content") or "")
            + estimate_tokens(m.get("reasoning_content") or "")
            for m in messages
        )
        if used > context_tokens:
            transcript.overflowed = True
            transcript.error = f"context overflow: about {used} tokens > {context_tokens}"
            return transcript
        try:
            reply = model.complete(messages, tools)
        except ModelError as error:
            transcript.error = str(error)
            return transcript
        transcript.replies.append(reply)
        transcript.prompt_tokens += reply.prompt_tokens
        transcript.completion_tokens += reply.completion_tokens
        messages.append(_assistant_message(reply))
        if not reply.tool_calls:
            transcript.final = reply.content or ""
            return transcript
        for call in reply.tool_calls:
            arguments, errors = check_call(call.name, call.arguments, task.tools)
            if errors:
                result = "Error: invalid tool call: " + "; ".join(errors)
            else:
                result = environment.handle(call.name, arguments)
            transcript.calls.append(
                CallRecord(turn, call.name, call.arguments, arguments, errors, result)
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    transcript.error = f"no final answer after {MAX_TURNS} turns"
    return transcript
