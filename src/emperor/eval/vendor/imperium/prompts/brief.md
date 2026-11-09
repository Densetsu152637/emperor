# Brief template

The `brief` argument of `spawn_agent` (and of `delegate_chunks` when it is offered). The
subagent sees only this text: it has none of your conversation, so everything it needs must be
in it. Keep it under 2,000 tokens. Do not paste large material: name where it lives (a file
path, a URL, or a stored artefact when `delegate_chunks` is offered) so the subagent can read
it.

## Goal

One or two sentences: the outcome wanted and why it matters.

## Context

Only the facts the task needs: relevant paths, names, decisions already made, findings from
earlier subagents. Say which of them are verified and which are assumptions.

## Constraints

What must not change, what is out of scope, limits on time, cost, tools or style.

## Acceptance

Checks that show the work is done, each one something the subagent can run or observe.

## Output format

Exactly what the final report contains and in what shape: for example a short summary, the list
of changed files, the evidence for each acceptance check, and anything left undone or uncertain.
