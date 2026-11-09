# Plan template

The shape of the plan you keep with `update_plan`. Each call replaces the whole plan, so send
every field each time. Revise it when evidence contradicts an assumption, and say what changed.

## Goal

`goal`: the measurable outcome, in the user's terms, and how you will know it is met.

## Stages

`steps`, in dependency order. For each stage:

- `title`: what it does, in a few words; unique within the plan.
- `status`: `pending`, `in_progress`, `done`, `blocked` or `skipped`. Keep exactly one stage
  `in_progress` while you work on it yourself.
- `depends_on`: the titles of the stages that must finish first. Stages with no unmet dependency
  run in parallel as separate subagent tasks.
- `owner`: yourself, or the subagent (and tier) that does it.
- `acceptance`: the check that shows the stage is done.

## Assumptions

`assumptions`: each gap you filled yourself, by inference, a default or research, with the
value you chose.

## Open questions

`open_questions`: only decisions that are the user's to make or cannot be researched. When
`ask_user` is offered, ask them together in one batch (see the question format). Otherwise put
the same batch in your reply to the user. Either way, keep working on stages that do not depend
on them.
