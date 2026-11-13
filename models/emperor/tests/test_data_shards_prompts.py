from emperor.data import prompts as prompts_module
from emperor.data.prompts import (
    CLASSES,
    HeldoutIndex,
    generate_prompts,
    normalise,
    overlap,
)
from emperor.data.shards import ShardWriter, done_ids, iter_records


def test_prompts_are_seeded_and_deduplicated():
    a, _ = generate_prompts(300, seed=5)
    b, _ = generate_prompts(300, seed=5)
    c, _ = generate_prompts(300, seed=6)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
    assert [p.text for p in a] != [p.text for p in c]
    keys = [normalise(p.text) for p in a]
    assert len(keys) == len(set(keys))
    assert len({p.id for p in a}) == len(a)


def test_prompts_cover_the_task_classes_and_vary_in_specificity():
    prompts, _ = generate_prompts(600, seed=1)
    assert {p.task_class for p in prompts} == set(CLASSES)
    assert {"vague", "partial", "specific"} <= {p.specificity for p in prompts}
    long_input = [p for p in prompts if p.task_class == "long_input"]
    assert long_input and all("delegate_chunks" in p.tools for p in long_input)


def test_existing_texts_are_not_regenerated():
    first, _ = generate_prompts(100, seed=2)
    again, stats = generate_prompts(100, seed=2, existing=[normalise(p.text) for p in first])
    assert not ({p.id for p in first} & {p.id for p in again})
    assert stats["duplicate"] > 0


def test_overlap_catches_copies_and_near_copies():
    reference = "Add unit tests for the date parser in ~/code/ledger/src/dates.py"
    assert overlap(reference, reference) == 1.0
    assert overlap(reference + " please", reference) > 0.5
    assert overlap("Build a recipe app with offline mode", reference) < 0.1


def test_heldout_overlap_is_filtered_out():
    index = HeldoutIndex({"h/1": "Make our web app faster."}, threshold=0.3)
    assert index.leaks("Make our web app faster")
    assert index.leaks("please make our web app faster!")
    assert not index.leaks("Compare PostgreSQL and Redis for a public API.")


def test_generated_prompts_do_not_overlap_the_bundled_eval_tasks():
    index = HeldoutIndex.bundled()
    assert len(index.texts) >= 15
    prompts, _ = generate_prompts(500, seed=3, heldout=index)
    assert len(prompts) == 500
    for prompt in prompts:
        assert index.closest(prompt.text)[0] < index.threshold, prompt.text


def test_generation_drops_candidates_that_overlap():
    texts = {f"t/{i}": p.text for i, p in enumerate(generate_prompts(50, seed=9)[0])}
    index = HeldoutIndex(texts)
    fresh, stats = prompts_module.generate_prompts(50, seed=9, heldout=index)
    assert stats["heldout_overlap"] > 0
    assert not {normalise(p.text) for p in fresh} & {normalise(t) for t in texts.values()}


def test_shards_roll_over_resume_and_skip_torn_lines(tmp_path):
    writer = ShardWriter(tmp_path, "s", max_records=2)
    for i in range(5):
        writer.write({"id": f"a{i}"})
    assert [p.name for p in sorted(tmp_path.glob("*.jsonl"))] == [
        "s-00001.jsonl",
        "s-00002.jsonl",
        "s-00003.jsonl",
    ]
    last = tmp_path / "s-00003.jsonl"
    with last.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "torn')  # a crash mid-write
    assert done_ids(tmp_path) == {f"a{i}" for i in range(5)}
    resumed = ShardWriter(tmp_path, "s", max_records=2)
    resumed.write({"id": "a5"})
    resumed.write({"id": "a6"})
    assert done_ids(tmp_path) == {f"a{i}" for i in range(7)}
    assert len(list(iter_records(tmp_path))) == 7
