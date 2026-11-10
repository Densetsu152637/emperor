"""Release tooling: GGUF export commands, model cards, publish guard, pipeline wiring.

No GPU, no model and no network: subprocess calls are faked and files are tiny stand-ins.
"""

import json
from pathlib import Path

import pytest

from emperor.export import gguf
from emperor.pipeline.idle import IdleGate
from emperor.pipeline.processes import OwnProcesses
from emperor.pipeline.run import PHASES, Pipeline, Settings
from emperor.release import card, publish
from emperor.release.cli import main as release_main
from emperor.train.config import ConfigError, load_config, parse_config

SAMPLE_REPORT = {
    "model": "llamacpp:emperor",
    "suite": "heldout",
    "backend": "replay",
    "dry_run": False,
    "context_tier": "64k",
    "context_tokens": 65536,
    "timestamp": "2026-01-02T03:04:05+00:00",
    "judge": "heuristic",
    "tasks_run": 15,
    "tasks_failed": 1,
    "metrics": {
        "tool_validity": {"goal": "G1", "n": 40, "score": 0.975, "target": 0.98, "passed": False},
        "routing": {"goal": "G2", "n": 15, "score": 0.9, "target": 0.85, "passed": True},
        "retrieval": {"goal": "G6", "n": 0, "score": None, "target": None, "passed": None},
    },
}
DATA_REPORT = {
    "prompts": 2000,
    "generated": 1800,
    "judged": 1700,
    "kept": 1200,
    "pass_rate": 0.7059,
    "kept_by_class": {"coding": 700, "planning": 500},
}


class FakeRun:
    """Stands in for subprocess.run: records commands and creates the output file they name."""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command, check=True):
        self.commands.append([str(c) for c in command])
        if command[0] == "git":
            target = Path(command[-1])
            target.mkdir(parents=True)
            (target / "convert_hf_to_gguf.py").write_text("# converter", encoding="utf-8")
        elif "--outfile" in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(b"bf16" * 10)
        else:  # llama-quantize <in> <out> <type>
            Path(command[2]).write_bytes(b"q" * 4)


@pytest.fixture
def model_dir(tmp_path):
    weights = tmp_path / "weights" / "emperor-4b-bf16"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text(
        json.dumps({"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128}),
        encoding="utf-8",
    )
    return tmp_path


def test_release_config_defaults_and_validation():
    config = parse_config({})
    assert config.release.name == "emperor-4b"
    assert config.release.gguf_types == ["Q8_0", "Q6_K", "Q4_K_M"]
    with pytest.raises(ConfigError):
        parse_config({"release": {"name": "a/b"}})
    with pytest.raises(ConfigError):
        parse_config({"release": {"gguf_types": []}})
    assert load_config(Path(__file__).parent.parent / "configs" / "sft.toml").release.title


def test_export_clones_pinned_tag_converts_then_quantizes(model_dir, tmp_path):
    run = FakeRun()
    quantize = tmp_path / "llama-quantize"
    quantize.write_text("", encoding="utf-8")
    out = model_dir / "weights" / "emperor-4b-gguf"
    result = gguf.export_gguf(
        model_dir / "weights" / "emperor-4b-bf16",
        out,
        "emperor-4b",
        ["Q8_0", "Q4_K_M"],
        cache=model_dir / "tools" / "llama.cpp",
        quantize_binary=quantize,
        run=run,
    )
    clone, convert, q8, q4 = run.commands
    assert clone[:6] == ["git", "clone", "--depth", "1", "--branch", gguf.LLAMA_CPP_TAG]
    assert "--outtype" in convert and convert[convert.index("--outtype") + 1] == "bf16"
    assert convert[1].endswith("convert_hf_to_gguf.py")
    assert q8[0] == str(quantize) and q8[-1] == "Q8_0" and q8[1].endswith("emperor-4b-bf16.gguf")
    assert q4[-1] == "Q4_K_M"
    names = sorted(f["file"] for f in result.files)
    assert names == [f"emperor-4b-{k}.gguf" for k in ("Q4_K_M", "Q8_0", "bf16")]
    assert not list(out.glob("*.tmp"))
    assert json.loads((out / "export.json").read_text(encoding="utf-8"))["files"]


def test_export_is_resumable(model_dir, tmp_path):
    quantize = tmp_path / "llama-quantize"
    quantize.write_text("", encoding="utf-8")
    args = (model_dir / "weights" / "emperor-4b-bf16", model_dir / "weights" / "g", "m", ["Q8_0"])
    kwargs = {"cache": model_dir / "tools" / "llama.cpp", "quantize_binary": quantize}
    gguf.export_gguf(*args, run=FakeRun(), **kwargs)
    again = FakeRun()
    gguf.export_gguf(*args, run=again, **kwargs)
    assert again.commands == []  # converter cached, every file present


def test_export_needs_merged_weights_and_a_quantizer(tmp_path, monkeypatch):
    monkeypatch.setattr(gguf, "find_llama_tool", lambda *a, **k: None)
    with pytest.raises(gguf.ExportError, match="no merged model"):
        gguf.export_gguf(tmp_path, tmp_path / "g", "m", ["Q8_0"], cache=tmp_path / "c")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(gguf.ExportError, match="llama-quantize"):
        gguf.export_gguf(tmp_path, tmp_path / "g", "m", ["Q8_0"], cache=tmp_path / "c")


def test_explicit_llama_cpp_checkout_is_used_without_cloning(tmp_path):
    (tmp_path / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    run = FakeRun()
    script = gguf.ensure_llama_cpp(tmp_path / "cache", llama_cpp=tmp_path, run=run)
    assert script == tmp_path / "convert_hf_to_gguf.py" and run.commands == []


def test_front_matter_and_kv_cache():
    text = card.front_matter({"license": "apache-2.0", "tags": ["a", "b"], "flag": True})
    assert text == '---\nlicense: "apache-2.0"\ntags:\n- "a"\n- "b"\nflag: true\n---'
    # Qwen3-4B: 36 layers x 8 KV heads x 128 x 2 (K,V) x 2 bytes = 144 KiB per token
    assert card.kv_cache_gib(1024) == pytest.approx(0.140625)


def test_card_renders_every_section_from_a_sample_report():
    config = parse_config({})
    inputs = card.CardInputs(
        config=config,
        eval_report=SAMPLE_REPORT,
        data_report=DATA_REPORT,
        train_result={"metrics": {"train_loss": 0.4321}},
        gguf_files=[{"type": "bf16", "file": "emperor-4b-bf16.gguf", "bytes": 8 * 2**30}],
    )
    text = card.render_card(inputs, "bf16")
    assert text.startswith('---\nlicense: "apache-2.0"\nbase_model: "Qwen/Qwen3-4B-Instruct-2507"')
    assert 'pipeline_tag: "text-generation"' in text
    assert "| tool_validity | G1 | 40 | 0.975 | 0.98 | no |" in text
    assert "| retrieval | G6 | 0 | n/a | n/a | n/a |" in text
    assert "r=32, alpha=64" in text and "Final training loss: 0.4321" in text
    assert "Teacher: gpt-oss-20b (Apache-2.0)" in text and "Judge pass rate: 70.6%" in text
    assert "coding 700, planning 500" in text
    assert "## Memory and context" in text and "| bf16 | 8.0 |" in text
    assert "dry run" not in text
    gguf_text = card.render_card(inputs, "gguf")
    assert 'library_name: "gguf"' in gguf_text and "`emperor-4b-bf16.gguf`" in gguf_text


def test_card_flags_a_dry_run_and_a_missing_report():
    config = parse_config({})
    dry = card.eval_table({**SAMPLE_REPORT, "dry_run": True})
    assert "not a measurement" in dry
    assert "No evaluation report" in card.eval_table(None)
    text = card.render_card(card.CardInputs(config=config), "bf16")
    assert "dataset report was not available" in text


def test_write_cards_picks_up_latest_report_and_export(model_dir):
    config = parse_config({})
    old = model_dir / "runs" / "20260101-000000"
    new = model_dir / "runs" / "20260102-000000"
    for run, score in ((old, 0.1), (new, 0.9)):
        run.mkdir(parents=True)
        report = {**SAMPLE_REPORT, "metrics": {"routing": {**SAMPLE_REPORT["metrics"]["routing"]}}}
        report["metrics"]["routing"]["score"] = score
        (run / "eval.json").write_text(json.dumps(report), encoding="utf-8")
    import os

    os.utime(old / "eval.json", (1, 1))
    gguf_dir = model_dir / "weights" / "emperor-4b-gguf"
    gguf_dir.mkdir()
    (gguf_dir / "export.json").write_text(
        json.dumps(
            {"files": [{"type": "Q8_0", "file": "emperor-4b-Q8_0.gguf", "bytes": 4 * 2**30}]}
        ),
        encoding="utf-8",
    )
    written = card.write_cards(config, model_dir)
    assert [p.parent.name for p in written] == ["emperor-4b-bf16", "emperor-4b-gguf"]
    assert "0.900" in written[0].read_text(encoding="utf-8")
    assert "emperor-4b-Q8_0.gguf" in written[1].read_text(encoding="utf-8")


def test_write_cards_needs_merged_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        card.write_cards(parse_config({}), tmp_path)


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs["repo_id"], kwargs["allow_patterns"]))


def test_publish_refuses_without_confirm_and_never_touches_the_hub(model_dir):
    folder = model_dir / "weights" / "emperor-4b-bf16"
    (folder / "README.md").write_text("card", encoding="utf-8")
    api = FakeApi()
    uploads = [publish.Upload("owner/emperor-4b", folder)]
    with pytest.raises(publish.PublishRefused, match="--confirm"):
        publish.publish(uploads, confirm=False, api_factory=lambda: api)
    assert api.calls == []
    urls = publish.publish(uploads, confirm=True, api_factory=lambda: api)
    assert urls == ["https://huggingface.co/owner/emperor-4b"]
    assert api.calls[0][2]["private"] is True
    assert api.calls[1][0] == "upload" and "*.gguf" in api.calls[1][2]


def test_publish_checks_repo_id_and_card(model_dir):
    folder = model_dir / "weights" / "emperor-4b-bf16"
    with pytest.raises(publish.PublishRefused, match="owner/name"):
        publish.publish([publish.Upload("nonsense", folder)], confirm=True)
    with pytest.raises(publish.PublishRefused, match="README"):
        publish.publish([publish.Upload("o/n", folder)], confirm=True)
    with pytest.raises(publish.PublishRefused, match="nothing"):
        publish.publish([], confirm=True)


def test_release_cli_publish_without_confirm_exits_2(model_dir, capsys):
    config = model_dir / "sft.toml"
    config.write_text("", encoding="utf-8")
    (model_dir / "weights" / "emperor-4b-bf16" / "README.md").write_text("c", encoding="utf-8")
    code = release_main(
        ["publish", "--config", str(config), "--root", str(model_dir), "--repo-id", "o/n"]
    )
    assert code == 2 and "--confirm" in capsys.readouterr().err


def test_release_cli_card_writes_readme(model_dir, capsys):
    config = model_dir / "sft.toml"
    config.write_text("", encoding="utf-8")
    assert release_main(["card", "--config", str(config), "--root", str(model_dir)]) == 0
    assert (model_dir / "weights" / "emperor-4b-bf16" / "README.md").is_file()


# -- pipeline wiring ---------------------------------------------------------------------------


def make_pipeline(root: Path, **kwargs) -> Pipeline:
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "sft.toml").write_text("", encoding="utf-8")
    run_dir = root / "runs" / "r1"
    run_dir.mkdir(parents=True)
    settings = Settings(root=root, run_dir=run_dir, data_dir=root / "data", **kwargs)
    return Pipeline(settings, IdleGate(asleep=True), OwnProcesses())


def test_phases_end_with_export_eval_card():
    assert PHASES[-3:] == ["export", "eval", "card"]


def test_export_phase_runs_the_release_cli_and_records_files(model_dir, monkeypatch):
    pipeline = make_pipeline(model_dir, llama_cpp=Path("L"), quantize=Path("Q"))
    seen = {}

    def fake_run_module(name, args):
        seen["args"] = args
        directory = model_dir / "weights" / "emperor-4b-gguf"
        directory.mkdir(parents=True)
        (directory / "export.json").write_text(
            json.dumps({"files": [{"file": "emperor-4b-bf16.gguf", "type": "bf16", "bytes": 1}]}),
            encoding="utf-8",
        )
        return {"status": "done"}

    monkeypatch.setattr(pipeline, "_run_module", fake_run_module)
    result = pipeline.phase_export()
    assert result["status"] == "done" and result["files"] == ["emperor-4b-bf16.gguf"]
    assert seen["args"][:2] == ["emperor.release.cli", "export"]
    assert "--llama-cpp" in seen["args"] and "--quantize" in seen["args"]


def test_eval_skips_without_a_gguf_and_card_runs_after(model_dir):
    pipeline = make_pipeline(model_dir)
    assert pipeline.phase_eval()["status"] == "skipped"
    result = pipeline.phase_card()
    assert result["status"] == "done" and len(result["cards"]) == 1


def test_eval_command_uses_the_configured_alias(model_dir):
    pipeline = make_pipeline(model_dir)
    command = pipeline.eval_command(Path("eval.json"))
    assert "llamacpp:emperor" in command


def test_only_runs_one_phase_and_leaves_other_state_alone(model_dir):
    pipeline = make_pipeline(model_dir, only={"card"})
    pipeline.state["phases"]["train"] = {"status": "done"}
    summary = pipeline.run()
    assert summary["phases"]["train"] == {"status": "done"}
    assert summary["phases"]["card"]["status"] == "done"
    assert set(summary["phases"]) == {"train", "card"}


def test_chat_only_samples_have_no_tool_definitions():
    from emperor.train.data import _tool_definitions

    assert _tool_definitions({"messages": []}) is None
    assert _tool_definitions({"tools": [], "messages": []}) is None
