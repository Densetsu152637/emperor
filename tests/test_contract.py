import pytest

from emperor.eval.contract import (
    ALL_TOOLS,
    SchemaError,
    check_call,
    contract_source,
    openai_tools,
    system_prompt,
    tool_schemas,
    validate,
)


def test_vendored_contract_has_every_tool():
    assert set(tool_schemas()) == set(ALL_TOOLS)
    assert len(contract_source()["commit"]) == 40


def test_validator_supports_every_vendored_schema():
    for name, schema in tool_schemas().items():
        validate({}, schema["parameters"])  # raises SchemaError on an unknown keyword
        assert openai_tools([name])[0]["function"]["name"] == name


def test_contract_fields_this_stage_depends_on():
    spawn = tool_schemas()["spawn_agent"]["parameters"]["properties"]
    assert {"brief", "tier", "tags", "location", "single_call"} <= set(spawn)
    step = tool_schemas()["update_plan"]["parameters"]["properties"]["steps"]["items"]
    assert {"depends_on", "owner", "acceptance"} <= set(step["properties"])


def test_validate_reports_violations():
    schema = tool_schemas()["spawn_agent"]["parameters"]
    assert validate({"brief": "b", "tier": "standard"}, schema) == []
    errors = validate({"tier": "huge", "tags": "code", "single_call": "yes"}, schema)
    assert any("missing required property 'brief'" in e for e in errors)
    assert any("'huge' is not one of" in e for e in errors)
    assert any("$.tags: expected array" in e for e in errors)
    assert any("$.single_call: expected boolean" in e for e in errors)
    ask = tool_schemas()["ask_user"]["parameters"]
    assert any("fewer than 1" in e for e in validate({"questions": []}, ask))


def test_integer_is_not_boolean():
    assert validate(True, {"type": "integer"})
    assert validate(3, {"type": "integer"}) == []


def test_unknown_schema_keyword_is_refused():
    with pytest.raises(SchemaError):
        validate({}, {"type": "object", "additionalProperties": False})


def test_check_call():
    offered = ["spawn_agent", "await_tasks"]
    assert check_call("await_tasks", "", offered) == ({}, [])
    assert check_call("web_fetch", "{}", offered)[1] == ["tool 'web_fetch' was not offered"]
    arguments, errors = check_call("spawn_agent", "{brief: 1}", offered)
    assert arguments is None and "not JSON" in errors[0]
    assert check_call("spawn_agent", "[]", offered)[1]


def test_system_prompt_states_the_tier():
    text = system_prompt(98_304)
    assert text.startswith("# Emperor: root system prompt")
    assert text.rstrip().endswith("context budget: 96K tokens")
