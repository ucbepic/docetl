"""Non-Jinja prompt confirmation should run once per shared config."""

from unittest.mock import MagicMock, patch

import pytest

from docetl.operations.map import MapOperation
from docetl.utils import ensure_non_jinja_prompt_confirmed


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.config = {"bypass_cache": True}
    runner.api = MagicMock()
    return runner


def test_ensure_non_jinja_prompt_confirmed_is_idempotent():
    config = {
        "name": "map_1",
        "type": "map",
        "prompt": "what are the medications in this doc?",
    }
    with patch(
        "docetl.utils.prompt_user_for_non_jinja_confirmation", return_value=True
    ) as confirm:
        ensure_non_jinja_prompt_confirmed(
            config, "prompt", "_append_document_to_prompt"
        )
        ensure_non_jinja_prompt_confirmed(
            config, "prompt", "_append_document_to_prompt"
        )

    assert confirm.call_count == 1
    assert config["_append_document_to_prompt"] is True


def test_ensure_non_jinja_prompt_confirmed_skips_jinja_prompts():
    config = {
        "name": "map_1",
        "type": "map",
        "prompt": "Extract meds from {{ input.text }}",
    }
    with patch(
        "docetl.utils.prompt_user_for_non_jinja_confirmation", return_value=True
    ) as confirm:
        ensure_non_jinja_prompt_confirmed(
            config, "prompt", "_append_document_to_prompt"
        )

    confirm.assert_not_called()
    assert "_append_document_to_prompt" not in config


def test_ensure_non_jinja_prompt_confirmed_stops_status_before_prompt():
    config = {
        "name": "map_1",
        "type": "map",
        "prompt": "plain prompt",
    }
    status = MagicMock()
    with patch(
        "docetl.utils.prompt_user_for_non_jinja_confirmation", return_value=True
    ) as confirm:
        ensure_non_jinja_prompt_confirmed(
            config,
            "prompt",
            "_append_document_to_prompt",
            status=status,
        )

    # Status is stopped inside prompt_user_for_non_jinja_confirmation
    confirm.assert_called_once()
    assert confirm.call_args.kwargs["status"] is status


def test_map_operation_confirms_once_across_instantiations(mock_runner):
    config = {
        "name": "map_1",
        "type": "map",
        "prompt": "what are the medications in this doc?",
        "output": {"schema": {"field": "string"}},
    }
    with patch(
        "docetl.utils.prompt_user_for_non_jinja_confirmation", return_value=True
    ) as confirm:
        MapOperation(mock_runner, config, "gpt-4o-mini", 4)
        MapOperation(mock_runner, config, "gpt-4o-mini", 4)

    assert confirm.call_count == 1
    assert config["_append_document_to_prompt"] is True


def test_runner_syntax_check_uses_live_config_so_execution_skips_reprompt(
    tmp_path, mock_runner
):
    """syntax_check + _run_operation must share one config and prompt once."""
    from docetl.runner import DSLRunner

    data_path = tmp_path / "data.json"
    data_path.write_text('[{"text": "aspirin"}]')

    config = {
        "datasets": {
            "data": {"type": "file", "path": str(data_path), "source": "local"}
        },
        "operations": [
            {
                "name": "map_1",
                "type": "map",
                "prompt": "what are the medications in this doc?",
                "output": {"schema": {"field": "string"}},
            }
        ],
        "pipeline": {
            "steps": [{"name": "step_map_1", "input": "data", "operations": ["map_1"]}],
            "output": {"type": "file", "path": str(tmp_path / "out.json")},
        },
        "default_model": "gpt-4o-mini",
        "bypass_cache": True,
    }

    with patch(
        "docetl.utils.prompt_user_for_non_jinja_confirmation", return_value=True
    ) as confirm:
        runner = DSLRunner(config, max_threads=2)
        # syntax_check already ran in __init__
        assert confirm.call_count == 1
        assert runner.find_operation("map_1")["_append_document_to_prompt"] is True

        # Constructing again for execution must not re-prompt.
        runner._make_operation(runner.find_operation("map_1"))
        assert confirm.call_count == 1
