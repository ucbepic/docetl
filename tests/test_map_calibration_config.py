import copy
from types import SimpleNamespace

from docetl.operations.map import MapOperation


def test_map_calibration_does_not_mutate_config():
    config = {
        "name": "calibrated-map",
        "type": "map",
        "prompt": "Classify {{ input.text }}",
        "output": {"schema": {"label": "string"}},
        "calibrate": True,
        "num_calibration_docs": 1,
    }
    original_config = copy.deepcopy(config)

    class CaptureAPI:
        def __init__(self):
            self.calls = []

        def call_llm(self, model, op_type, messages, schema, **kwargs):
            assert config == original_config
            self.calls.append(
                {
                    "op_type": op_type,
                    "prompt": messages[0]["content"],
                    "op_config": copy.deepcopy(kwargs["op_config"]),
                }
            )
            if op_type == "calibration":
                response = {"calibration_context": "Use the stable anchor."}
            else:
                response = {"label": "stable"}
            return SimpleNamespace(response=response, validated=True, total_cost=0.0)

        def parse_llm_response(self, response, **kwargs):
            return [response]

        def validate_output(self, config, output, console):
            return True

    api = CaptureAPI()
    runner = SimpleNamespace(config={}, api=api, is_cancelled=False)
    operation = MapOperation(runner, config, "gpt-4o-mini", max_threads=1)

    for _ in range(2):
        output, cost = operation.execute([{"text": "this"}])
        assert output == [{"text": "this", "label": "stable"}]
        assert cost == 0.0
        assert config == original_config

    map_calls = [call for call in api.calls if call["op_type"] == "map"]
    assert [call["prompt"] for call in map_calls] == [
        "Classify this",
        "Classify this\n\nUse the stable anchor.",
        "Classify this",
        "Classify this\n\nUse the stable anchor.",
    ]
    assert [call["op_config"]["prompt"] for call in map_calls] == [
        original_config["prompt"],
        f"{original_config['prompt']}\n\nUse the stable anchor.",
        original_config["prompt"],
        f"{original_config['prompt']}\n\nUse the stable anchor.",
    ]
