"""Regression tests for filling rounded-down stratified sample quotas."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from docetl.operations.sample import SampleOperation


@pytest.mark.parametrize("samples", [3, 0.75])
@pytest.mark.parametrize(
    "data, stratify_key, expected",
    [
        (
            [{"group": key} for key in "ABCD"],
            "group",
            [{"group": key} for key in "ABC"],
        ),
        (
            [{"group": key} for key in "AABB"],
            "group",
            [{"group": key} for key in "ABA"],
        ),
        (
            [{"group": "A", "part": key} for key in "1234"],
            ["group", "part"],
            [{"group": "A", "part": key} for key in "123"],
        ),
    ],
    ids=["exhausted-largest-group", "equal-rows", "multiple-stratify-keys"],
)
def test_stratified_first_fills_remainder(data, stratify_key, expected, samples):
    # A separate process bounds failures if sampling stops making progress.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
from types import SimpleNamespace
from docetl.operations.sample import SampleOperation

data, config = json.load(sys.stdin)
operation = SampleOperation(SimpleNamespace(config={}), config, "gpt-4o-mini", 1)
json.dump(operation.execute(data), sys.stdout)
""",
        ],
        input=json.dumps(
            [
                data,
                {
                    "name": "sample",
                    "type": "sample",
                    "method": "first",
                    "samples": samples,
                    "stratify_key": stratify_key,
                },
            ]
        ),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    output, cost = json.loads(result.stdout)
    assert output == expected
    assert cost == 0.0


@pytest.mark.parametrize("samples", [0.1, 1, 2, 3, 4, 10])
def test_stratified_first_preserves_existing_selection(samples):
    data = [{"group": key, "id": index} for index, key in enumerate("AAAB")]
    operation = SampleOperation(
        SimpleNamespace(config={}),
        {
            "name": "sample",
            "type": "sample",
            "method": "first",
            "samples": samples,
            "stratify_key": "group",
        },
        "gpt-4o-mini",
        1,
    )

    output, cost = operation.execute(data)

    count = int(samples * len(data)) if isinstance(samples, float) else samples
    assert output == data[:count]
    assert cost == 0.0
