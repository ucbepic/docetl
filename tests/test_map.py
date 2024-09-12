import pytest
from docetl.operations.map import MapOperation


@pytest.fixture
def map_config_with_tools():
    return {
        "type": "map",
        "name": "word_count",
        "prompt": "Count the number of words in the following text: '{{ input.text }}'",
        "output": {"schema": {"word_count": "integer"}},
        "model": "gpt-4o-mini",
        "tools": [
            {
                "required": True,
                "code": """
def count_words(text):
    return {"word_count": len(text.split())}
                """,
                "function": {
                    "name": "count_words",
                    "description": "Count the number of words in a text string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                            }
                        },
                        "required": ["text"],
                    },
                },
            }
        ],
        "validate": ["len(output['text']) > 0"],
        "num_retries_on_validate_failure": 3,
    }


@pytest.fixture
def synthetic_data():
    return [
        {"text": "This is a short sentence."},
        {"text": "This sentence has exactly six words."},
        {"text": "Pneumonoultramicroscopicsilicovolcanoconiosis is a long word."},
        {"text": "One"},
    ]


def test_map_operation_with_word_count_tool(map_config_with_tools, synthetic_data):
    operation = MapOperation(map_config_with_tools, "gpt-4o-mini", 4)
    results, cost = operation.execute(synthetic_data)

    assert len(results) == len(synthetic_data)
    assert all("word_count" in result for result in results)
    assert [result["word_count"] for result in results] == [5, 6, 5, 1]
    assert cost > 0  # Ensure there was some cost associated with the operation
