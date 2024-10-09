import pytest
from docetl.operations.cluster import ClusterOperation
from tests.conftest import api_wrapper, default_model, max_threads


@pytest.fixture
def cluster_config():
    return {
        "name": "test_cluster",
        "type": "cluster",
        "embedding_keys": ["concept", "description"],
        "output_key": "categories",
        "summary_schema": {"concept": "string", "description": "string"},
        "summary_prompt": """
        The following describes two related concepts. What concept
        encompasses both? Try not to be too broad; it might be that one of
        these two concepts already encompasses the other; in that case,
        you should just use that concept.

        {{left.concept}}:
        {{left.description}}

        {{right.concept}}:
        {{right.description}}

        Provide the title of the super-concept, and a description.
        """,
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def sample_data():
    return [
        {
            "concept": "Shed",
            "description": "A simple, single-story roofed structure, often used for storage or as a workshop.",
        },
        {
            "concept": "Barn",
            "description": "A large agricultural building used for storing farm products and sheltering livestock.",
        },
        {
            "concept": "Tree house",
            "description": "A small house built among the branches of a tree for children to play in.",
        },
        {
            "concept": "Skyscraper",
            "description": "A very tall building of many stories, typically found in urban areas.",
        },
        {
            "concept": "Castle",
            "description": "A large fortified building or set of buildings from the medieval period.",
        },
        {
            "concept": "Igloo",
            "description": "A dome-shaped dwelling made of blocks of solid snow, traditionally built by Inuit people.",
        },
        {
            "concept": "Lighthouse",
            "description": "A tower with a bright light at the top, used to warn or guide ships at sea.",
        },
        {
            "concept": "Windmill",
            "description": "A building with sails or vanes that turn in the wind and generate power to grind grain into flour.",
        },
    ]


def test_cluster_operation(
    cluster_config, sample_data, api_wrapper, default_model, max_threads
):
    operation = ClusterOperation(
        api_wrapper, cluster_config, default_model, max_threads
    )
    results, cost = operation.execute(sample_data)

    assert len(results) == len(sample_data)
    assert cost > 0

    for result in results:
        assert "categories" in result
        assert isinstance(result["categories"], tuple)
        assert len(result["categories"]) > 0

        for category in result["categories"]:
            assert "concept" in category
            assert "description" in category


def test_cluster_operation_empty_input(
    cluster_config, api_wrapper, default_model, max_threads
):
    operation = ClusterOperation(
        api_wrapper, cluster_config, default_model, max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_cluster_operation_single_item(
    cluster_config, api_wrapper, default_model, max_threads
):
    single_item = [
        {"concept": "House", "description": "A building for human habitation."}
    ]
    operation = ClusterOperation(
        api_wrapper, cluster_config, default_model, max_threads
    )
    results, cost = operation.execute(single_item)

    assert len(results) == 1
    assert cost == 0
    assert "categories" in results[0]
    assert isinstance(results[0]["categories"], tuple)
