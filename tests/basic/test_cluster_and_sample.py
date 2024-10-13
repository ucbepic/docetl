import pytest
from docetl.operations.cluster import ClusterOperation
from docetl.operations.sample import SampleOperation
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

        {% for input in inputs %}
        {{input.concept}}:
        {{input.description}}
        {% endfor %}

        Provide the title of the super-concept, and a description.
        """,
        "model": "gpt-4o-mini",
    }


@pytest.fixture
def sample_data():
    return [
        {
            "id": 1,
            "concept": "Shed",
            "description": "A simple, single-story roofed structure, often used for storage or as a workshop.",
            "group": "A",
        },
        {
            "id": 2,
            "concept": "Barn",
            "description": "A large agricultural building used for storing farm products and sheltering livestock.",
            "group": "B",
        },
        {
            "id": 3,
            "concept": "Tree house",
            "description": "A small house built among the branches of a tree for children to play in.",
            "group": "A",
        },
        {
            "id": 4,
            "concept": "Skyscraper",
            "description": "A very tall building of many stories, typically found in urban areas.",
            "group": "B",
        },
        {
            "id": 5,
            "concept": "Castle",
            "description": "A large fortified building or set of buildings from the medieval period.",
            "group": "A",
        },
        {
            "id": 6,
            "concept": "Igloo",
            "description": "A dome-shaped dwelling made of blocks of solid snow, traditionally built by Inuit people.",
            "group": "B",
        },
        {
            "id": 7,
            "concept": "Lighthouse",
            "description": "A tower with a bright light at the top, used to warn or guide ships at sea.",
            "group": "A",
        },
        {
            "id": 8,
            "concept": "Windmill",
            "description": "A building with sails or vanes that turn in the wind and generate power to grind grain into flour.",
            "group": "B",
        },
    ]


def test_cluster_operation(
    cluster_config, sample_data, api_wrapper, default_model, max_threads
):
    cluster_config["bypass_cache"] = True
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


@pytest.fixture
def sample_config():
    return {
        "name": "sample_operation",
        "type": "sample",
        "random_state": 42,  # For reproducibility
    }


def test_sample_operation_with_count(
    sample_config, sample_data, api_wrapper, default_model, max_threads
):
    sample_config["samples"] = 5
    sample_config["method"] = "uniform"
    operation = SampleOperation(api_wrapper, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == 5
    assert cost == 0
    assert all(item in sample_data for item in results)


def test_sample_operation_with_fraction(
    sample_config, sample_data, api_wrapper, default_model, max_threads
):
    sample_config["samples"] = 0.5
    sample_config["method"] = "uniform"
    operation = SampleOperation(api_wrapper, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == len(sample_data) // 2
    assert cost == 0
    assert all(item in sample_data for item in results)


def test_sample_operation_with_list(
    sample_config, sample_data, api_wrapper, default_model, max_threads
):
    sample_list = [{"id": 1}, {"id": 3}, {"id": 5}]
    sample_config["samples"] = sample_list
    sample_config["method"] = "custom"
    operation = SampleOperation(api_wrapper, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == len(sample_list)
    assert cost == 0
    assert all(item["id"] in [1, 3, 5] for item in results)


def test_sample_operation_with_stratify(
    sample_config, sample_data, api_wrapper, default_model, max_threads
):
    sample_config["samples"] = 5
    sample_config["method"] = "stratify"
    sample_config["method_kwargs"] = {"stratify_key": "group"}
    operation = SampleOperation(api_wrapper, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == 5
    assert cost == 0
    assert all(item in sample_data for item in results)
    assert len(set(item["group"] for item in results)) > 1


def test_sample_operation_with_outliers(
    sample_config, sample_data, api_wrapper, default_model, max_threads
):
    sample_config["method"] = "outliers"
    sample_config["method_kwargs"] = {
        "std": 2,
        "embedding_keys": ["concept", "description"],
        "keep": True,
    }
    operation = SampleOperation(api_wrapper, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) < len(sample_data)
    assert cost > 0
    assert all(item in sample_data for item in results)


def test_sample_operation_empty_input(
    sample_config, api_wrapper, default_model, max_threads
):
    sample_config["samples"] = 3
    sample_config["method"] = "uniform"
    operation = SampleOperation(api_wrapper, sample_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_sample_operation_with_outliers_and_center(
    sample_config, sample_data, api_wrapper, default_model, max_threads
):
    sample_config["method"] = "outliers"
    sample_config["method_kwargs"] = {
        "std": 2,
        "embedding_keys": ["concept", "description"],
        "keep": True,
        "center": {
            "concept": "Tree house",
            "description": "A small house built among the branches of a tree for children to play in.",
        },
    }
    operation = SampleOperation(api_wrapper, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) < len(sample_data)
    assert cost > 0
    assert all(item in sample_data for item in results)
