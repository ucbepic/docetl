import pytest
from docetl.operations.cluster import ClusterOperation
from docetl.operations.sample import SampleOperation
from tests.conftest import runner, default_model, max_threads


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
    cluster_config, sample_data, runner, default_model, max_threads
):
    cluster_config["bypass_cache"] = True
    operation = ClusterOperation(
        runner, cluster_config, default_model, max_threads
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
    cluster_config, runner, default_model, max_threads
):
    operation = ClusterOperation(
        runner, cluster_config, default_model, max_threads
    )
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_cluster_operation_single_item(
    cluster_config, runner, default_model, max_threads
):
    single_item = [
        {"concept": "House", "description": "A building for human habitation."}
    ]
    operation = ClusterOperation(
        runner, cluster_config, default_model, max_threads
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
    sample_config, sample_data, runner, default_model, max_threads
):
    sample_config["samples"] = 5
    sample_config["method"] = "uniform"
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == 5
    assert cost == 0
    assert all(item in sample_data for item in results)


def test_sample_operation_with_fraction(
    sample_config, sample_data, runner, default_model, max_threads
):
    sample_config["samples"] = 0.5
    sample_config["method"] = "uniform"
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == len(sample_data) // 2
    assert cost == 0
    assert all(item in sample_data for item in results)


def test_sample_operation_with_list(
    sample_config, sample_data, runner, default_model, max_threads
):
    sample_list = [{"id": 1}, {"id": 3}, {"id": 5}]
    sample_config["samples"] = sample_list
    sample_config["method"] = "custom"
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == len(sample_list)
    assert cost == 0
    assert all(item["id"] in [1, 3, 5] for item in results)


def test_sample_operation_with_stratify(
    sample_config, sample_data, runner, default_model, max_threads
):
    sample_config["samples"] = 5
    sample_config["method"] = "stratify"
    sample_config["method_kwargs"] = {"stratify_key": "group"}
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == 5
    assert cost == 0
    assert all(item in sample_data for item in results)
    assert len(set(item["group"] for item in results)) > 1


def test_sample_operation_with_outliers(
    sample_config, sample_data, runner, default_model, max_threads
):
    sample_config["method"] = "outliers"
    sample_config["method_kwargs"] = {
        "std": 2,
        "embedding_keys": ["concept", "description"],
        "keep": True,
    }
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) < len(sample_data)
    assert cost > 0
    assert all(item in sample_data for item in results)


def test_sample_operation_empty_input(
    sample_config, runner, default_model, max_threads
):
    sample_config["samples"] = 3
    sample_config["method"] = "uniform"
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute([])

    assert len(results) == 0
    assert cost == 0


def test_sample_operation_with_outliers_and_center(
    sample_config, sample_data, runner, default_model, max_threads
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
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) < len(sample_data)
    assert cost > 0
    assert all(item in sample_data for item in results)


def test_sample_first_method(
    sample_config, sample_data, runner, default_model, max_threads
):
    """Test the 'first' sampling method."""
    sample_config["method"] = "first"
    sample_config["samples"] = 3
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)

    assert len(results) == 3
    assert cost == 0
    assert results == sample_data[:3]  # Should be the first 3 items


def test_sample_stratify_maintains_proportions(
    sample_config, runner, default_model, max_threads
):
    """Test that stratified sampling maintains group proportions."""
    # Create data with known proportions
    stratified_data = []
    for i in range(70):
        stratified_data.append({"id": i, "category": "A"})
    for i in range(30):
        stratified_data.append({"id": i + 70, "category": "B"})
    
    sample_config["method"] = "stratify"
    sample_config["samples"] = 0.2  # Sample 20%
    sample_config["method_kwargs"] = {"stratify_key": "category"}
    sample_config["random_state"] = 42
    
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(stratified_data)
    
    # Count categories in results
    category_counts = {"A": 0, "B": 0}
    for item in results:
        category_counts[item["category"]] += 1
    
    # Check proportions (should be roughly 70/30)
    total = len(results)
    assert abs(category_counts["A"] / total - 0.7) < 0.1  # Allow 10% tolerance
    assert abs(category_counts["B"] / total - 0.3) < 0.1


def test_sample_outliers_with_samples_param(
    sample_config, sample_data, runner, default_model, max_threads
):
    """Test outlier sampling using samples parameter instead of std."""
    sample_config["method"] = "outliers"
    sample_config["method_kwargs"] = {
        "samples": 3,  # Keep 3 closest items
        "embedding_keys": ["concept", "description"],
        "keep": False,  # Keep inliers (closest to center)
    }
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)
    
    assert len(results) == 3
    assert cost > 0
    assert all(item in sample_data for item in results)


def test_sample_custom_with_multiple_keys(
    sample_config, runner, default_model, max_threads
):
    """Test custom sampling with multiple matching keys."""
    # Create data with multiple keys
    multi_key_data = [
        {"id": 1, "type": "A", "status": "active"},
        {"id": 2, "type": "B", "status": "inactive"},
        {"id": 3, "type": "A", "status": "active"},
        {"id": 4, "type": "A", "status": "inactive"},
        {"id": 5, "type": "B", "status": "active"},
    ]
    
    sample_config["method"] = "custom"
    sample_config["samples"] = [
        {"type": "A", "status": "active"},  # Should match items 1 and 3
        {"id": 5},  # Should match item 5
    ]
    
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(multi_key_data)
    
    assert len(results) == 3
    assert cost == 0
    # Check that we got the right items
    result_ids = [r["id"] for r in results]
    assert set(result_ids) == {1, 3, 5}


def test_sample_random_state_reproducibility(
    sample_config, sample_data, runner, default_model, max_threads
):
    """Test that random_state produces reproducible results."""
    sample_config["method"] = "uniform"
    sample_config["samples"] = 5
    sample_config["random_state"] = 12345
    
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results1, _ = operation.execute(sample_data)
    results2, _ = operation.execute(sample_data)
    
    # Should get the same results with the same random state
    assert results1 == results2
    
    # Now without random state, results should differ (most of the time)
    sample_config.pop("random_state")
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results3, _ = operation.execute(sample_data)
    results4, _ = operation.execute(sample_data)
    # This might occasionally fail due to random chance, but very unlikely
    assert results3 != results4 or len(sample_data) <= 5


def test_sample_fraction_edge_cases(
    sample_config, sample_data, runner, default_model, max_threads
):
    """Test edge cases for fractional sampling."""
    # Test very small fraction
    sample_config["method"] = "uniform"
    sample_config["samples"] = 0.01  # 1%
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, _ = operation.execute(sample_data)
    assert len(results) >= 1  # Should have at least 1 item
    
    # Test fraction close to 1
    sample_config["samples"] = 0.99
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, _ = operation.execute(sample_data)
    assert len(results) < len(sample_data)  # Should not include everything


def test_sample_custom_no_matches(
    sample_config, sample_data, runner, default_model, max_threads
):
    """Test custom sampling when no items match."""
    sample_config["method"] = "custom"
    sample_config["samples"] = [
        {"id": 999},  # Non-existent ID
        {"id": 1000},
    ]
    
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(sample_data)
    
    assert len(results) == 0
    assert cost == 0


def test_sample_outliers_all_same_embedding(
    sample_config, runner, default_model, max_threads
):
    """Test outlier sampling when all items have identical embeddings."""
    # Create data where all items have the same text
    identical_data = [
        {"id": i, "text": "identical text"} for i in range(10)
    ]
    
    sample_config["method"] = "outliers"
    sample_config["method_kwargs"] = {
        "std": 1,
        "embedding_keys": ["text"],
        "keep": True,  # Keep outliers
    }
    
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(identical_data)
    
    # When all embeddings are identical, there are no outliers
    assert len(results) == 0
    assert cost > 0


def test_sample_retrieve_vector(
    sample_config, sample_data, runner, default_model, max_threads
):
    """Test retrieve_vector method."""
    from unittest.mock import MagicMock, patch
    
    sample_config["method"] = "retrieve_vector"
    sample_config["method_kwargs"] = {
        "query": "machine learning concepts",
        "embedding_keys": ["concept", "description"],
        "num_chunks": 3,
        "output_key": "similar_docs"
    }
    
    # Mock LanceDB
    with patch("docetl.operations.sample.lancedb") as mock_lancedb:
        with patch("docetl.operations.sample.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_lancedb.connect.return_value = mock_db
            mock_db.table_names.return_value = []
            mock_db.create_table.return_value = mock_table
            
            # Mock search results
            mock_search_results = [
                {"_index": 0, "_distance": 0.1},
                {"_index": 2, "_distance": 0.2},
                {"_index": 1, "_distance": 0.3}
            ]
            mock_table.search.return_value.limit.return_value.to_list.return_value = mock_search_results
            
            operation = SampleOperation(runner, sample_config, default_model, max_threads)
            results, cost = operation.execute(sample_data)
            
            # Check results
            assert len(results) == len(sample_data)
            assert cost > 0
            for result in results:
                assert "similar_docs" in result
                assert len(result["similar_docs"]) == 3
                assert all("_distance" in doc for doc in result["similar_docs"])


def test_sample_retrieve_fts(
    sample_config, sample_data, runner, default_model, max_threads
):
    """Test retrieve_fts method."""
    from unittest.mock import MagicMock, patch
    
    sample_config["method"] = "retrieve_fts"
    sample_config["method_kwargs"] = {
        "query": "clustering machine learning",
        "embedding_keys": ["concept", "description"],
        "num_chunks": 2,
        "rerank": True
    }
    
    # Mock LanceDB
    with patch("docetl.operations.sample.lancedb") as mock_lancedb:
        with patch("docetl.operations.sample.Path") as mock_path:
            mock_path.return_value.mkdir.return_value = None
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_lancedb.connect.return_value = mock_db
            mock_db.table_names.return_value = []
            mock_db.create_table.return_value = mock_table
            
            # Mock search results
            mock_search_results = [
                {"_index": 3, "_distance": 0.15},
                {"_index": 1, "_distance": 0.25}
            ]
            mock_table.search.return_value.limit.return_value.to_list.return_value = mock_search_results
            
            operation = SampleOperation(runner, sample_config, default_model, max_threads)
            results, cost = operation.execute(sample_data)
            
            # Check results
            assert len(results) == len(sample_data)
            assert cost > 0
            for result in results:
                assert "_retrieved" in result
                assert len(result["_retrieved"]) <= 2


def test_sample_stratify_compound_keys(
    sample_config, runner, default_model, max_threads
):
    """Test stratified sampling with compound keys."""
    # Create data with compound stratification
    compound_data = []
    for region in ["North", "South"]:
        for category in ["A", "B", "C"]:
            for i in range(10):
                compound_data.append({
                    "id": f"{region}_{category}_{i}",
                    "region": region,
                    "category": category
                })
    
    sample_config["method"] = "stratify"
    sample_config["samples"] = 3
    sample_config["method_kwargs"] = {
        "stratify_key": ["region", "category"],
        "samples_per_group": True  # 3 samples from each region-category combination
    }
    sample_config["random_state"] = 42
    
    operation = SampleOperation(runner, sample_config, default_model, max_threads)
    results, cost = operation.execute(compound_data)
    
    # Should have 3 samples from each of the 6 combinations (2 regions * 3 categories)
    assert len(results) == 18  # 6 groups * 3 samples per group
    assert cost == 0
    
    # Verify each group has exactly 3 samples
    from collections import Counter
    group_counts = Counter()
    for item in results:
        group_key = (item["region"], item["category"])
        group_counts[group_key] += 1
    
    for count in group_counts.values():
        assert count == 3
