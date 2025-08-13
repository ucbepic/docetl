import pytest
from unittest.mock import MagicMock, patch
from docetl.operations.retrieve_vector import RetrieveVectorOperation
from docetl.operations.retrieve_fts import RetrieveFTSOperation
from tests.conftest import FakeRunner


@pytest.fixture
def sample_documents():
    """Sample documents for retrieval tests."""
    return [
        {
            "id": 1,
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that focuses on training algorithms.",
            "category": "AI"
        },
        {
            "id": 2,
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning uses neural networks with multiple layers to process complex data.",
            "category": "AI"
        },
        {
            "id": 3,
            "title": "Natural Language Processing",
            "content": "NLP enables computers to understand and generate human language using machine learning.",
            "category": "AI"
        },
        {
            "id": 4,
            "title": "Computer Vision Applications",
            "content": "Computer vision allows machines to interpret and understand visual information from images.",
            "category": "AI"
        },
        {
            "id": 5,
            "title": "Data Science Best Practices",
            "content": "Effective data science requires proper data cleaning, analysis, and visualization techniques.",
            "category": "Data"
        }
    ]


@pytest.fixture
def retrieve_vector_config():
    """Configuration for retrieve_vector operation."""
    return {
        "name": "retrieve_similar",
        "type": "retrieve_vector",
        "query": "neural network architectures for image recognition",
        "embedding_keys": ["title", "content"],
        "num_chunks": 3,
        "embedding_model": "text-embedding-3-small"
    }


@pytest.fixture
def retrieve_fts_config():
    """Configuration for retrieve_fts operation."""
    return {
        "name": "search_docs",
        "type": "retrieve_fts",
        "query": "machine learning data science",
        "embedding_keys": ["title", "content"],
        "num_chunks": 3,
        "rerank": True
    }


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],  # Doc 1
        [0.2, 0.3, 0.4, 0.5],  # Doc 2
        [0.3, 0.4, 0.5, 0.6],  # Doc 3
        [0.4, 0.5, 0.6, 0.7],  # Doc 4
        [0.5, 0.6, 0.7, 0.8],  # Doc 5
        [0.15, 0.25, 0.35, 0.45]  # Query embedding
    ]


@patch("docetl.operations.retrieve_vector.Path")
@patch("docetl.operations.retrieve_vector.lancedb")
def test_retrieve_vector_basic(
    mock_lancedb, mock_path, retrieve_vector_config, sample_documents, 
    runner, default_model, max_threads, mock_embeddings
):
    """Test basic retrieve_vector functionality."""
    # Setup mocks
    mock_path.return_value.mkdir.return_value = None
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_lancedb.connect.return_value = mock_db
    mock_db.table_names.return_value = []
    mock_db.create_table.return_value = mock_table
    
    # Mock search results
    mock_search_results = [
        {"_index": 1, "_distance": 0.1},
        {"_index": 2, "_distance": 0.2},
        {"_index": 0, "_distance": 0.3}
    ]
    mock_table.search.return_value.limit.return_value.to_list.return_value = mock_search_results
    
    # Mock embeddings
    with patch("docetl.operations.clustering_utils.get_embeddings_for_clustering") as mock_get_embeddings:
        mock_get_embeddings.side_effect = [
            (mock_embeddings[:-1], 0.1),  # Document embeddings
            ([mock_embeddings[-1]], 0.01)  # Query embedding
        ]
        
        operation = RetrieveVectorOperation(
            runner, retrieve_vector_config, default_model, max_threads
        )
        results, cost = operation.execute(sample_documents)
    
    # Verify results
    assert len(results) == len(sample_documents)
    assert cost > 0
    
    # Check that each result has retrieved documents
    for result in results:
        assert "_retrieved" in result
        assert len(result["_retrieved"]) == 3
        assert all("_distance" in doc for doc in result["_retrieved"])


@patch("docetl.operations.retrieve_vector.Path")
@patch("docetl.operations.retrieve_vector.lancedb")
def test_retrieve_vector_persist(
    mock_lancedb, mock_path, retrieve_vector_config, sample_documents,
    runner, default_model, max_threads, mock_embeddings
):
    """Test retrieve_vector with persistence enabled."""
    retrieve_vector_config["persist"] = True
    retrieve_vector_config["table_name"] = "test_persist"
    
    # Setup mocks
    mock_path.return_value.mkdir.return_value = None
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_lancedb.connect.return_value = mock_db
    
    # Simulate existing table
    mock_db.table_names.return_value = ["test_persist"]
    mock_db.open_table.return_value = mock_table
    
    # Mock search results
    mock_search_results = [{"_index": 0, "_distance": 0.1}]
    mock_table.search.return_value.limit.return_value.to_list.return_value = mock_search_results
    
    with patch("docetl.operations.clustering_utils.get_embeddings_for_clustering") as mock_get_embeddings:
        mock_get_embeddings.side_effect = [
            (mock_embeddings[:-1], 0.1),
            ([mock_embeddings[-1]], 0.01)
        ]
        
        operation = RetrieveVectorOperation(
            runner, retrieve_vector_config, default_model, max_threads
        )
        results, cost = operation.execute(sample_documents)
    
    # Verify table.add was called instead of create_table
    mock_table.add.assert_called_once()
    mock_db.create_table.assert_not_called()


@patch("docetl.operations.retrieve_fts.Path")
@patch("docetl.operations.retrieve_fts.lancedb")
def test_retrieve_fts_with_rerank(
    mock_lancedb, mock_path, retrieve_fts_config, sample_documents,
    runner, default_model, max_threads, mock_embeddings
):
    """Test retrieve_fts with semantic reranking enabled."""
    # Setup mocks
    mock_path.return_value.mkdir.return_value = None
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_lancedb.connect.return_value = mock_db
    mock_db.table_names.return_value = []
    mock_db.create_table.return_value = mock_table
    
    # Mock search results
    mock_search_results = [
        {"_index": 0, "_distance": 0.1},
        {"_index": 4, "_distance": 0.2}
    ]
    mock_table.search.return_value.limit.return_value.to_list.return_value = mock_search_results
    
    with patch("docetl.operations.clustering_utils.get_embeddings_for_clustering") as mock_get_embeddings:
        mock_get_embeddings.side_effect = [
            (mock_embeddings[:-1], 0.1),  # Document embeddings
            ([mock_embeddings[-1]], 0.01)  # Query embedding
        ]
        
        operation = RetrieveFTSOperation(
            runner, retrieve_fts_config, default_model, max_threads
        )
        results, cost = operation.execute(sample_documents)
    
    # Verify results
    assert len(results) == len(sample_documents)
    assert cost > 0
    
    # Check retrieved documents
    for result in results:
        assert "_retrieved" in result
        assert len(result["_retrieved"]) <= retrieve_fts_config["num_chunks"]


@patch("docetl.operations.retrieve_fts.Path")
@patch("docetl.operations.retrieve_fts.lancedb")
def test_retrieve_fts_without_rerank(
    mock_lancedb, mock_path, retrieve_fts_config, sample_documents,
    runner, default_model, max_threads
):
    """Test retrieve_fts with pure text search (no reranking)."""
    retrieve_fts_config["rerank"] = False
    
    # Setup mocks
    mock_path.return_value.mkdir.return_value = None
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_lancedb.connect.return_value = mock_db
    mock_db.table_names.return_value = []
    mock_db.create_table.return_value = mock_table
    
    # Mock text search results
    mock_search_results = [
        {"_index": 0, "_score": 0.9},
        {"_index": 4, "_score": 0.7}
    ]
    mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = mock_search_results
    
    operation = RetrieveFTSOperation(
        runner, retrieve_fts_config, default_model, max_threads
    )
    results, cost = operation.execute(sample_documents)
    
    # Verify no embeddings were generated
    assert cost == 0  # No embedding cost
    assert len(results) == len(sample_documents)


def test_retrieve_vector_empty_input(
    retrieve_vector_config, runner, default_model, max_threads
):
    """Test retrieve_vector with empty input."""
    operation = RetrieveVectorOperation(
        runner, retrieve_vector_config, default_model, max_threads
    )
    results, cost = operation.execute([])
    
    assert results == []
    assert cost == 0


def test_retrieve_fts_empty_input(
    retrieve_fts_config, runner, default_model, max_threads
):
    """Test retrieve_fts with empty input."""
    operation = RetrieveFTSOperation(
        runner, retrieve_fts_config, default_model, max_threads
    )
    results, cost = operation.execute([])
    
    assert results == []
    assert cost == 0


@patch("docetl.operations.retrieve_vector.Path")
@patch("docetl.operations.retrieve_vector.lancedb")
def test_retrieve_vector_custom_output_key(
    mock_lancedb, mock_path, retrieve_vector_config, sample_documents,
    runner, default_model, max_threads, mock_embeddings
):
    """Test retrieve_vector with custom output key."""
    retrieve_vector_config["output_key"] = "similar_docs"
    
    # Setup mocks
    mock_path.return_value.mkdir.return_value = None
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_lancedb.connect.return_value = mock_db
    mock_db.table_names.return_value = []
    mock_db.create_table.return_value = mock_table
    
    mock_search_results = [{"_index": 0, "_distance": 0.1}]
    mock_table.search.return_value.limit.return_value.to_list.return_value = mock_search_results
    
    with patch("docetl.operations.clustering_utils.get_embeddings_for_clustering") as mock_get_embeddings:
        mock_get_embeddings.side_effect = [
            (mock_embeddings[:-1], 0.1),
            ([mock_embeddings[-1]], 0.01)
        ]
        
        operation = RetrieveVectorOperation(
            runner, retrieve_vector_config, default_model, max_threads
        )
        results, cost = operation.execute(sample_documents)
    
    # Check custom output key
    for result in results:
        assert "similar_docs" in result
        assert "_retrieved" not in result


def test_retrieve_vector_missing_embedding_keys(runner, default_model, max_threads):
    """Test retrieve_vector with missing embedding_keys."""
    config = {
        "name": "test",
        "type": "retrieve_vector",
        "query": "test query"
        # Missing embedding_keys
    }
    
    with pytest.raises(ValueError):
        RetrieveVectorOperation(runner, config, default_model, max_threads)


def test_retrieve_fts_missing_query(runner, default_model, max_threads):
    """Test retrieve_fts with missing query."""
    config = {
        "name": "test",
        "type": "retrieve_fts",
        "embedding_keys": ["content"]
        # Missing query
    }
    
    with pytest.raises(ValueError):
        RetrieveFTSOperation(runner, config, default_model, max_threads)