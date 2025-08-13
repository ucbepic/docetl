"""Tests for TopK operation."""

import pytest
from docetl.operations.topk import TopKOperation
from tests.conftest import runner, default_model, max_threads


@pytest.fixture
def sample_documents():
    """Sample documents for testing topk retrieval."""
    return [
        {
            "id": 1,
            "title": "Introduction to Python",
            "content": "Python is a versatile programming language great for beginners.",
            "category": "programming",
        },
        {
            "id": 2,
            "title": "Machine Learning Fundamentals",
            "content": "Machine learning enables computers to learn from data without explicit programming.",
            "category": "ai",
        },
        {
            "id": 3,
            "title": "Web Development with JavaScript",
            "content": "JavaScript powers interactive web applications and modern frameworks.",
            "category": "programming",
        },
        {
            "id": 4,
            "title": "Deep Learning and Neural Networks",
            "content": "Deep learning uses artificial neural networks to solve complex problems.",
            "category": "ai",
        },
        {
            "id": 5,
            "title": "Database Management Systems",
            "content": "Databases organize and store data efficiently for applications.",
            "category": "programming",
        },
        {
            "id": 6,
            "title": "Natural Language Processing",
            "content": "NLP helps computers understand and generate human language.",
            "category": "ai",
        },
    ]


class TestTopKEmbedding:
    """Tests for TopK operation with embedding method."""

    def test_topk_embedding_basic(self, sample_documents, runner, default_model, max_threads):
        """Test basic topk embedding retrieval."""
        config = {
            "name": "test_topk_embedding",
            "type": "topk",
            "method": "embedding",
            "k": 2,
            "keys": ["title", "content"],
            "query": "artificial intelligence and deep learning",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        assert len(results) == 2
        assert cost > 0  # Embedding calls have cost
        
        # Should retrieve AI-related documents
        ai_ids = {2, 4, 6}
        result_ids = {doc["id"] for doc in results}
        assert result_ids.issubset(ai_ids)

    def test_topk_embedding_with_stratification(self, sample_documents, runner, default_model, max_threads):
        """Test topk embedding with stratification."""
        config = {
            "name": "test_topk_stratified",
            "type": "topk",
            "method": "embedding",
            "k": 2,  # Will get 2 from each category
            "keys": ["content"],
            "query": "modern development techniques",
            "stratify_key": "category",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should get results from both categories
        categories = {doc["category"] for doc in results}
        assert "programming" in categories
        assert "ai" in categories
        assert cost > 0

    def test_topk_embedding_percentage(self, sample_documents, runner, default_model, max_threads):
        """Test topk with percentage instead of fixed count."""
        config = {
            "name": "test_topk_percentage",
            "type": "topk",
            "method": "embedding",
            "k": 0.5,  # 50% of documents
            "keys": ["title"],
            "query": "programming",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return 3 documents (50% of 6)
        assert len(results) == 3
        assert cost > 0

    def test_topk_embedding_with_custom_model(self, sample_documents, runner, default_model, max_threads):
        """Test topk with custom embedding model."""
        config = {
            "name": "test_topk_custom_model",
            "type": "topk",
            "method": "embedding",
            "k": 2,
            "keys": ["title", "content"],
            "query": "data science",
            "embedding_model": "text-embedding-3-large",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        assert len(results) == 2
        assert cost > 0


class TestTopKFTS:
    """Tests for TopK operation with FTS method."""

    def test_topk_fts_basic(self, sample_documents, runner, default_model, max_threads):
        """Test basic topk FTS retrieval."""
        config = {
            "name": "test_topk_fts",
            "type": "topk",
            "method": "fts",
            "k": 2,
            "keys": ["title", "content"],
            "query": "machine learning neural networks",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        assert len(results) == 2
        assert cost == 0  # FTS doesn't use API calls
        
        # Should prioritize ML/DL documents
        ml_ids = {2, 4}
        result_ids = {doc["id"] for doc in results}
        assert len(result_ids.intersection(ml_ids)) >= 1

    def test_topk_fts_with_template(self, sample_documents, runner, default_model, max_threads):
        """Test topk FTS with Jinja template query."""
        docs = sample_documents.copy()
        docs[0]["search_query"] = "JavaScript web development"
        
        config = {
            "name": "test_topk_fts_template",
            "type": "topk",
            "method": "fts",
            "k": 2,
            "keys": ["title", "content"],
            "query": "{{ input.search_query }}",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs)
        
        assert len(results) == 2
        assert cost == 0
        
        # Should find JavaScript document
        result_ids = [doc["id"] for doc in results]
        assert 3 in result_ids

    def test_topk_fts_stratified(self, sample_documents, runner, default_model, max_threads):
        """Test topk FTS with stratification."""
        config = {
            "name": "test_topk_fts_stratified",
            "type": "topk",
            "method": "fts",
            "k": 1,  # Get 1 from each category
            "keys": ["content"],
            "query": "learning",
            "stratify_key": "category",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should get at least one from each category that has "learning"
        assert len(results) >= 2
        assert cost == 0
        categories = {doc["category"] for doc in results}
        assert len(categories) == 2

    def test_topk_fts_multiple_stratify_keys(self, sample_documents, runner, default_model, max_threads):
        """Test topk FTS with multiple stratification keys."""
        # Add subcategory for testing
        docs = sample_documents.copy()
        for i, doc in enumerate(docs):
            doc["subcategory"] = f"sub_{i % 2}"
        
        config = {
            "name": "test_topk_multi_stratified",
            "type": "topk",
            "method": "fts",
            "k": 1,
            "keys": ["title", "content"],
            "query": "programming",
            "stratify_key": ["category", "subcategory"],
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs)
        
        assert len(results) > 0
        assert cost == 0


class TestTopKValidation:
    """Tests for TopK operation validation."""

    def test_invalid_k_value(self, runner, default_model, max_threads):
        """Test that invalid k values raise errors."""
        config = {
            "name": "test_invalid_k",
            "type": "topk",
            "method": "embedding",
            "k": -1,  # Invalid negative value
            "keys": ["title"],
            "query": "test",
        }
        
        with pytest.raises(ValueError, match="'k' must be a positive number"):
            TopKOperation(runner, config, default_model, max_threads)

    def test_empty_keys(self, runner, default_model, max_threads):
        """Test that empty keys list raises error."""
        config = {
            "name": "test_empty_keys",
            "type": "topk",
            "method": "fts",
            "k": 5,
            "keys": [],  # Empty keys
            "query": "test",
        }
        
        with pytest.raises(ValueError, match="'keys' cannot be empty"):
            TopKOperation(runner, config, default_model, max_threads)

    def test_empty_query(self, runner, default_model, max_threads):
        """Test that empty query raises error."""
        config = {
            "name": "test_empty_query",
            "type": "topk",
            "method": "embedding",
            "k": 5,
            "keys": ["title"],
            "query": "",  # Empty query
        }
        
        with pytest.raises(ValueError, match="'query' cannot be empty"):
            TopKOperation(runner, config, default_model, max_threads)

    def test_invalid_stratify_key_type(self, runner, default_model, max_threads):
        """Test that invalid stratify_key type raises error."""
        from pydantic import ValidationError
        
        config = {
            "name": "test_invalid_stratify",
            "type": "topk",
            "method": "fts",
            "k": 5,
            "keys": ["title"],
            "query": "test",
            "stratify_key": 123,  # Invalid type
        }
        
        with pytest.raises(ValidationError):
            TopKOperation(runner, config, default_model, max_threads)


class TestTopKLLMCompare:
    """Tests for TopK operation with llm_compare method."""

    def test_topk_llm_compare_basic(self, sample_documents, runner, default_model, max_threads):
        """Test basic topk LLM comparison."""
        config = {
            "name": "test_topk_llm",
            "type": "topk",
            "method": "llm_compare",
            "k": 3,
            "keys": ["title", "content"],
            "query": "Rank by relevance to artificial intelligence and machine learning",
            "model": "gpt-4o-mini",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        assert len(results) <= 3
        assert cost > 0  # LLM calls have cost
        
        # Should prioritize AI-related documents
        ai_ids = {2, 4, 6}
        if len(results) > 0:
            result_ids = {doc["id"] for doc in results}
            # At least one should be AI-related
            assert len(result_ids.intersection(ai_ids)) >= 1

    def test_topk_llm_compare_percentage(self, sample_documents, runner, default_model, max_threads):
        """Test llm_compare with percentage k value."""
        config = {
            "name": "test_topk_llm_percentage",
            "type": "topk",
            "method": "llm_compare",
            "k": 0.5,  # 50% of documents
            "keys": ["title", "content"],
            "query": "Rank by technical complexity and implementation difficulty",
            "model": "gpt-4o-mini",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return 3 documents (50% of 6)
        assert len(results) <= 3
        assert cost > 0

    def test_topk_llm_compare_with_batch_size(self, sample_documents, runner, default_model, max_threads):
        """Test llm_compare with custom batch size."""
        config = {
            "name": "test_topk_llm_batch",
            "type": "topk",
            "method": "llm_compare",
            "k": 2,
            "keys": ["title"],
            "query": "Rank by innovation and cutting-edge technology",
            "model": "gpt-4o-mini",
            "batch_size": 3,  # Small batch size for testing
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        assert len(results) <= 2
        assert cost > 0

    def test_topk_llm_compare_no_stratification(self, sample_documents, runner, default_model, max_threads):
        """Test that llm_compare raises error with stratification."""
        config = {
            "name": "test_topk_llm_no_strat",
            "type": "topk",
            "method": "llm_compare",
            "k": 2,
            "keys": ["title"],
            "query": "Rank by importance",
            "model": "gpt-4o-mini",
            "stratify_key": "category",  # Should raise error
        }
        
        with pytest.raises(ValueError, match="stratify_key.*not supported.*llm_compare"):
            TopKOperation(runner, config, default_model, max_threads)

    def test_topk_llm_compare_missing_model(self, runner, default_model, max_threads):
        """Test that llm_compare requires model parameter."""
        config = {
            "name": "test_topk_llm_no_model",
            "type": "topk",
            "method": "llm_compare",
            "k": 2,
            "keys": ["title"],
            "query": "Rank by importance",
            # Missing model parameter
        }
        
        with pytest.raises(ValueError, match="'model' must be specified.*llm_compare"):
            TopKOperation(runner, config, default_model, max_threads)

    def test_topk_llm_compare_no_jinja(self, runner, default_model, max_threads):
        """Test that llm_compare doesn't allow Jinja templates."""
        config = {
            "name": "test_topk_llm_no_jinja",
            "type": "topk",
            "method": "llm_compare",
            "k": 2,
            "keys": ["title"],
            "query": "Rank by {{ input.criteria }}",  # Jinja template not allowed
            "model": "gpt-4o-mini",
        }
        
        with pytest.raises(ValueError, match="cannot contain Jinja templates.*llm_compare"):
            TopKOperation(runner, config, default_model, max_threads)


class TestTopKEdgeCases:
    """Tests for TopK operation edge cases."""

    def test_topk_empty_data(self, runner, default_model, max_threads):
        """Test topk with empty input data."""
        config = {
            "name": "test_empty_data",
            "type": "topk",
            "method": "fts",
            "k": 5,
            "keys": ["title"],
            "query": "test",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute([])
        
        assert len(results) == 0
        assert cost == 0

    def test_topk_k_exceeds_data(self, sample_documents, runner, default_model, max_threads):
        """Test when k exceeds available documents."""
        config = {
            "name": "test_k_exceeds",
            "type": "topk",
            "method": "fts",
            "k": 100,  # More than 6 documents
            "keys": ["title"],
            "query": "programming",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return all available documents
        assert len(results) == len(sample_documents)
        assert cost == 0

    def test_topk_missing_keys_in_documents(self, runner, default_model, max_threads):
        """Test behavior when specified keys don't exist in documents."""
        docs = [
            {"id": 1, "title": "Doc 1"},
            {"id": 2, "title": "Doc 2"},
            {"id": 3, "content": "Doc 3 content"},  # Missing title
        ]
        
        config = {
            "name": "test_missing_keys",
            "type": "topk",
            "method": "fts",
            "k": 2,
            "keys": ["title", "content"],
            "query": "doc",
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs)
        
        # Should handle missing keys gracefully
        assert len(results) == 2
        assert cost == 0

    def test_topk_embedding_model_ignored_for_fts(self, sample_documents, runner, default_model, max_threads, capsys):
        """Test that embedding_model is ignored for FTS method with warning."""
        config = {
            "name": "test_model_ignored",
            "type": "topk",
            "method": "fts",
            "k": 2,
            "keys": ["title"],
            "query": "test",
            "embedding_model": "some-other-model",  # Should be ignored
            "bypass_cache": True,
        }
        
        operation = TopKOperation(runner, config, default_model, max_threads)
        captured = capsys.readouterr()
        
        # Should see warning about ignored embedding_model
        assert "Warning" in captured.out
        assert "embedding_model" in captured.out
        
        # Should still work
        results, cost = operation.execute(sample_documents)
        assert len(results) == 2
        assert cost == 0