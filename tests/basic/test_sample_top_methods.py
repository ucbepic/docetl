"""Tests for top_embedding and top_fts sample methods."""

import pytest
from docetl.operations.sample import SampleOperation
from tests.conftest import runner, default_model, max_threads


@pytest.fixture
def sample_documents():
    """Sample documents with different topics for testing."""
    return [
        {
            "id": 1,
            "title": "Python Programming",
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "category": "programming",
            "subcategory": "languages",
        },
        {
            "id": 2,
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "category": "ai",
            "subcategory": "ml",
        },
        {
            "id": 3,
            "title": "JavaScript Frameworks",
            "content": "JavaScript frameworks like React and Vue help developers build modern web applications efficiently.",
            "category": "programming",
            "subcategory": "web",
        },
        {
            "id": 4,
            "title": "Deep Learning",
            "content": "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
            "category": "ai",
            "subcategory": "dl",
        },
        {
            "id": 5,
            "title": "Database Design",
            "content": "Good database design ensures efficient data storage and retrieval in applications.",
            "category": "programming",
            "subcategory": "databases",
        },
        {
            "id": 6,
            "title": "Natural Language Processing",
            "content": "NLP enables computers to understand, interpret, and generate human language.",
            "category": "ai",
            "subcategory": "nlp",
        },
        {
            "id": 7,
            "title": "Web Development",
            "content": "Web development involves creating websites and web applications using HTML, CSS, and JavaScript.",
            "category": "programming",
            "subcategory": "web",
        },
        {
            "id": 8,
            "title": "Computer Vision",
            "content": "Computer vision allows machines to interpret and understand visual information from images and videos.",
            "category": "ai",
            "subcategory": "cv",
        },
    ]


class TestTopEmbedding:
    """Tests for the top_embedding sampling method."""

    def test_top_embedding_basic(self, sample_documents, runner, default_model, max_threads):
        """Test basic top_embedding functionality without stratification."""
        config = {
            "name": "test_top_embedding",
            "type": "sample",
            "method": "top_embedding",
            "samples": 3,
            "method_kwargs": {
                "keys": ["title", "content"],
                "query": "artificial intelligence and machine learning",
                "embedding_model": "text-embedding-3-small",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return 3 items
        assert len(results) == 3
        assert cost > 0  # Embedding calls have a cost
        
        # Results should be AI-related documents (IDs 2, 4, 6, 8)
        ai_ids = {2, 4, 6, 8}
        result_ids = {doc["id"] for doc in results}
        assert result_ids.issubset(ai_ids), f"Expected AI-related docs, got IDs: {result_ids}"

    def test_top_embedding_with_template(self, sample_documents, runner, default_model, max_threads):
        """Test top_embedding with Jinja template in query."""
        # Modify first document to have a query field
        docs_with_query = sample_documents.copy()
        docs_with_query[0]["user_query"] = "web programming tutorials"
        
        config = {
            "name": "test_top_embedding_template",
            "type": "sample",
            "method": "top_embedding",
            "samples": 2,
            "method_kwargs": {
                "keys": ["title", "content"],
                "query": "{{ input.user_query }}",
                "embedding_model": "text-embedding-3-small",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs_with_query)
        
        assert len(results) == 2
        assert cost > 0
        
        # Should return web-related documents
        web_ids = {3, 7}  # JavaScript Frameworks and Web Development
        result_ids = {doc["id"] for doc in results}
        # At least one should be web-related
        assert len(result_ids.intersection(web_ids)) >= 1

    def test_top_embedding_with_stratification(self, sample_documents, runner, default_model, max_threads):
        """Test top_embedding with stratification by category."""
        config = {
            "name": "test_top_embedding_stratified",
            "type": "sample",
            "method": "top_embedding",
            "samples": 4,
            "stratify_key": "category",
            "samples_per_group": True,  # Sample from each group
            "method_kwargs": {
                "keys": ["title", "content"],
                "query": "advanced techniques and frameworks",
                "embedding_model": "text-embedding-3-small",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return samples from both categories
        assert len(results) > 0
        assert cost > 0
        
        # Check that we have results from both categories
        categories = {doc["category"] for doc in results}
        assert "programming" in categories
        assert "ai" in categories

    def test_top_embedding_float_samples(self, sample_documents, runner, default_model, max_threads):
        """Test top_embedding with float sample size (percentage)."""
        config = {
            "name": "test_top_embedding_float",
            "type": "sample",
            "method": "top_embedding",
            "samples": 0.5,  # 50% of documents
            "method_kwargs": {
                "keys": ["title"],
                "query": "programming",
                "embedding_model": "text-embedding-3-small",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return 4 documents (50% of 8)
        assert len(results) == 4
        assert cost > 0

    def test_top_embedding_empty_data(self, runner, default_model, max_threads):
        """Test top_embedding with empty input data."""
        config = {
            "name": "test_top_embedding_empty",
            "type": "sample",
            "method": "top_embedding",
            "samples": 3,
            "method_kwargs": {
                "keys": ["title"],
                "query": "test query",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute([])
        
        assert len(results) == 0
        assert cost == 0


class TestTopFTS:
    """Tests for the top_fts sampling method."""

    def test_top_fts_basic(self, sample_documents, runner, default_model, max_threads):
        """Test basic top_fts functionality without stratification."""
        config = {
            "name": "test_top_fts",
            "type": "sample",
            "method": "top_fts",
            "samples": 3,
            "method_kwargs": {
                "keys": ["title", "content"],
                "query": "machine learning neural networks",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return 3 items
        assert len(results) == 3
        assert cost == 0  # FTS doesn't use API calls
        
        # Should prioritize ML/DL documents
        ml_ids = {2, 4}  # Machine Learning Basics and Deep Learning
        result_ids = {doc["id"] for doc in results}
        # At least one should be ML-related
        assert len(result_ids.intersection(ml_ids)) >= 1

    def test_top_fts_text_normalization(self, runner, default_model, max_threads):
        """Test that FTS properly normalizes text (lowercase, special chars)."""
        docs = [
            {"id": 1, "text": "UPPERCASE TEXT WITH SPECIAL CHARS!!!"},
            {"id": 2, "text": "lowercase text with special chars..."},
            {"id": 3, "text": "MiXeD CaSe TeXt #$%^&*"},
            {"id": 4, "text": "normal text without issues"},
        ]
        
        config = {
            "name": "test_fts_normalization",
            "type": "sample",
            "method": "top_fts",
            "samples": 2,
            "method_kwargs": {
                "keys": ["text"],
                "query": "UPPERCASE text",  # Query with mixed case
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs)
        
        assert len(results) == 2
        assert cost == 0
        
        # Should match documents 1 and 2 (both have "text" and one has "uppercase")
        result_ids = {doc["id"] for doc in results}
        assert 1 in result_ids  # Has both "uppercase" and "text"

    def test_top_fts_with_template(self, sample_documents, runner, default_model, max_threads):
        """Test top_fts with Jinja template in query."""
        docs_with_query = sample_documents.copy()
        docs_with_query[0]["search_term"] = "JavaScript React Vue"
        
        config = {
            "name": "test_fts_template",
            "type": "sample",
            "method": "top_fts",
            "samples": 2,
            "method_kwargs": {
                "keys": ["title", "content"],
                "query": "{{ input.search_term }}",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs_with_query)
        
        assert len(results) == 2
        assert cost == 0
        
        # Should return JavaScript Frameworks (ID 3) as top result
        result_ids = [doc["id"] for doc in results]
        assert 3 in result_ids

    def test_top_fts_with_stratification(self, sample_documents, runner, default_model, max_threads):
        """Test top_fts with stratification by category."""
        config = {
            "name": "test_fts_stratified",
            "type": "sample",
            "method": "top_fts",
            "samples": 4,
            "stratify_key": "category",
            "samples_per_group": True,
            "method_kwargs": {
                "keys": ["title", "content"],
                "query": "learning data",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        assert len(results) > 0
        assert cost == 0
        
        # Check that we have results from both categories
        categories = {doc["category"] for doc in results}
        assert len(categories) == 2  # Should have both categories

    def test_top_fts_with_multiple_stratify_keys(self, sample_documents, runner, default_model, max_threads):
        """Test top_fts with multiple stratification keys."""
        config = {
            "name": "test_fts_multi_stratified",
            "type": "sample",
            "method": "top_fts",
            "samples": 4,
            "stratify_key": ["category", "subcategory"],
            "samples_per_group": True,
            "method_kwargs": {
                "keys": ["content"],
                "query": "development",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        assert len(results) > 0
        assert cost == 0

    def test_top_fts_ngrams(self, runner, default_model, max_threads):
        """Test BM25 ranking with phrase-like queries."""
        docs = [
            {"id": 1, "text": "machine learning is powerful"},
            {"id": 2, "text": "learning machine basics"},
            {"id": 3, "text": "powerful machines for learning"},
            {"id": 4, "text": "basic machine operation"},
        ]
        
        config = {
            "name": "test_fts_ngrams",
            "type": "sample",
            "method": "top_fts",
            "samples": 2,
            "method_kwargs": {
                "keys": ["text"],
                "query": "machine learning",  # BM25 will match both terms
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs)
        
        assert len(results) == 2
        assert cost == 0
        
        # Documents with both "machine" and "learning" should rank higher
        result_ids = {doc["id"] for doc in results}
        # Docs 1 and 2 have both terms
        assert 1 in result_ids or 2 in result_ids

    def test_top_fts_empty_documents(self, runner, default_model, max_threads):
        """Test top_fts with documents that have empty text."""
        docs = [
            {"id": 1, "text": ""},
            {"id": 2, "text": ""},
            {"id": 3, "text": "some content here"},
        ]
        
        config = {
            "name": "test_fts_empty_docs",
            "type": "sample",
            "method": "top_fts",
            "samples": 2,
            "method_kwargs": {
                "keys": ["text"],
                "query": "content",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs)
        
        # Should handle empty documents gracefully
        assert len(results) == 2
        assert cost == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_missing_keys_parameter(self, runner, default_model, max_threads):
        """Test that missing 'keys' parameter raises an error."""
        config = {
            "name": "test_missing_keys",
            "type": "sample",
            "method": "top_embedding",
            "samples": 3,
            "method_kwargs": {
                "query": "test query",
                # Missing 'keys'
            },
        }
        
        with pytest.raises(ValueError, match="'keys' must be specified"):
            SampleOperation(runner, config, default_model, max_threads)

    def test_missing_query_parameter(self, runner, default_model, max_threads):
        """Test that missing 'query' parameter raises an error."""
        config = {
            "name": "test_missing_query",
            "type": "sample",
            "method": "top_fts",
            "samples": 3,
            "method_kwargs": {
                "keys": ["title"],
                # Missing 'query'
            },
        }
        
        with pytest.raises(ValueError, match="'query' must be specified"):
            SampleOperation(runner, config, default_model, max_threads)

    def test_missing_samples_parameter(self, runner, default_model, max_threads):
        """Test that missing 'samples' parameter raises an error."""
        config = {
            "name": "test_missing_samples",
            "type": "sample",
            "method": "top_embedding",
            # Missing 'samples'
            "method_kwargs": {
                "keys": ["title"],
                "query": "test",
            },
        }
        
        with pytest.raises(ValueError, match="Must specify 'samples'"):
            SampleOperation(runner, config, default_model, max_threads)

    def test_samples_exceed_data_size(self, sample_documents, runner, default_model, max_threads):
        """Test requesting more samples than available data."""
        config = {
            "name": "test_exceed_samples",
            "type": "sample",
            "method": "top_fts",
            "samples": 100,  # More than 8 documents
            "method_kwargs": {
                "keys": ["title"],
                "query": "test",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(sample_documents)
        
        # Should return all available documents
        assert len(results) == len(sample_documents)
        assert cost == 0

    def test_nonexistent_keys_in_documents(self, runner, default_model, max_threads):
        """Test behavior when specified keys don't exist in documents."""
        docs = [
            {"id": 1, "title": "Document 1"},
            {"id": 2, "title": "Document 2"},
            {"id": 3, "title": "Document 3"},
        ]
        
        config = {
            "name": "test_nonexistent_keys",
            "type": "sample",
            "method": "top_fts",
            "samples": 2,
            "method_kwargs": {
                "keys": ["title", "nonexistent_field"],
                "query": "document",
            },
            "bypass_cache": True,
        }
        
        operation = SampleOperation(runner, config, default_model, max_threads)
        results, cost = operation.execute(docs)
        
        # Should handle missing keys gracefully
        assert len(results) == 2
        assert cost == 0