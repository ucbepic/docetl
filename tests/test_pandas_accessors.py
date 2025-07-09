"""Tests for pandas semantic accessors."""

import pytest
import pandas as pd
import numpy as np
from docetl import SemanticAccessor

@pytest.fixture
def sample_df():
    """Sample DataFrame with text data."""
    # Generate synthetic data with diverse topics and content
    categories = [
        "technology", "environment", "business", "health", "science", 
        "education", "politics", "sports", "entertainment", "culture"
    ]
    
    texts = [
        # Technology texts
        "New AI model achieves breakthrough in natural language processing.",
        "Cloud computing adoption continues to rise among enterprises.",
        "Quantum computing researchers make significant progress in error correction.",
        "5G networks transform mobile connectivity landscape.",
        "Cybersecurity threats evolve with rise of IoT devices.",
        "Blockchain technology revolutionizes supply chain management.",
        "Machine learning algorithms improve healthcare diagnostics.",
        "Edge computing enables real-time data processing.",
        "Virtual reality applications expand beyond gaming.",
        "Artificial intelligence enhances autonomous vehicle capabilities.",
        "Tech startups focus on sustainable innovation.",
        "Digital transformation accelerates in post-pandemic era.",
        "Robotics automation reshapes manufacturing processes.",
        "Data privacy concerns influence tech development.",
        "Software development practices embrace DevOps methodologies.",
        "Cloud security measures strengthen data protection.",
        "AI ethics guidelines shape technology development.",
        "Quantum supremacy demonstrates computational advantages.",
        "Tech industry addresses environmental impact.",
        "Neural networks advance pattern recognition capabilities.",
        
        # Environment texts
        "Global warming impacts arctic ice coverage.",
        "Renewable energy adoption reduces carbon emissions.",
        "Ocean pollution threatens marine ecosystems.",
        "Sustainable agriculture practices gain momentum.",
        "Deforestation rates alarm environmental scientists.",
        "Clean energy initiatives transform power generation.",
        "Biodiversity loss affects ecosystem stability.",
        "Climate change influences weather patterns.",
        "Conservation efforts protect endangered species.",
        "Green technology advances environmental protection.",
        "Circular economy reduces waste production.",
        "Air quality improvements show policy success.",
        "Water conservation methods address scarcity.",
        "Environmental regulations shape industry practices.",
        "Habitat restoration supports wildlife recovery.",
        "Carbon capture technology shows promise.",
        "Sustainable urban planning reduces environmental impact.",
        "Plastic pollution reduction efforts expand.",
        "Renewable resource management improves sustainability.",
        "Environmental monitoring systems advance.",
        
        # Business texts
        "Market volatility affects investment strategies.",
        "Digital payments transform financial services.",
        "Remote work policies reshape corporate culture.",
        "Supply chain optimization increases efficiency.",
        "Customer experience drives business innovation.",
        "Economic indicators suggest growth trends.",
        "Mergers and acquisitions reshape industry landscape.",
        "Start-up ecosystem attracts venture capital.",
        "Corporate sustainability initiatives expand.",
        "E-commerce growth continues post-pandemic.",
        "Business analytics drive decision making.",
        "Global trade patterns show recovery signs.",
        "Employee wellness programs gain importance.",
        "Digital marketing strategies evolve.",
        "Financial technology disrupts traditional banking.",
        "Business model innovation accelerates.",
        "Workplace diversity initiatives expand.",
        "Corporate governance standards strengthen.",
        "Small business adaptation shows resilience.",
        "International markets present opportunities.",
        
        # Health texts
        "Medical research advances treatment options.",
        "Public health measures show effectiveness.",
        "Mental health awareness increases globally.",
        "Telemedicine adoption transforms healthcare delivery.",
        "Vaccine development accelerates disease prevention.",
        "Wellness programs promote preventive care.",
        "Healthcare technology improves patient outcomes.",
        "Nutrition research reveals dietary impacts.",
        "Exercise benefits mental and physical health.",
        "Precision medicine personalizes treatment approaches.",
        "Healthcare access expands through technology.",
        "Chronic disease management improves.",
        "Medical device innovation continues.",
        "Health education programs show results.",
        "Preventive care reduces healthcare costs.",
        "Mental health support systems expand.",
        "Healthcare data analytics advance care.",
        "Patient care coordination improves.",
        "Health monitoring technology evolves.",
        "Medical training embraces simulation.",
        
        # Science texts
        "Space exploration reveals new discoveries.",
        "Genetic research advances understanding.",
        "Particle physics experiments yield results.",
        "Astronomical observations expand knowledge.",
        "Materials science enables new applications.",
        "Neuroscience research reveals brain function.",
        "Chemical engineering advances clean energy.",
        "Biological research improves medicine.",
        "Scientific collaboration accelerates discovery.",
        "Research methodology improves accuracy.",
        "Laboratory techniques advance capabilities.",
        "Scientific instruments increase precision.",
        "Research funding supports innovation.",
        "Scientific communication improves access.",
        "Experimental results validate theories.",
        "Research ethics guide scientific work.",
        "Scientific education expands access.",
        "Research infrastructure supports discovery.",
        "Scientific data management improves.",
        "Research impact measures evolve.",
        
        # Additional diverse texts
        "Educational technology transforms learning.",
        "Political developments influence policy.",
        "Sports analytics improve performance.",
        "Entertainment industry embraces streaming.",
        "Cultural exchange promotes understanding.",
        "Social media impacts communication patterns.",
        "Transportation systems improve efficiency.",
        "Urban development addresses sustainability.",
        "Agricultural innovation increases yields.",
        "Energy efficiency measures expand.",
        "Manufacturing processes optimize production.",
        "Retail transformation continues online.",
        "Tourism industry adapts to changes.",
        "Construction methods improve sustainability.",
        "Fashion industry embraces sustainability.",
        "Food industry addresses health concerns.",
        "Media landscape evolves digitally.",
        "Real estate market shows trends.",
        "Infrastructure development continues.",
        "Legal framework adapts to technology."
    ]
    
    # Multiply texts to get more than 100 records with variations
    expanded_texts = []
    expanded_categories = []
    expanded_ids = []
    
    for i, text in enumerate(texts):
        # Add original text
        expanded_texts.append(text)
        expanded_categories.append(categories[min(i // 10, len(categories)-1)])  # Prevent index out of range
        expanded_ids.append(i + 1)
        
        # Add variation with different wording
        variation = f"Recent reports indicate that {text.lower()}"
        expanded_texts.append(variation)
        expanded_categories.append(categories[min(i // 10, len(categories)-1)])  # Prevent index out of range
        expanded_ids.append(len(texts) + i + 1)
        
    return pd.DataFrame({
        "text": expanded_texts,
        "category": expanded_categories,
        "id": expanded_ids
    }).head(25)

@pytest.fixture
def sample_df2():
    """Second sample DataFrame for merge tests."""
    return pd.DataFrame({
        "description": [
            "Looking for Python developer with ML experience",
            "Environmental scientist needed for climate research",
            "Creative writer for children's stories"
        ],
        "department": ["tech", "science", "creative"],
        "job_id": [101, 102, 103]
    })

def test_semantic_map(sample_df):
    """Test semantic map operation."""
    result = sample_df.semantic.map(
        prompt="Extract key entities from: {{input.text}}",
        output_schema={"entities": "list[str]", "main_topic": "str"},
        model="gpt-4o-mini"
    )
    
    assert isinstance(result, pd.DataFrame)
    assert "entities" in result.columns
    assert "main_topic" in result.columns
    assert len(result) == len(sample_df)
    assert all(isinstance(x, list) for x in result["entities"])
    assert all(isinstance(x, str) for x in result["main_topic"])

def test_semantic_filter(sample_df):
    """Test semantic filter operation."""
    result = sample_df.semantic.filter(
        prompt="Determine if this text is about technology: {{input.text}}",
        model="gpt-4o-mini"
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(sample_df)  # Should filter some rows
    assert all(col in result.columns for col in sample_df.columns)  # All original columns preserved
    assert "keep" not in result.columns  # Filter column removed

def test_semantic_merge(sample_df, sample_df2):
    """Test semantic merge operation."""
    result = sample_df.semantic.merge(
        sample_df2,
        comparison_prompt="""Compare the text and job description:
        Text: {{ left.text }}
        Job: {{ right.description }}
        Are they related in topic/domain?""",
        fuzzy=True,
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_semantic_agg_simple(sample_df):
    """Test simple semantic aggregation without fuzzy matching."""
    result = sample_df.semantic.agg(
        reduce_prompt="""Summarize these texts:
        {% for item in inputs %}
        - {{item.text}}
        {% endfor %}""",
        reduce_keys="category",
        output_schema={"summary": "str", "key_points": "list[str]"}
    )
    
    assert isinstance(result, pd.DataFrame)
    assert "summary" in result.columns
    assert "key_points" in result.columns
    assert all(isinstance(x, str) for x in result["summary"])
    assert all(isinstance(x, list) for x in result["key_points"])

def test_semantic_agg_fuzzy_auto(sample_df):
    """Test fuzzy semantic aggregation with auto-synthesized comparison prompt."""
    result = sample_df.semantic.agg(
        reduce_prompt="""Summarize these texts:
        {% for item in inputs %}
        - {{item.text}}
        {% endfor %}""",
        reduce_keys=["category"],
        output_schema={"summary": "str", "key_points": "list[str]"},
        fuzzy=True  # Should auto-synthesize comparison prompt
    )
    
    assert isinstance(result, pd.DataFrame)
    assert "summary" in result.columns
    assert "key_points" in result.columns
    assert all(isinstance(x, str) for x in result["summary"])
    assert all(isinstance(x, list) for x in result["key_points"])

def test_semantic_agg_fuzzy_custom(sample_df):
    """Test fuzzy semantic aggregation with custom comparison and resolution."""
    result = sample_df.semantic.agg(
        # Reduction config
        reduce_prompt="""Summarize these texts:
        {% for item in inputs %}
        - {{item.text}}
        {% endfor %}""",
        reduce_keys="category",
        output_schema={"summary": "str", "key_points": "list[str]"},
        
        # Resolution config
        fuzzy=True,
        comparison_prompt="Are these categories similar: {{input1.category}} vs {{input2.category}}?",
        resolution_prompt="""Standardize the category name:
        {% for item in inputs %}
        - {{item.category}}
        {% endfor %}""",
        resolution_output_schema={"standardized_category": "str"}
    )
    
    assert isinstance(result, pd.DataFrame)
    assert "summary" in result.columns
    assert "key_points" in result.columns
    assert all(isinstance(x, str) for x in result["summary"])
    assert all(isinstance(x, list) for x in result["key_points"])

def test_semantic_agg_global(sample_df):
    """Test global semantic aggregation using _all."""
    result = sample_df.semantic.agg(
        reduce_prompt="""Summarize all texts:
        {% for item in inputs %}
        - {{item.text}}
        {% endfor %}""",
        reduce_keys=["_all"],  # Should skip resolution phase
        output_schema={"summary": "str", "key_points": "list[str]"}
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # Global aggregation should return one row
    assert "summary" in result.columns
    assert "key_points" in result.columns
    assert isinstance(result["summary"].iloc[0], str)
    assert isinstance(result["key_points"].iloc[0], list)

def test_cost_tracking(sample_df):
    """Test that cost tracking works across operations."""
    initial_cost = sample_df.semantic.total_cost
    
    # Run a map operation
    _ = sample_df.semantic.map(
        prompt="Count words in: {{input.text}}",
        output_schema={"word_count": "int"},
        model="gpt-4o-mini",
        bypass_cache=True
    )
    
    map_cost = sample_df.semantic.total_cost
    assert map_cost > initial_cost
    
    # Run a filter operation
    _ = sample_df.semantic.filter(
        prompt="Is this about technology? {{input.text}}",
        model="gpt-4o-mini",
        bypass_cache=True
    )
    
    final_cost = sample_df.semantic.total_cost
    assert final_cost > map_cost

def test_config_setting():
    """Test that config can be set and updated."""
    df = pd.DataFrame({"text": ["test"]})
    
    # Set custom config
    df.semantic.set_config(default_model="gpt-4o-mini", optimizer_config={"rewrite_agent_model": "gpt-4o", "judge_agent_model": "gpt-4o-mini"})
    assert df.semantic.runner.config["default_model"] == "gpt-4o-mini"
    
    # Update config
    df.semantic.set_config(max_threads=64)
    assert df.semantic.runner.config["max_threads"] == 64

def test_error_handling(sample_df):
    """Test error handling for invalid inputs."""
    # Test invalid schema
    with pytest.raises(ValueError):
        sample_df.semantic.map(
            prompt="test",
            output_schema={"invalid_type": "not_a_real_type"}
        )
    
    # Test invalid reduce keys
    with pytest.raises(ValueError):
        sample_df.semantic.agg(
            comparison_prompt="test",
            reduce_prompt="test",
            reduce_keys=123,  # Should be str or list
            output_schema={"summary": "str"}
        )
    
    # Test missing required args
    with pytest.raises(TypeError):
        sample_df.semantic.merge(
            sample_df,  # Missing comparison_prompt
            blocking_keys={"left": ["id"], "right": ["id"]}
        )

def test_operation_history(sample_df):
    """Test that operation history is tracked correctly."""
    # Run a map operation
    result = sample_df.semantic.map(
        prompt="Extract entities from: {{input.text}}",
        output_schema={"entities": "list[str]", "topic": "str"}
    )
    
    # Check history
    assert len(sample_df.semantic.history) == 1
    op = sample_df.semantic.history[0]
    assert op.op_type == "map"
    assert op.config["prompt"] == "Extract entities from: {{input.text}}"
    assert set(op.output_columns) == {"entities", "topic"}
    
    # Run a filter
    filtered = result.semantic.filter(
        prompt="Is this about tech? {{input.text}}"
    )
    assert len(result.semantic.history) == 2
    
def test_fuzzy_agg_with_context(sample_df):
    """Test that fuzzy aggregation uses context from previous operations."""
    # First create a derived column
    df_with_topic = sample_df.semantic.map(
        prompt="Extract the main topic: {{input.text}}",
        output_schema={"main_topic": "str"}
    )
    
    # Now do fuzzy aggregation using that column
    result = df_with_topic.semantic.agg(
        reduce_prompt="Summarize texts for topic: {% for item in inputs %}{{item.text}}{% endfor %}",
        reduce_keys=["main_topic"],
        output_schema={"summary": "str"},
        fuzzy=True  # Should auto-synthesize comparison with context
    )
    
    # Result should have resolve and reduce operations
    result_history = result.semantic.history
    assert len(result_history) == 3
    assert result_history[0].op_type == "map"
    assert result_history[1].op_type == "resolve"
    assert result_history[2].op_type == "reduce"
    assert "main_topic" in result.columns
    


def test_history_preservation(sample_df):
    """Test that history is preserved across operations."""
    # Run a map operation
    df1 = sample_df.semantic.map(
        prompt="Extract entities from: {{input.text}}",
        output_schema={"entities": "list[str]"}
    )
    assert len(df1.semantic.history) == 1
    
    # Run a filter operation on the result
    df2 = df1.semantic.filter(
        prompt="Is this about tech? {{input.text}}"
    )
    # Should have both map and filter in history
    assert len(df2.semantic.history) == 2
    assert df2.semantic.history[0].op_type == "map"
    assert df2.semantic.history[1].op_type == "filter"

def test_cost_preservation(sample_df):
    """Test that costs are preserved across operations."""
    # Run a map operation
    df1 = sample_df.semantic.map(
        prompt="Extract entities from: {{input.text}}",
        output_schema={"entities": "list[str]"},
        bypass_cache=True
    )
    map_cost = df1.semantic.total_cost
    assert map_cost > 0
    
    # Run a filter operation on the result
    df2 = df1.semantic.filter(
        prompt="Is this about tech? {{input.text}}",
        bypass_cache=True
    )
    # Total cost should include both operations
    assert df2.semantic.total_cost > map_cost

def test_context_preservation_in_agg(sample_df):
    """Test that operation context is preserved and used in aggregation."""
    # First create a derived column
    df1 = sample_df.semantic.map(
        prompt="Extract topic and sentiment: {{input.text}}",
        output_schema={"topic": "str", "sentiment": "str"}
    )
    
    # Then create another derived column
    df2 = df1.semantic.map(
        prompt="Rate confidence in topic (1-5): {{input.text}}",
        output_schema={"topic_confidence": "int"}
    )
    
    # Now aggregate with fuzzy matching
    result = df2.semantic.agg(
        reduce_prompt="Summarize sentiments by topic for these inputs: {{ inputs }}",
        reduce_keys=["topic"],
        output_schema={"summary": "str"},
        fuzzy=True  # Should auto-synthesize with context from both operations
    )
    
    # Check that context includes both previous operations
    resolve_config = result.semantic.history[-2].config  # Second to last operation is resolve
    prompt = resolve_config["comparison_prompt"]
    assert "topic' was created using this prompt: Extract topic and sentiment" in prompt


def test_semantic_split_token(sample_df):
    """Test semantic split operation with token count method."""
    result = sample_df.semantic.split(
        split_key="text",
        method="token_count",
        method_kwargs={"num_tokens": 10}
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= len(sample_df)  # Should create more rows
    assert "text_chunk" in result.columns
    assert f"semantic_split_0_id" in result.columns
    assert f"semantic_split_0_chunk_num" in result.columns
    
    # Check that all chunks have sequential numbering
    for doc_id in result[f"semantic_split_0_id"].unique():
        doc_chunks = result[result[f"semantic_split_0_id"] == doc_id]
        chunk_nums = sorted(doc_chunks[f"semantic_split_0_chunk_num"].tolist())
        assert chunk_nums == list(range(1, len(chunk_nums) + 1))


def test_semantic_split_delimiter():
    """Test semantic split operation with delimiter method."""
    df = pd.DataFrame({
        "content": [
            "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            "Another doc.\n\nWith multiple.\n\nParagraphs here."
        ],
        "id": [1, 2]
    })
    
    result = df.semantic.split(
        split_key="content",
        method="delimiter", 
        method_kwargs={"delimiter": "\n\n", "num_splits_to_group": 1}
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) > len(df)  # Should create more rows
    assert "content_chunk" in result.columns
    assert f"semantic_split_0_id" in result.columns
    assert f"semantic_split_0_chunk_num" in result.columns
    
    # Check that each chunk contains one paragraph
    for chunk in result["content_chunk"]:
        assert "\n\n" not in chunk  # Delimiter should be removed from chunks


def test_semantic_gather_basic():
    """Test semantic gather operation basic functionality."""
    # Create test data that simulates document chunks
    df = pd.DataFrame({
        "doc_id": ["doc1", "doc1", "doc1", "doc2", "doc2"],
        "chunk_num": [1, 2, 3, 1, 2],
        "content": [
            "First chunk of document 1",
            "Second chunk of document 1", 
            "Third chunk of document 1",
            "First chunk of document 2",
            "Second chunk of document 2"
        ]
    })
    
    result = df.semantic.gather(
        content_key="content",
        doc_id_key="doc_id",
        order_key="chunk_num"
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)  # Same number of rows
    assert "content_rendered" in result.columns
    
    # Check that rendered content includes the original content
    for i, row in result.iterrows():
        assert row["content"] in row["content_rendered"]


def test_semantic_gather_with_peripheral():
    """Test semantic gather operation with peripheral chunks configuration."""
    # Create test data with more chunks for context
    df = pd.DataFrame({
        "doc_id": ["doc1"] * 5,
        "chunk_num": [1, 2, 3, 4, 5],
        "content": [f"Chunk {i} content" for i in range(1, 6)]
    })
    
    result = df.semantic.gather(
        content_key="content",
        doc_id_key="doc_id",
        order_key="chunk_num",
        peripheral_chunks={
            "previous": {"head": {"count": 1}, "tail": {"count": 1}},
            "next": {"head": {"count": 1}, "tail": {"count": 1}}
        }
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)
    assert "content_rendered" in result.columns
    
    # Check that middle chunks have context from surrounding chunks
    middle_chunk = result[result["chunk_num"] == 3].iloc[0]
    rendered = middle_chunk["content_rendered"]
    
    # Should contain previous and next context markers
    assert "--- Previous Context ---" in rendered
    assert "--- Next Context ---" in rendered
    assert "--- Begin Main Chunk ---" in rendered
    assert "--- End Main Chunk ---" in rendered


def test_semantic_unnest_list():
    """Test semantic unnest operation with list values."""
    df = pd.DataFrame({
        "id": [1, 2],
        "tags": [["python", "pandas", "data"], ["ml", "ai"]]
    })
    
    result = df.semantic.unnest(unnest_key="tags")
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5  # 3 + 2 tags
    assert all(result.columns == ["id", "tags"])
    
    # Check that each tag becomes a separate row
    expected_tags = ["python", "pandas", "data", "ml", "ai"]
    actual_tags = result["tags"].tolist()
    assert set(actual_tags) == set(expected_tags)
    
    # Check that original data is preserved
    python_rows = result[result["tags"] == "python"]
    assert len(python_rows) == 1
    assert python_rows.iloc[0]["id"] == 1


def test_semantic_unnest_dict():
    """Test semantic unnest operation with dictionary values."""
    df = pd.DataFrame({
        "id": [1, 2],
        "user_info": [
            {"name": "Alice", "age": 30, "email": "alice@example.com"},
            {"name": "Bob", "age": 25, "email": "bob@example.com"}
        ]
    })
    
    result = df.semantic.unnest(
        unnest_key="user_info",
        expand_fields=["name", "age"]
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Same number of rows for dict unnesting
    assert "name" in result.columns
    assert "age" in result.columns
    assert "user_info" in result.columns  # Original dict preserved
    
    # Check expanded values
    alice_row = result[result["name"] == "Alice"].iloc[0]
    assert alice_row["age"] == 30
    assert alice_row["id"] == 1
    
    bob_row = result[result["name"] == "Bob"].iloc[0]
    assert bob_row["age"] == 25
    assert bob_row["id"] == 2


def test_semantic_unnest_recursive():
    """Test semantic unnest operation with recursive unnesting."""
    df = pd.DataFrame({
        "id": [1],
        "nested": [
            [{"values": [1, 2]}]
        ]
    })
    
    result = df.semantic.unnest(
        unnest_key="nested",
        recursive=True,
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # Should create one row per innermost value
    assert "values" in result.columns

    # Check that deeply nested values are unnested into individual rows
    expected_values = [1, 2]
    actual_values = []
    for val in result["values"]:
        if isinstance(val, list):
            actual_values.extend(val)
        else:
            actual_values.append(val)
    assert sorted(actual_values) == expected_values


def test_semantic_unnest_keep_empty():
    """Test semantic unnest operation with keep_empty option."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "tags": [["a", "b"], [], ["c"]]
    })
    
    result = df.semantic.unnest(
        unnest_key="tags",
        keep_empty=True
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4  # 2 + 1 (empty) + 1 tags
    
    # Check that empty list row is preserved
    empty_rows = result[result["tags"].isna()]
    assert len(empty_rows) == 1
    assert empty_rows.iloc[0]["id"] == 2


def test_new_operations_in_history():
    """Test that new operations are properly recorded in history."""
    df = pd.DataFrame({
        "content": ["Some text to split"],
        "tags": [["a", "b"]]
    })
    
    # Test split operation history
    split_result = df.semantic.split(
        split_key="content",
        method="token_count",
        method_kwargs={"num_tokens": 3}
    )
    
    assert len(split_result.semantic.history) == 1
    assert split_result.semantic.history[0].op_type == "split"
    assert "content_chunk" in split_result.semantic.history[0].output_columns
    
    # Test unnest operation history
    unnest_result = split_result.semantic.unnest(unnest_key="tags")
    
    assert len(unnest_result.semantic.history) == 2
    assert unnest_result.semantic.history[1].op_type == "unnest"
    
    # Test gather operation history (need appropriate data structure)
    gather_df = pd.DataFrame({
        "doc_id": ["doc1", "doc1"],
        "chunk_num": [1, 2],
        "content": ["chunk1", "chunk2"]
    })
    
    gather_result = gather_df.semantic.gather(
        content_key="content",
        doc_id_key="doc_id", 
        order_key="chunk_num"
    )
    
    assert len(gather_result.semantic.history) == 1
    assert gather_result.semantic.history[0].op_type == "gather"
    assert "content_rendered" in gather_result.semantic.history[0].output_columns


def test_chained_split_gather_workflow():
    """Test a realistic workflow combining split and gather operations."""
    # Start with a document
    df = pd.DataFrame({
        "document": [
            "This is the first paragraph of a long document. " * 5 + 
            "This is the second paragraph with different content. " * 5 +
            "This is the third and final paragraph. " * 5
        ],
        "doc_id": ["doc1"]
    })
    
    # Split the document into chunks
    split_result = df.semantic.split(
        split_key="document",
        method="token_count",
        method_kwargs={"num_tokens": 20}
    )
    
    # Gather context for each chunk
    gather_result = split_result.semantic.gather(
        content_key="document_chunk",
        doc_id_key="semantic_split_0_id",
        order_key="semantic_split_0_chunk_num",
        peripheral_chunks={
            "previous": {"head": {"count": 1}},
            "next": {"head": {"count": 1}}
        }
    )
    
    assert len(gather_result) >= len(df)  # Should have multiple chunks
    assert "document_chunk_rendered" in gather_result.columns
    assert len(gather_result.semantic.history) == 2
    assert gather_result.semantic.history[0].op_type == "split"
    assert gather_result.semantic.history[1].op_type == "gather" 
