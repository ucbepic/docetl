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
