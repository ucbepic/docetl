import pytest
import json
import tempfile
import os
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

from docetl.api import (
    Pipeline,
    Dataset,
    MapOp,
    ReduceOp,
    FilterOp,
    ResolveOp,
    PipelineStep,
    PipelineOutput,
)
from docetl.operations.utils.validation import (
    is_pydantic_model,
    convert_schema_to_dict_format,
    pydantic_to_openapi_schema,
)
from docetl.operations.utils.api import OutputMode
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# PYDANTIC MODEL FIXTURES
# =============================================================================

class SentimentSchema(BaseModel):
    """Schema for sentiment analysis results"""
    sentiment: str = Field(description="The sentiment (positive, negative, neutral)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class CompanyInfo(BaseModel):
    """Schema for company information extraction"""
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    employees: Optional[int] = Field(None, ge=0, description="Number of employees")


class FilterResult(BaseModel):
    """Schema for filtering results (must have exactly one boolean field)"""
    is_positive: bool = Field(description="Whether the sentiment is positive")


class ReduceResult(BaseModel):
    """Schema for reduce operation results"""
    total_count: int = Field(description="Total number of items")
    avg_confidence: float = Field(description="Average confidence score")


class PersonInfo(BaseModel):
    """Schema for person information (for resolve operations)"""
    name: str = Field(description="Person's full name")
    email: str = Field(description="Email address")


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def pydantic_test_data():
    return [
        {"text": "Apple Inc. is a technology company with over 160,000 employees."},
        {"text": "Microsoft Corporation develops software and has around 220,000 employees."},
        {"text": "Tesla is an electric vehicle company with approximately 120,000 employees."},
    ]


@pytest.fixture
def sentiment_test_data():
    return [
        {"text": "I love this product!", "group": "A"},
        {"text": "This is terrible and disappointing.", "group": "B"},
        {"text": "It's okay, nothing special.", "group": "A"},
    ]


@pytest.fixture
def person_test_data():
    return [
        {"name": "John Doe", "email": "john@example.com"},
        {"name": "Jane Smith", "email": "jane@example.com"},
        {"name": "John D.", "email": "johndoe@example.com"},  # Similar to first entry
    ]


@pytest.fixture
def temp_input_file_from_data(request):
    """Create a temporary file from provided data"""
    data = request.param
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(data, tmp)
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def temp_output_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        pass
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def temp_intermediate_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


# =============================================================================
# UNIT TESTS FOR PYDANTIC UTILITIES
# =============================================================================

def test_pydantic_model_detection():
    """Test that we correctly detect Pydantic models"""
    # Should detect Pydantic models
    assert is_pydantic_model(SentimentSchema)
    assert is_pydantic_model(CompanyInfo)
    assert is_pydantic_model(FilterResult)

    # Should not detect regular dicts or other types
    assert not is_pydantic_model({"name": "str"})
    assert not is_pydantic_model("string")
    assert not is_pydantic_model(42)
    assert not is_pydantic_model(list)


def test_pydantic_to_openapi_conversion():
    """Test conversion of Pydantic models to OpenAPI schema"""
    schema = pydantic_to_openapi_schema(SentimentSchema)

    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "sentiment" in schema["properties"]
    assert "confidence" in schema["properties"]

    # Check that field descriptions are preserved
    assert schema["properties"]["sentiment"]["description"] == "The sentiment (positive, negative, neutral)"
    assert schema["properties"]["confidence"]["description"] == "Confidence score"

    # Check type mapping
    assert schema["properties"]["sentiment"]["type"] == "string"
    assert schema["properties"]["confidence"]["type"] == "number"

    # Check constraints
    assert schema["properties"]["confidence"]["minimum"] == 0.0
    assert schema["properties"]["confidence"]["maximum"] == 1.0


def test_pydantic_to_dict_conversion():
    """Test conversion of Pydantic models to DocETL dict format"""
    dict_schema = convert_schema_to_dict_format(CompanyInfo)

    assert isinstance(dict_schema, dict)
    assert "name" in dict_schema
    assert "industry" in dict_schema
    assert "employees" in dict_schema

    # Check type mapping
    assert dict_schema["name"] == "str"
    assert dict_schema["industry"] == "str"
    assert dict_schema["employees"] == "int"


def test_dict_schema_passthrough():
    """Test that regular dict schemas pass through unchanged"""
    original_schema = {"sentiment": "str", "confidence": "float"}
    converted_schema = convert_schema_to_dict_format(original_schema)

    assert converted_schema == original_schema


# =============================================================================
# INTEGRATION TESTS WITH OPERATIONS
# =============================================================================

@pytest.mark.parametrize("temp_input_file_from_data", [
    [{"text": "This is a great product!"}]
], indirect=True)
def test_map_operation_with_pydantic_schema(temp_input_file_from_data, temp_output_file, temp_intermediate_dir):
    """Test MapOp with Pydantic schema"""
    map_op = MapOp(
        name="sentiment_analysis",
        type="map",
        prompt="Analyze the sentiment of: '{{ input.text }}'. Return the sentiment and a confidence score.",
        output={"schema": SentimentSchema},
        model="gpt-4o-mini",
    )

    pipeline = Pipeline(
        name="pydantic_map_test",
        datasets={"test_input": Dataset(type="file", path=temp_input_file_from_data)},
        operations=[map_op],
        steps=[
            PipelineStep(name="map_step", input="test_input", operations=["sentiment_analysis"])
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    # Test that pipeline can be created and converted to dict
    config = pipeline._to_dict()
    assert "operations" in config
    assert len(config["operations"]) == 1

    # The operation should store the Pydantic model
    op_config = config["operations"][0]
    assert "output" in op_config
    assert "schema" in op_config["output"]
    assert op_config["output"]["schema"] == SentimentSchema


@pytest.mark.parametrize("temp_input_file_from_data", [
    [{"text": "I love this!", "group": "positive"}, {"text": "This is bad.", "group": "negative"}]
], indirect=True)
def test_reduce_operation_with_pydantic_schema(temp_input_file_from_data, temp_output_file, temp_intermediate_dir):
    """Test ReduceOp with Pydantic schema"""
    reduce_op = ReduceOp(
        name="sentiment_summary",
        type="reduce",
        reduce_key="group",
        prompt="Summarize the sentiment data: {{ inputs }}. Count total items and calculate average confidence.",
        output={"schema": ReduceResult},
        model="gpt-4o-mini",
    )

    pipeline = Pipeline(
        name="pydantic_reduce_test",
        datasets={"test_input": Dataset(type="file", path=temp_input_file_from_data)},
        operations=[reduce_op],
        steps=[
            PipelineStep(name="reduce_step", input="test_input", operations=["sentiment_summary"])
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    # Test that pipeline creation works
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.operations) == 1


@pytest.mark.parametrize("temp_input_file_from_data", [
    [{"text": "This is amazing!"}, {"text": "This is terrible!"}]
], indirect=True)
def test_filter_operation_with_pydantic_schema(temp_input_file_from_data, temp_output_file, temp_intermediate_dir):
    """Test FilterOp with Pydantic schema"""
    filter_op = FilterOp(
        name="positive_filter",
        type="filter",
        prompt="Is this text positive? '{{ input.text }}'",
        output={"schema": FilterResult},
        model="gpt-4o-mini",
    )

    pipeline = Pipeline(
        name="pydantic_filter_test",
        datasets={"test_input": Dataset(type="file", path=temp_input_file_from_data)},
        operations=[filter_op],
        steps=[
            PipelineStep(name="filter_step", input="test_input", operations=["positive_filter"])
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    # Test that pipeline creation works (the validation should pass)
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.operations) == 1


@pytest.mark.parametrize("temp_input_file_from_data", [
    [{"name": "John Doe", "email": "john@example.com"}, {"name": "Jane Smith", "email": "jane@example.com"}]
], indirect=True)
def test_resolve_operation_with_pydantic_schema(temp_input_file_from_data, temp_output_file, temp_intermediate_dir):
    """Test ResolveOp with Pydantic schema"""
    resolve_op = ResolveOp(
        name="person_resolver",
        type="resolve",
        blocking_keys=["name"],
        blocking_threshold=0.8,
        comparison_prompt="Are these the same person? Person 1: {{ input1 }} Person 2: {{ input2 }}",
        output={"schema": PersonInfo},
        embedding_model="text-embedding-3-small",
        comparison_model="gpt-4o-mini",
        resolution_model="gpt-4o-mini",
        resolution_prompt="Resolve these similar entries: {{ inputs }}",
    )

    pipeline = Pipeline(
        name="pydantic_resolve_test",
        datasets={"test_input": Dataset(type="file", path=temp_input_file_from_data)},
        operations=[resolve_op],
        steps=[
            PipelineStep(name="resolve_step", input="test_input", operations=["person_resolver"])
        ],
        output=PipelineOutput(
            type="file", path=temp_output_file, intermediate_dir=temp_intermediate_dir
        ),
        default_model="gpt-4o-mini",
    )

    # Test that pipeline creation works
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.operations) == 1


# =============================================================================
# STRUCTURED OUTPUT MODE TESTS
# =============================================================================

def test_structured_output_mode_auto_enablement():
    """Test that Pydantic schemas automatically enable STRUCTURED_OUTPUT mode"""
    from docetl.operations.utils.validation import is_pydantic_model

    # Test the logic that would be in the API wrapper
    def simulate_api_wrapper_logic(schema):
        op_config = {}
        if is_pydantic_model(schema):
            op_config = op_config.copy()
            if "output" not in op_config:
                op_config["output"] = {}
            if "mode" not in op_config["output"]:
                op_config["output"]["mode"] = OutputMode.STRUCTURED_OUTPUT.value
        return op_config

    # Test with Pydantic schema
    pydantic_config = simulate_api_wrapper_logic(SentimentSchema)
    assert pydantic_config.get("output", {}).get("mode") == OutputMode.STRUCTURED_OUTPUT.value

    # Test with dict schema
    dict_config = simulate_api_wrapper_logic({"sentiment": "str"})
    assert dict_config.get("output", {}).get("mode") != OutputMode.STRUCTURED_OUTPUT.value


def test_explicit_mode_preservation():
    """Test that explicitly set output modes are not overridden"""
    from docetl.operations.utils.validation import is_pydantic_model

    def simulate_api_wrapper_logic_with_existing_config(schema, existing_config):
        op_config = existing_config.copy()
        if is_pydantic_model(schema):
            if "output" not in op_config:
                op_config["output"] = {}
            if "mode" not in op_config["output"]:
                op_config["output"]["mode"] = OutputMode.STRUCTURED_OUTPUT.value
        return op_config

    # Test that existing TOOLS mode is preserved
    existing_config = {"output": {"mode": OutputMode.TOOLS.value}}
    result_config = simulate_api_wrapper_logic_with_existing_config(SentimentSchema, existing_config)
    assert result_config["output"]["mode"] == OutputMode.TOOLS.value


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_invalid_filter_schema_validation():
    """Test that FilterOp validates Pydantic schemas correctly"""
    # This should work - FilterResult has exactly one boolean field
    filter_op = FilterOp(
        name="valid_filter",
        type="filter",
        prompt="Test prompt",
        output={"schema": FilterResult},
        model="gpt-4o-mini",
    )
    assert filter_op.name == "valid_filter"

    # This should fail - SentimentSchema doesn't have a boolean field
    with pytest.raises((ValueError, TypeError)):
        FilterOp(
            name="invalid_filter",
            type="filter",
            prompt="Test prompt",
            output={"schema": SentimentSchema},  # This should fail validation
            model="gpt-4o-mini",
        )


def test_mixed_schema_types():
    """Test that we can mix dict and Pydantic schemas in the same pipeline"""
    map_op_pydantic = MapOp(
        name="pydantic_map",
        type="map",
        prompt="Extract sentiment: {{ input.text }}",
        output={"schema": SentimentSchema},
        model="gpt-4o-mini",
    )

    map_op_dict = MapOp(
        name="dict_map",
        type="map",
        prompt="Count words: {{ input.text }}",
        output={"schema": {"word_count": "int"}},
        model="gpt-4o-mini",
    )

    # Both should be valid
    assert map_op_pydantic.name == "pydantic_map"
    assert map_op_dict.name == "dict_map"


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

def test_backward_compatibility_with_dict_schemas():
    """Test that existing dict schemas continue to work unchanged"""
    traditional_config = {
        "sentiment": "string",
        "confidence": "float",
        "word_count": "integer"
    }

    # Should pass through unchanged
    converted = convert_schema_to_dict_format(traditional_config)
    assert converted == traditional_config

    # Should work in operations
    map_op = MapOp(
        name="traditional_map",
        type="map",
        prompt="Analyze: {{ input.text }}",
        output={"schema": traditional_config},
        model="gpt-4o-mini",
    )

    assert map_op.name == "traditional_map"


def test_reduce_pydantic_format():
    """Test that reduce operation works with Pydantic schemas in proper format"""
    # Test the correct format: {"schema": PydanticModel}
    reduce_op = ReduceOp(
        name="pydantic_reduce",
        type="reduce",
        reduce_key="group",
        prompt="Summarize these items: {{ inputs }}",  # Valid prompt with inputs variable
        output={"schema": ReduceResult},
        model="gpt-4o-mini",
    )
    assert reduce_op.name == "pydantic_reduce"