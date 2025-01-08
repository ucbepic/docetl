'''
import pytest
from unittest.mock import Mock, patch, MagicMock
from docetl.operations.utils.oss_llm import OSSLLMOperation
from docetl.operations.utils.llm import LLMResult
import json
@pytest.fixture
def mock_runner():
    return Mock()

@pytest.fixture
def sample_config():
    return {
        "name": "test_oss_operation",
        "type": "oss_llm",
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "output_schema": {
            "first_name": "str",
            "last_name": "str"
        },
        "max_tokens": 4096
    }

@pytest.fixture
def research_config():
    return {
        "name": "research_analyzer",
        "type": "oss_llm",
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "output_schema": {
            "title": "str",
            "authors": "list",
            "methodology": "str",
            "findings": "list",
            "limitations": "list",
            "future_work": "list"
        },
        "max_tokens": 4096
    }

@pytest.fixture
def mock_research_output():
    research_data = {
        "title": "Deep Learning in Natural Language Processing",
        "authors": ["John Smith", "Jane Doe"],
        "methodology": "Comparative analysis of transformer architectures",
        "findings": [
            "Improved accuracy by 15%",
            "Reduced training time by 30%"
        ],
        "limitations": [
            "Limited dataset size",
            "Computational constraints"
        ],
        "future_work": [
            "Extend to multilingual models",
            "Optimize for edge devices"
        ]
    }
    
    class MockOutput:
        def model_dump(self):
            return research_data
    return MockOutput()

def test_process_messages(sample_config, mock_runner):
    mock_model = MagicMock()
    
    class MockOutput:
        def model_dump(self):
            return {
                "first_name": "John",
                "last_name": "Doe"
            }
    
    mock_processor = Mock(return_value=MockOutput())
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        operation = OSSLLMOperation(sample_config, runner=mock_runner)
        messages = [
            {"role": "user", "content": "Extract information about John Doe"}
        ]
        
        result = operation.process_messages(messages)
        
        assert isinstance(result, LLMResult)
        assert result.total_cost == 0.0
        assert result.validated == True
        assert "tool_calls" in result.response["choices"][0]["message"]
        
        # Verify the tool call contains correct data
        tool_call = result.response["choices"][0]["message"]["tool_calls"][0]
        output_data = json.loads(tool_call["function"]["arguments"])
        assert output_data["first_name"] == "John"
        assert output_data["last_name"] == "Doe"

def test_research_paper_analysis(research_config, mock_research_output, mock_runner):
    mock_model = MagicMock()
    mock_processor = Mock(return_value=mock_research_output)
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        operation = OSSLLMOperation(research_config, runner=mock_runner)
        messages = [{
            "role": "user",
            "content": """
            Analyze this research paper:
            This paper presents a comprehensive analysis of deep learning approaches 
            in natural language processing. We compare various transformer architectures 
            and their performance on standard NLP tasks.
            """
        }]
        
        result = operation.process_messages(messages)
        
        # Verify LLMResult structure
        assert isinstance(result, LLMResult)
        assert result.total_cost == 0.0
        assert result.validated == True
        
        # Extract output data from tool call
        tool_call = result.response["choices"][0]["message"]["tool_calls"][0]
        output_data = json.loads(tool_call["function"]["arguments"])
        
        # Verify structure and types
        assert isinstance(output_data["title"], str)
        assert isinstance(output_data["authors"], list)
        assert isinstance(output_data["methodology"], str)
        assert isinstance(output_data["findings"], list)
        assert len(output_data["findings"]) > 0
        assert isinstance(output_data["limitations"], list)
        assert isinstance(output_data["future_work"], list)

def test_execute(sample_config, mock_runner):
    mock_model = MagicMock()
    
    class MockOutput:
        def model_dump(self):
            return {
                "first_name": "John",
                "last_name": "Doe"
            }
    
    mock_processor = Mock(return_value=MockOutput())
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        messages = [{"role": "user", "content": "Extract information about John Doe"}]
        result, timing = OSSLLMOperation.execute(sample_config, messages)
        
        assert isinstance(result, LLMResult)
        assert isinstance(timing, float)
        assert result.total_cost == 0.0
        assert result.validated == True

@pytest.mark.parametrize("invalid_input", [
    {"missing_model": "test"},  # Missing model_path
    {"model_path": "test", "output_schema": None},  # Invalid schema
    {"model_path": "test", "output_schema": {"invalid_type": "unknown"}},  # Invalid type
])
def test_invalid_configs(invalid_input, mock_runner):
    with pytest.raises(Exception):
        OSSLLMOperation(invalid_input, runner=mock_runner)
'''

import pytest
from unittest.mock import Mock, patch, MagicMock
from docetl.operations.utils.oss_llm import OutlinesBackend
from docetl.operations.utils.llm import LLMResult, InvalidOutputError
import json

@pytest.fixture
def mock_global_config():
    return {
        "max_tokens": 4096,
    }

@pytest.fixture
def mock_output_schema():
    return {
        "first_name": "str",
        "last_name": "str"
    }

@pytest.fixture
def mock_research_schema():
    return {
        "title": "str",
        "authors": "list",
        "methodology": "str",
        "findings": "list",
        "limitations": "list",
        "future_work": "list"
    }

@pytest.fixture
def mock_research_output():
    research_data = {
        "title": "Deep Learning in Natural Language Processing",
        "authors": ["John Smith", "Jane Doe"],
        "methodology": "Comparative analysis of transformer architectures",
        "findings": [
            "Improved accuracy by 15%",
            "Reduced training time by 30%"
        ],
        "limitations": [
            "Limited dataset size",
            "Computational constraints"
        ],
        "future_work": [
            "Extend to multilingual models",
            "Optimize for edge devices"
        ]
    }
    
    class MockOutput:
        def model_dump(self):
            return research_data
    return MockOutput()

def test_process_messages(mock_global_config, mock_output_schema):
    mock_model = MagicMock()
    
    class MockOutput:
        def model_dump(self):
            return {
                "first_name": "John",
                "last_name": "Doe"
            }
    
    mock_processor = Mock(return_value=MockOutput())
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        backend = OutlinesBackend(mock_global_config)
        messages = [
            {"role": "user", "content": "Extract information about John Doe"}
        ]
        
        result = backend.process_messages(
            model_path="meta-llama/Llama-3.2-1B-Instruct",
            messages=messages,
            output_schema=mock_output_schema
        )
        
        assert isinstance(result, LLMResult)
        assert result.total_cost == 0.0
        assert result.validated == True
        assert "tool_calls" in result.response["choices"][0]["message"]
        
        tool_call = result.response["choices"][0]["message"]["tool_calls"][0]
        output_data = json.loads(tool_call["function"]["arguments"])
        assert output_data["first_name"] == "John"
        assert output_data["last_name"] == "Doe"

def test_research_paper_analysis(mock_global_config, mock_research_schema, mock_research_output):
    mock_model = MagicMock()
    mock_processor = Mock(return_value=mock_research_output)
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        backend = OutlinesBackend(mock_global_config)
        messages = [{
            "role": "user",
            "content": """
            Analyze this research paper:
            This paper presents a comprehensive analysis of deep learning approaches 
            in natural language processing. We compare various transformer architectures 
            and their performance on standard NLP tasks.
            """
        }]
        
        result = backend.process_messages(
            model_path="meta-llama/Llama-3.2-1B-Instruct",
            messages=messages,
            output_schema=mock_research_schema
        )
        
        # Verify LLMResult structure
        assert isinstance(result, LLMResult)
        assert result.total_cost == 0.0
        assert result.validated == True
        
        # Extract output data from tool call
        tool_call = result.response["choices"][0]["message"]["tool_calls"][0]
        output_data = json.loads(tool_call["function"]["arguments"])
        
        # Verify structure and types
        assert isinstance(output_data["title"], str)
        assert isinstance(output_data["authors"], list)
        assert isinstance(output_data["methodology"], str)
        assert isinstance(output_data["findings"], list)
        assert len(output_data["findings"]) > 0
        assert isinstance(output_data["limitations"], list)
        assert isinstance(output_data["future_work"], list)

def test_model_reuse(mock_global_config, mock_output_schema):
    """Test that the same model is reused for multiple calls"""
    mock_model = MagicMock()
    mock_processor = Mock(return_value=MagicMock(model_dump=lambda: {"first_name": "John", "last_name": "Doe"}))
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        backend = OutlinesBackend(mock_global_config)
        messages = [{"role": "user", "content": "Test message"}]
        model_path = "meta-llama/Llama-3.2-1B-Instruct"
        
        # First call should initialize the model
        backend.process_messages(model_path, messages, mock_output_schema)
        # Second call should reuse the model
        backend.process_messages(model_path, messages, mock_output_schema)
        
        # Check that transformers was only called once
        mock_transformers.assert_called_once()

def test_invalid_output_schema(mock_global_config):
    backend = OutlinesBackend(mock_global_config)
    messages = [{"role": "user", "content": "Test"}]
    
    with pytest.raises(Exception):
        backend.process_messages(
            model_path="test-model",
            messages=messages,
            output_schema={"invalid_type": "unknown"}
        )

def test_model_error_handling(mock_global_config, mock_output_schema):
    """Test handling of model processing errors"""
    mock_model = MagicMock()
    mock_processor = Mock(side_effect=Exception("Model processing error"))
    
    with patch('outlines.models.transformers', return_value=mock_model), \
         patch('outlines.generate.json', return_value=mock_processor):
        
        backend = OutlinesBackend(mock_global_config)
        messages = [{"role": "user", "content": "Test message"}]
        
        with pytest.raises(InvalidOutputError) as exc_info:
            backend.process_messages(
                model_path="test-model",
                messages=messages,
                output_schema=mock_output_schema
            )
        
        assert "Model processing error" in str(exc_info.value)