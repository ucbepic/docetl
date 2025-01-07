import pytest
from unittest.mock import Mock, patch, MagicMock
from docetl.operations.hf_outlines import HuggingFaceMapOperation

@pytest.fixture
def mock_runner():
    return Mock()

@pytest.fixture
def sample_config():
    return {
        "name": "test_hf_operation",
        "type": "hf_map",
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "output_schema": {
            "first_name": "str",
            "last_name": "str"
        },
        "prompt_template": "Extract customer information from this text",
        "max_tokens": 4096
    }

@pytest.fixture
def research_config():
    return {
        "name": "research_analyzer",
        "type": "hf_map",
        "model_path": "meta-llama/Llama-3.2-1B-Instruct",
        "output_schema": {
            "title": "str",
            "authors": "list",
            "methodology": "str",
            "findings": "list",
            "limitations": "list",
            "future_work": "list"
        },
        "prompt_template": "Analyze the following research paper abstract.\nExtract key components and summarize findings.",
        "max_tokens": 4096
    }

@pytest.fixture
def mock_research_output():
    class MockOutput:
        def model_dump(self):
            return {
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
    return MockOutput()

def test_process_item(sample_config, mock_runner):
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
        
        operation = HuggingFaceMapOperation(sample_config, runner=mock_runner)
        test_item = {"message": "test message"}
        result = operation.process_item(test_item)
        
        assert isinstance(result, dict)
        assert "first_name" in result
        assert "last_name" in result
        assert "message" in result

def test_research_paper_analysis(research_config, mock_research_output, mock_runner):
    mock_model = MagicMock()
    mock_processor = Mock(return_value=mock_research_output)
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        operation = HuggingFaceMapOperation(research_config, runner=mock_runner)
        test_item = {
            "abstract": """
            This paper presents a comprehensive analysis of deep learning approaches 
            in natural language processing. We compare various transformer architectures 
            and their performance on standard NLP tasks.
            """
        }
        result = operation.process_item(test_item)
        
        # Verify structure and types
        assert isinstance(result, dict)
        assert "title" in result
        assert isinstance(result["title"], str)
        assert "authors" in result
        assert isinstance(result["authors"], list)
        assert "methodology" in result
        assert isinstance(result["methodology"], str)
        assert "findings" in result
        assert isinstance(result["findings"], list)
        assert len(result["findings"]) > 0
        assert "limitations" in result
        assert isinstance(result["limitations"], list)
        assert "future_work" in result
        assert isinstance(result["future_work"], list)
        
        # Verify original input is preserved
        assert "abstract" in result

def test_execute(sample_config, mock_runner):
    mock_model = MagicMock()
    mock_processor = Mock(return_value={"first_name": "John", "last_name": "Doe"})
    
    with patch('outlines.models.transformers', return_value=mock_model) as mock_transformers, \
         patch('outlines.generate.json', return_value=mock_processor):
        
        input_data = [{"message": "test message"}]
        results, timing = HuggingFaceMapOperation.execute(sample_config, input_data)
        assert len(results) == 1
        assert isinstance(timing, float)