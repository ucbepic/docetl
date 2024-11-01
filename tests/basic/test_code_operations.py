import pytest
from docetl.operations.code_operations import CodeMapOperation

class MockRunner:
    """A simple mock runner for testing"""
    def __init__(self, config=None):
        self.config = config or {}
        self.console = None

@pytest.fixture
def mock_runner():
    return MockRunner()

@pytest.fixture
def sample_docs():
    return [
        {
            "text": "The quick brown fox jumped over two tired turtles. Today is terrific!"
        },
        {
            "text": "Testing multiple sentences. This is another test. Totally awesome."
        },
        {
            "text": "No words with t here."
        }
    ]

@pytest.fixture
def sentence_counter_op(mock_runner):
    return CodeMapOperation(
        config={
            "name": "sentence_counter",
            "type": "code_map",
            "code": """
def transform(doc):
    text = doc.get('text', '')
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return {
        'text': text,
        'sentence_count': len(sentences)
    }
"""
        },
        runner=mock_runner,
        default_model="gpt-staru-turbo",
        max_threads=4
    )

@pytest.fixture
def t_word_extractor_op(mock_runner):
    return CodeMapOperation(
        config={
            "name": "t_word_extractor",
            "type": "code_map",
            "code": """
def transform(doc):
    text = doc.get('text', '')
    # Only match actual words (more than one character)
    t_words = [
        word.strip('.,!?') 
        for word in text.split() 
        if word.lower().startswith('t') and len(word.strip('.,!?')) > 1
    ]
    return {
        'text': text,
        't_words': t_words,
        't_word_count': len(t_words)
    }
"""
        },
        runner=mock_runner,
        default_model="gpt-staru-turbo",
        max_threads=4
    )

def test_sentence_counter(sentence_counter_op, sample_docs):
    results, cost = sentence_counter_op.execute(sample_docs)
    
    assert len(results) == 3
    assert results[0]['sentence_count'] == 2 
    assert results[1]['sentence_count'] == 3  
    assert results[2]['sentence_count'] == 1 
    
    assert cost == 0.0
    
    for original, result in zip(sample_docs, results):
        assert result['text'] == original['text']

def test_t_word_extractor(t_word_extractor_op, sample_docs):
    results, cost = t_word_extractor_op.execute(sample_docs)
    
    assert len(results[0]['t_words']) == 6
    assert set(results[0]['t_words']) == {'The', 'two', 'tired', 'turtles', 'Today', 'terrific'}
    assert results[0]['t_word_count'] == 6
    
    assert len(results[1]['t_words']) == 4
    assert set(results[1]['t_words']) == {'Testing', 'This', 'test', 'Totally'}
    assert results[1]['t_word_count'] == 4
    
    assert len(results[2]['t_words']) == 0
    assert results[2]['t_word_count'] == 0
    
    assert cost == 0.0

def test_invalid_code():
    """Test that invalid Python code raises appropriate error"""
    with pytest.raises(ValueError) as exc_info:
        CodeMapOperation(
            config={
                "name": "invalid_code",
                "type": "code_map",
                "code": """
def transform(doc):
    this is invalid python code
    return {}
"""
            },
            runner=MockRunner(),
            default_model="gpt-staru-turbo",  
            max_threads=4  
        )
    assert "Invalid code configuration" in str(exc_info.value)

def test_missing_transform_function():
    """Test that code without transform function raises error"""
    with pytest.raises(ValueError) as exc_info:
        CodeMapOperation(
            config={
                "name": "missing_transform",
                "type": "code_map",
                "code": """
def some_other_function(doc):
    return {}
"""
            },
            runner=MockRunner(),
            default_model="gpt-staru-turbo",  
            max_threads=4  
        )
    assert "Code must define a 'transform' function" in str(exc_info.value)

def test_empty_input(sentence_counter_op):
    """Test handling of empty input list"""
    results, cost = sentence_counter_op.execute([])
    assert results == []
    assert cost == 0.0

def test_missing_text_field(sentence_counter_op):
    """Test handling of documents without 'text' field"""
    doc_without_text = [{"other_field": "value"}]
    results, cost = sentence_counter_op.execute(doc_without_text)
    assert results[0]['sentence_count'] == 0

def test_drop_keys(mock_runner):
    """Test that drop_keys configuration works"""
    op = CodeMapOperation(
        config={
            "name": "drop_test",
            "type": "code_map",
            "code": """
def transform(doc):
    return {
        'keep_this': 'value',
        'drop_this': 'should not appear'
    }
""",
            "drop_keys": ["drop_this"]
        },
        runner=mock_runner,
        default_model="gpt-staru-turbo",  
        max_threads=4  
    )
    
    results, _ = op.execute([{"text": "dummy"}])
    assert 'keep_this' in results[0]
    assert 'drop_this' not in results[0]