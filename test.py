from docetl.operations.code_operations import CodeMapOperation

doc_data = [
    {"text": "Hello world. This is a test. Another sentence."},
    {"text": "Single sentence only."}
]

class MockRunner:
    """A simple mock runner for testing"""
    def __init__(self, config=None):
        self.config = config or {}
        self.console = None

def main():
    mock_runner = MockRunner()

    sentence_counter = CodeMapOperation(
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
        default_model="gpt-3.5-turbo",  # This won't be used but is required
        max_threads=4
    )

    results, cost = sentence_counter.execute(doc_data)
    print("Results:", results)
    print("Cost:", cost)

if __name__ == "__main__":
    main()