from docetl.operations.code_operations import CodeMapOperation

doc_data = [
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

class MockRunner:
    """A simple mock runner for testing"""
    def __init__(self, config=None):
        self.config = config or {}
        self.console = None

def main():
    mock_runner = MockRunner()

    t_word_extractor = CodeMapOperation(
        config={
            "name": "t_word_extractor",
            "type": "code_map",
            "code": """
def transform(doc):
    # Get text and split into words
    text = doc.get('text', '')
    # Split words, convert to lowercase, and filter words starting with 't'
    t_words = [
        word.strip('.,!?') 
        for word in text.split() 
        if word.lower().startswith('t')
    ]
    
    return {
        'text': text,
        't_words': t_words,
        't_word_count': len(t_words)
    }
"""
        },
        runner=mock_runner,
        default_model="gpt-staru-turbo",  # This won't be used but is required
        max_threads=4
    )

    results, cost = t_word_extractor.execute(doc_data)
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\nDocument {i}:")
        print(f"Original text: {result['text']}")
        print(f"Words starting with 't': {result['t_words']}")
        print(f"Count: {result['t_word_count']}")
    print("\nCost:", cost)

if __name__ == "__main__":
    main()