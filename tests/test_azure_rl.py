from docetl.runner import DSLRunner
import pytest
from docetl.operations.map import MapOperation
import random
import os
from dotenv import load_dotenv
from tests.conftest import api_wrapper

load_dotenv()


@pytest.fixture
def simple_map_config():
    return {
        "name": "simple_sentiment_analysis",
        "type": "map",
        "prompt": "Analyze the sentiment of the following text: '{{ input.text }}'. Classify it as either positive, negative, or neutral.",
        "output": {"schema": {"sentiment": "string"}},
        "model": "azure/gpt-4o",
    }


@pytest.fixture
def sample_documents():
    sentiments = ["positive", "negative", "neutral"]
    documents = []
    for _ in range(8):
        sentiment = random.choice(sentiments)
        if sentiment == "positive":
            text = f"I absolutely love this product! It's amazing and works perfectly."
        elif sentiment == "negative":
            text = f"This is the worst experience I've ever had. Terrible service."
        else:
            text = f"The product works as expected. Nothing special to report."
        documents.append({"text": text})
    return documents


def test_map_operation_over_15_documents(simple_map_config, sample_documents):
    # Set environment variables specific to this test
    os.environ["AZURE_API_BASE"] = os.getenv("LOW_RES_AZURE_API_BASE")
    os.environ["AZURE_API_VERSION"] = os.getenv("LOW_RES_AZURE_API_VERSION")
    os.environ["AZURE_API_KEY"] = os.getenv("LOW_RES_AZURE_API_KEY")

    runner = DSLRunner(
        {
            "default_model": "gpt-4o-mini",
            "operations": [],
            "pipeline": {"steps": [], "output": {"path": "/tmp/testingdocetl.json"}},
        },
        max_threads=64,
    )

    operation = MapOperation(runner, simple_map_config, "azure/gpt-4o", 4)
    results, cost = operation.execute(sample_documents + sample_documents)

    assert len(results) == 16
    assert all("sentiment" in result for result in results)
    assert all(
        result["sentiment"] in ["positive", "negative", "neutral"] for result in results
    )
