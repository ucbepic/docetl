# Output Modes Testing

This directory contains comprehensive tests for the new output modes functionality in DocETL, which allows operations to use either "tools" mode (default) or "structured_output" mode.

## Test Files

### `test_output_modes.py` - Full Integration Tests
This is the complete test suite that requires the full DocETL environment with dependencies:

- Tests both map and reduce operations with complex schemas
- Uses synthetic documents containing fruit mentions
- Compares accuracy between "tools" and "structured_output" modes
- Includes folding configuration for reduce operations
- Requires pytest and all DocETL dependencies

**Complex Schemas Tested:**
- Map operation: `list[{fruit: str, most_similar_veggie: str, reasoning: str}]`
- Reduce operation: `list[{fruit: str, count: int, document_ids: list[int]}]`

### `test_output_modes_simple.py` - Demonstration Version
A simplified version using mock data that demonstrates the testing concept without requiring dependencies:

- Shows the testing approach and structure
- Uses mock results to demonstrate comparison logic
- Can run immediately without setup
- Perfect for understanding the test methodology

## Running the Tests

### Prerequisites for Full Tests
```bash
# Install dependencies (example using pip)
pip install pytest pyyaml
pip install -e .  # Install docetl in development mode

# Or using poetry (if available)
poetry install
```

### Run Full Integration Tests
```bash
# Run all output mode tests
pytest tests/test_output_modes.py -v

# Run specific test
pytest tests/test_output_modes.py::test_map_operation_complex_schema -v

# Run with output mode parameter
pytest tests/test_output_modes.py::test_map_operation_complex_schema[tools] -v
pytest tests/test_output_modes.py::test_map_operation_complex_schema[structured_output] -v
```

### Run Demonstration Version
```bash
# No dependencies required
python3 tests/test_output_modes_simple.py
```

## Test Structure

### Synthetic Data
Both test files use the same synthetic documents containing various fruits:
- Apples (mentioned multiple times for reduce testing)
- Bananas, strawberries, oranges, grapes, mangoes, etc.
- Each document also mentions vegetables for comparison mapping

### Map Operation Testing
Tests the extraction of fruits and their most similar vegetables:
```yaml
output:
  schema:
    fruits_and_veggies: "list[{fruit: str, most_similar_veggie: str, reasoning: str}]"
  mode: "tools"  # or "structured_output"
```

### Reduce Operation Testing
Tests finding duplicate fruits across documents with folding:
```yaml
fold_batch_size: 3
fold_prompt: "Summarize the fruits mentioned in these documents and their counts..."
output:
  schema:
    duplicate_fruits: "list[{fruit: str, count: int, document_ids: list[int]}]"
  mode: "structured_output"  # or "tools"
```

## Accuracy Comparison

The tests automatically compare:

1. **Map Operations:**
   - Number of fruit-veggie mappings extracted
   - Number of unique fruits identified
   - Quality of vegetable mappings

2. **Reduce Operations:**
   - Number of duplicate fruits found
   - Accuracy against expected duplicates (apple, banana, strawberry)
   - Correctness of document ID tracking

## Expected Results

Based on the current implementation, structured output mode typically shows:
- More consistent schema adherence
- Better extraction completeness
- More accurate duplicate detection
- Improved reasoning quality

## Configuration

Tests use the new output mode configuration:
```yaml
operations:
  - name: "extract_fruits"
    type: "map"
    output:
      mode: "structured_output"  # or "tools" (default)
      schema:
        fruits_and_veggies: "list[{fruit: str, most_similar_veggie: str}]"
```

The output mode is automatically extracted from `op_config.get("output", {}).get("mode")` as implemented in the refactored API.

## Notes

- Tools mode remains the default for backward compatibility
- Structured output mode uses LiteLLM's `response_format` with `json_schema`
- Both modes support validation, gleaning, and caching
- The tests demonstrate real-world complex schema usage patterns