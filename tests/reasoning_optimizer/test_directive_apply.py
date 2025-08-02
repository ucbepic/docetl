#!/usr/bin/env python3
"""
Simple apply tests for directive testing - just ensure apply() doesn't crash.
"""

# Simple apply tests - no pytest needed

from docetl.reasoning_optimizer.directives import (
    ChainingDirective, 
    GleaningDirective,
    ReduceGleaningDirective,
    ChangeModelDirective,
    OperatorFusionDirective,
    DocSummarizationDirective,
    IsolatingSubtasksDirective,
    DocCompressionDirective,
    DeterministicDocCompressionDirective,
    DocumentChunkingDirective,
    ChunkHeaderSummaryDirective,
    ChunkSamplingDirective,
    TakeHeadTailDirective
)


def test_chaining_apply():
    """Test that chaining apply doesn't crash"""
    directive = ChainingDirective()
    
    # Simple pipeline with one map op
    ops_list = [
        {
            "name": "extract_info",
            "type": "map", 
            "prompt": "Extract name and age from: {{ input.text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"info": "string"}}
        }
    ]
    
    # Mock rewrite schema
    from docetl.reasoning_optimizer.instantiate_schemas import ChainingInstantiateSchema, MapOpConfig
    rewrite = ChainingInstantiateSchema(
        new_ops=[
            MapOpConfig(
                name="extract_name",
                prompt="Extract name from: {{ input.text }}",
                output_keys=["name"],
                model="gpt-4o-mini"
            ),
            MapOpConfig(
                name="extract_final",
                prompt="Format name {{ input.name }} from: {{ input.text }}",
                output_keys=["info"],
                model="gpt-4o-mini"
            )
        ]
    )
    
    # Should not crash
    result = directive.apply("azure/gpt-4o-mini", ops_list, "extract_info", rewrite)
    assert isinstance(result, list)
    assert len(result) == 2  # Should replace 1 op with 2 ops


def test_gleaning_apply():
    """Test that gleaning apply doesn't crash"""
    directive = GleaningDirective()
    
    ops_list = [
        {
            "name": "extract_entities",
            "type": "map",
            "prompt": "Extract entities from: {{ input.text }}",
            "model": "gpt-4o-mini", 
            "output": {"schema": {"entities": "list"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import GleaningInstantiateSchema
    rewrite = GleaningInstantiateSchema(
        validation_prompt="Check if all entities are correctly identified",
        num_rounds=2,
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "extract_entities", rewrite)
    assert isinstance(result, list)
    assert len(result) == 1
    # Should add gleaning config
    assert "gleaning" in result[0]


def test_change_model_apply():
    """Test that change model apply doesn't crash"""
    directive = ChangeModelDirective()
    
    ops_list = [
        {
            "name": "analyze_text",
            "type": "map",
            "prompt": "Analyze: {{ input.text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"analysis": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import ChangeModelInstantiateSchema
    rewrite = ChangeModelInstantiateSchema(model="gpt-4o")
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "analyze_text", rewrite)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["model"] == "gpt-4o"


def test_operator_fusion_map_map_apply():
    """Test that operator fusion apply doesn't crash - map + map"""
    directive = OperatorFusionDirective()
    
    ops_list = [
        {
            "name": "extract_names",
            "type": "map",
            "prompt": "Extract names from: {{ input.text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"names": "list"}}
        },
        {
            "name": "clean_names",
            "type": "map", 
            "prompt": "Clean these names: {{ input.names }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"clean_names": "list"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import OperatorFusionInstantiateSchema
    rewrite = OperatorFusionInstantiateSchema(
        fused_prompt="Extract and clean names from: {{ input.text }}",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["extract_names", "clean_names"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 1  # Should fuse 2 ops into 1


def test_operator_fusion_filter_map_apply():
    """Test that operator fusion apply doesn't crash - filter + map"""
    directive = OperatorFusionDirective()
    
    ops_list = [
        {
            "name": "filter_valid",
            "type": "filter",
            "prompt": "Keep only valid entries: {{ input.text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"_bool": "bool"}}
        },
        {
            "name": "extract_info",
            "type": "map", 
            "prompt": "Extract info from: {{ input.text }}", 
            "model": "gpt-4o-mini",
            "output": {"schema": {"info": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import OperatorFusionInstantiateSchema
    rewrite = OperatorFusionInstantiateSchema(
        fused_prompt="Extract info from valid entries: {{ input.text }}",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["filter_valid", "extract_info"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 2 
    assert result[1]["type"] == "code_filter", f"Second op should be code_filter, got {result[1]['type']}"


def test_operator_fusion_map_filter_apply():
    """Test that operator fusion apply doesn't crash - map + filter"""
    directive = OperatorFusionDirective()
    
    ops_list = [
        {
            "name": "extract_sentiment",
            "type": "map",
            "prompt": "Extract sentiment from: {{ input.text }}",
            "model": "gpt-4o-mini", 
            "output": {"schema": {"sentiment": "string"}}
        },
        {
            "name": "filter_positive",
            "type": "filter",
            "prompt": "Keep only positive sentiment: {{ input.sentiment }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"_bool": "bool"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import OperatorFusionInstantiateSchema
    rewrite = OperatorFusionInstantiateSchema(
        fused_prompt="Keep entries with positive sentiment from: {{ input.text }}",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["extract_sentiment", "filter_positive"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 2  # Should fuse 2 ops into 1
    assert result[0]["type"] == "map", f"First op should be map, got {result[0]['type']}"
    assert result[1]["type"] == "code_filter", f"Second op should be code_filter, got {result[1]['type']}"


def test_operator_fusion_filter_filter_apply():
    """Test that operator fusion apply doesn't crash - filter + filter"""
    directive = OperatorFusionDirective()
    
    ops_list = [
        {
            "name": "filter_valid",
            "type": "filter",
            "prompt": "Keep only valid entries: {{ input.text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"_bool": "bool"}}
        },
        {
            "name": "filter_recent",
            "type": "filter",
            "prompt": "Keep only recent entries: {{ input.text }}",
            "model": "gpt-4o-mini", 
            "output": {"schema": {"_bool": "bool"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import OperatorFusionInstantiateSchema
    rewrite = OperatorFusionInstantiateSchema(
        fused_prompt="Keep only valid and recent entries: {{ input.text }}",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["filter_valid", "filter_recent"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 1  # Should fuse 2 ops into 1


def test_operator_fusion_map_reduce_apply():
    """Test that operator fusion apply doesn't crash - map + reduce"""
    directive = OperatorFusionDirective()
    
    ops_list = [
        {
            "name": "extract_themes",
            "type": "map",
            "prompt": "Extract themes from: {{ input.text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"themes": "list"}}
        },
        {
            "name": "summarize_themes",
            "type": "reduce",
            "prompt": "Summarize these themes: {{ input.themes }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"summary": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import OperatorFusionInstantiateSchema
    rewrite = OperatorFusionInstantiateSchema(
        fused_prompt="Extract themes and create summary from: {{ input.text }}",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["extract_themes", "summarize_themes"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 1  # Should fuse 2 ops into 1
    assert result[0]["type"] == "reduce", f"First op should be reduce, got {result[0]['type']}"


def test_doc_summarization_apply():
    """Test that doc summarization apply doesn't crash"""
    directive = DocSummarizationDirective()
    
    ops_list = [
        {
            "name": "analyze_doc",
            "type": "map",
            "prompt": "Analyze this document: {{ input.document }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"analysis": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import DocSummarizationInstantiateSchema
    rewrite = DocSummarizationInstantiateSchema(
        prompt="Summarize the key points of: {{ input.document }}",
        model="gpt-4o-mini",
        document_key="document",
        name="summarize_document"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "analyze_doc", rewrite)
    assert isinstance(result, list)
    assert len(result) == 2  # Should add summarization op before target


def test_isolating_subtasks_apply():
    """Test that isolating subtasks apply doesn't crash"""
    directive = IsolatingSubtasksDirective()
    
    ops_list = [
        {
            "name": "complex_analysis", 
            "type": "map",
            "prompt": "Analyze sentiment and extract entities from: {{ input.text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"result": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import IsolatingSubtasksInstantiateSchema, SubtaskConfig
    rewrite = IsolatingSubtasksInstantiateSchema(
        subtasks=[
            SubtaskConfig(
                name="sentiment_analysis",
                prompt="Analyze sentiment of: {{ input.text }}",
                output_keys=["sentiment"]
            ),
            SubtaskConfig(
                name="entity_extraction", 
                prompt="Extract entities from: {{ input.text }}",
                output_keys=["entities"]
            )
        ],
        aggregation_prompt="Combine sentiment {{ input.subtask_1_output }} and entities {{ input.subtask_2_output }}"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "complex_analysis", rewrite)
    assert isinstance(result, list)
    assert len(result) == 2, f"Expected 2 ops, got {len(result)}"
    assert result[0]["type"] == "parallel_map", f"First op should be parallel_map, got {result[0]['type']}"


def test_reduce_gleaning_apply():
    """Test that reduce gleaning apply doesn't crash"""
    directive = ReduceGleaningDirective()
    
    ops_list = [
        {
            "name": "summarize_docs",
            "type": "reduce", 
            "prompt": "Summarize these documents: {{ input.docs }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"summary": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import GleaningInstantiateSchema
    rewrite = GleaningInstantiateSchema(
        validation_prompt="Check if the summary covers all key points",
        num_rounds=2,
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "summarize_docs", rewrite)
    assert isinstance(result, list)
    assert len(result) == 1
    # Should add gleaning config to reduce op
    assert "gleaning" in result[0]


def test_doc_compression_apply():
    """Test that doc compression apply doesn't crash"""
    directive = DocCompressionDirective()
    
    ops_list = [
        {
            "name": "analyze_paper",
            "type": "map",
            "prompt": "Analyze this research paper: {{ input.paper }}",
            "model": "gpt-4o-mini", 
            "output": {"schema": {"analysis": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import DocCompressionInstantiateSchema
    rewrite = DocCompressionInstantiateSchema(
        name="extract_key_points",
        document_key="paper",
        prompt="Extract the key methodology and findings from this research paper",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["analyze_paper"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 2  # Should add extract op before target


def test_deterministic_doc_compression_apply():
    """Test that deterministic doc compression apply doesn't crash"""
    directive = DeterministicDocCompressionDirective()
    
    ops_list = [
        {
            "name": "process_doc",
            "type": "map",
            "prompt": "Process this document: {{ input.document }}", 
            "model": "gpt-4o-mini",
            "output": {"schema": {"processed": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import DeterministicDocCompressionInstantiateSchema
    rewrite = DeterministicDocCompressionInstantiateSchema(
        name="compress_document",
        code="""
def transform(input_doc):
    import re
    compressed_doc = re.sub(r'\\s+', ' ', input_doc['document'])
    return {'document': compressed_doc}
"""
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "process_doc", rewrite)
    assert isinstance(result, list)
    assert len(result) == 2  # Should add preprocessing op before target


def test_doc_chunking_apply():
    """Test that doc chunking apply doesn't crash"""
    directive = DocumentChunkingDirective()
    
    ops_list = [
        {
            "name": "analyze_document",
            "type": "map",
            "prompt": "Analyze this document: {{ input.document }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"analysis": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import DocumentChunkingInstantiateSchema
    rewrite = DocumentChunkingInstantiateSchema(
        chunk_size=1000,
        split_key="document",
        sub_prompt="Analyze this document chunk: {{ input.document_chunk_rendered }}",
        reduce_prompt="Combine the analysis results: {% for input in inputs %}{{ input.analysis }}{% endfor %}"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "analyze_document", rewrite)
    assert isinstance(result, list)
    assert len(result) == 4  # Should be split -> gather -> map -> reduce


def test_chunk_header_summary_apply():
    """Test that chunk header summary apply doesn't crash"""
    directive = ChunkHeaderSummaryDirective()
    
    ops_list = [
        {
            "name": "split_legal_docs",
            "type": "split",
            "split_key": "agreement_text",
            "method": "token_count",
            "method_kwargs": {"num_tokens": 1000}
        },
        {
            "name": "gather_legal_context",
            "type": "gather",
            "content_key": "agreement_text_chunk",
            "doc_id_key": "split_legal_docs_id",
            "order_key": "split_legal_docs_chunk_num",
            "peripheral_chunks": {
                "previous": {"tail": {"count": 1}},
                "next": {"head": {"count": 1}}
            }
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import ChunkHeaderSummaryInstantiateSchema
    rewrite = ChunkHeaderSummaryInstantiateSchema(
        header_extraction_prompt="Extract headers from: {{ input.agreement_text_chunk }}",
        summary_prompt="Summarize this legal text: {{ input.agreement_text_chunk }}",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["split_legal_docs", "gather_legal_context"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 3  # Should insert parallel_map between split and gather
    assert result[1]["type"] == "parallel_map"
    assert "doc_header_key" in result[2]  # gather should have doc_header_key


def test_chunk_sampling_apply():
    """Test that chunk sampling apply doesn't crash"""
    directive = ChunkSamplingDirective()
    
    # Simple Gather -> Map sequence
    ops_list = [
        {
            "name": "gather_chunks",
            "type": "gather",
            "content_key": "document_chunk",
            "doc_id_key": "split_docs_id",
            "order_key": "split_docs_chunk_num",
        },
        {
            "name": "categorize_document",
            "type": "map",
            "prompt": "What category does this document belong to? {{ input.document_chunk_rendered }}",
            "output": {"schema": {"category": "string"}},
        },
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import ChunkSamplingInstantiateSchema
    rewrite = ChunkSamplingInstantiateSchema(
        method="uniform",
        samples=0.05
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, ["gather_chunks", "categorize_document"], rewrite)
    assert isinstance(result, list)
    assert len(result) == 3  # Should insert sample between gather and map
    assert result[0]["name"] == "gather_chunks"
    assert result[1]["type"] == "sample"
    assert result[1]["method"] == "uniform"
    assert result[1]["samples"] == 0.05
    assert result[2]["name"] == "categorize_document"


def test_take_head_tail_apply():
    """Test that take head tail apply doesn't crash"""
    directive = TakeHeadTailDirective()
    
    # Test with Map operation
    ops_list = [
        {
            "name": "classify_document",
            "type": "map",
            "prompt": "Classify this document: {{ input.content }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"category": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import TakeHeadTailInstantiateSchema
    rewrite = TakeHeadTailInstantiateSchema(
        name="truncate_content",
        document_key="content",
        head_words=5,
        tail_words=5
    )
    
    result = directive.apply(ops_list, "classify_document", rewrite)
    assert isinstance(result, list)
    assert len(result) == 2  # Should add code_map before target
    assert result[0]["name"] == "truncate_content"
    assert result[0]["type"] == "code_map"
    assert "def transform" in result[0]["function"]
    assert result[1]["name"] == "classify_document"
    
    # Test with Filter operation
    filter_ops_list = [
        {
            "name": "filter_spam",
            "type": "filter",
            "prompt": "Is this spam? {{ input.email_text }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"_bool": "bool"}}
        }
    ]
    
    filter_rewrite = TakeHeadTailInstantiateSchema(
        name="truncate_email",
        document_key="email_text",
        head_words=10,
        tail_words=0
    )
    
    filter_result = directive.apply(filter_ops_list, "filter_spam", filter_rewrite)
    assert isinstance(filter_result, list)
    assert len(filter_result) == 2
    assert filter_result[0]["type"] == "code_map"
    assert filter_result[1]["type"] == "filter"
    
    # Assert that you can run the function on some document
    doc = {"content": "This is a test document with more than ten words that should be truncated properly"}
    namespace = {}
    exec(result[0]["function"], namespace)
    transformed_result = namespace["transform"](doc)
    # With 5 head words and 5 tail words, and original has more than 10 words, should be truncated
    assert "content" in transformed_result
    assert len(transformed_result["content"].split()) <= 12  # 5 + " ... " + 5 = at most 12 tokens


if __name__ == "__main__":
    # Run all tests
    test_chaining_apply()
    print("✅ Chaining apply test passed")
    
    test_gleaning_apply()
    print("✅ Gleaning apply test passed")
    
    test_change_model_apply()
    print("✅ Change model apply test passed")
    
    test_operator_fusion_map_map_apply()
    print("✅ Operator fusion (map+map) apply test passed")
    
    test_operator_fusion_filter_map_apply()
    print("✅ Operator fusion (filter+map) apply test passed")
    
    test_operator_fusion_map_filter_apply()
    print("✅ Operator fusion (map+filter) apply test passed")
    
    test_operator_fusion_filter_filter_apply()
    print("✅ Operator fusion (filter+filter) apply test passed")
    
    test_operator_fusion_map_reduce_apply()
    print("✅ Operator fusion (map+reduce) apply test passed")
    
    test_doc_summarization_apply()
    print("✅ Doc summarization apply test passed")
    
    test_isolating_subtasks_apply()
    print("✅ Isolating subtasks apply test passed")
    
    test_reduce_gleaning_apply()
    print("✅ Reduce gleaning apply test passed")
    
    test_doc_compression_apply()
    print("✅ Doc compression apply test passed")
    
    test_deterministic_doc_compression_apply()
    print("✅ Deterministic doc compression apply test passed")
    
    test_doc_chunking_apply()
    print("✅ Doc chunking apply test passed")
    
    test_chunk_header_summary_apply()
    print("✅ Chunk header summary apply test passed")

    test_chunk_sampling_apply()
    print("✅ Chunk sampling apply test passed")

    test_take_head_tail_apply()
    print("✅ Take head tail apply test passed")

    print("\n🎉 All directive apply tests passed!")
