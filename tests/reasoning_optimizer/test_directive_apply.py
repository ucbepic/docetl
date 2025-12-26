#!/usr/bin/env python3
"""
Simple apply tests for directive testing - just ensure apply() doesn't crash.
"""

# Simple apply tests - no pytest needed

from docetl.reasoning_optimizer.directives import (
    ChainingDirective, 
    GleaningDirective,
    ReduceGleaningDirective,
    ReduceChainingDirective,
    ChangeModelDirective,
    OperatorFusionDirective,
    DocSummarizationDirective,
    IsolatingSubtasksDirective,
    DocCompressionDirective,
    DeterministicDocCompressionDirective,
    DocumentChunkingDirective,
    DocumentChunkingTopKDirective,
    ChunkHeaderSummaryDirective,
    TakeHeadTailDirective,
    ClarifyInstructionsDirective,
    SwapWithCodeDirective,
    HierarchicalReduceDirective
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


def test_reduce_chaining_apply():
    """Test that reduce chaining apply doesn't crash"""
    directive = ReduceChainingDirective()
    
    ops_list = [
        {
            "name": "extract_all_locations",
            "type": "reduce",
            "reduce_key": "document_collection",
            "prompt": "Extract all distinct locations mentioned across these documents:\n{% for input in inputs %}\nDocument: {{ input.document }}\n{% endfor %}\nReturn a list of unique location names.",
            "model": "gpt-4o-mini",
            "output": {"schema": {"locations": "list[str]"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import ReduceChainingInstantiateSchema
    rewrite = ReduceChainingInstantiateSchema(
        map_name="extract_document_locations",
        map_prompt="Extract all location names mentioned in this document:\n{{ input.document }}\nReturn a list of locations.",
        new_key="locations",
        modified_reduce_prompt="Combine and deduplicate all locations from these documents:\n{% for input in inputs %}\nLocations from document: {{ input.locations }}\n{% endfor %}\nReturn a list of unique location names.",
        model="gpt-4o-mini"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "extract_all_locations", rewrite)
    assert isinstance(result, list)
    assert len(result) == 2  # Should add map op before reduce op
    assert result[0]["type"] == "map"
    assert result[0]["name"] == "extract_document_locations"
    assert result[1]["type"] == "reduce"
    assert result[1]["name"] == "extract_all_locations"
    # The reduce prompt should reference the new key
    assert "{{ input.locations }}" in result[1]["prompt"]
    # The reduce prompt should NOT reference the original document key
    assert "{{ input.document }}" not in result[1]["prompt"]


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
    
    from docetl.reasoning_optimizer.instantiate_schemas import DocumentChunkingInstantiateSchema, SamplingConfig
    
    # Test without sampling
    rewrite_no_sample = DocumentChunkingInstantiateSchema(
        chunk_size=1000,
        split_key="document",
        sub_prompt="Analyze this document chunk: {{ input.document_chunk_rendered }}",
        reduce_prompt="Combine the analysis results: {% for input in inputs %}{{ input.analysis }}{% endfor %}"
    )
    
    result = directive.apply("azure/gpt-4o-mini", ops_list, "analyze_document", rewrite_no_sample)
    assert isinstance(result, list)
    assert len(result) == 4  # Should be split -> gather -> map -> reduce
    
    # Test with sampling
    rewrite_with_sample = DocumentChunkingInstantiateSchema(
        chunk_size=1000,
        split_key="document",
        sub_prompt="Analyze this document chunk: {{ input.document_chunk_rendered }}",
        reduce_prompt="Combine the analysis results: {% for input in inputs %}{{ input.analysis }}{% endfor %}",
        sampling_config=SamplingConfig(
            method="uniform",
            samples=5
        )
    )
    
    result_with_sample = directive.apply("azure/gpt-4o-mini", ops_list, "analyze_document", rewrite_with_sample)
    assert isinstance(result_with_sample, list)
    assert len(result_with_sample) == 5  # Should be split -> gather -> sample -> map -> reduce
    assert result_with_sample[2]["type"] == "sample"
    assert result_with_sample[2]["method"] == "uniform"
    assert result_with_sample[2]["samples"] == 5


def test_doc_chunking_topk_apply():
    """Test that doc chunking with topk apply doesn't crash"""
    directive = DocumentChunkingTopKDirective()
    
    ops_list = [
        {
            "name": "extract_clinical_findings",
            "type": "map",
            "prompt": """Analyze this clinical trial document and extract safety findings:
                Protocol: {{ input.protocol_id }}
                Document: {{ input.trial_document }}
                
                Extract all adverse events, serious adverse events, and laboratory abnormalities.""",
            "model": "gpt-4o",
            "output": {
                "schema": {
                    "adverse_events": "list[dict]",
                    "serious_adverse_events": "list[dict]",
                    "lab_abnormalities": "list[dict]"
                }
            }
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import (
        DocumentChunkingTopKInstantiateSchema,
        TopKConfig
    )
    
    # Test with embedding-based topk
    rewrite_embedding = DocumentChunkingTopKInstantiateSchema(
        chunk_size=6000,
        split_key="trial_document",
        reduce_prompt="Analyzing clinical trial chunks for protocol {{ inputs[0].protocol_id }}. Extract adverse events, SAEs, and lab abnormalities from each chunk:\n{% for input in inputs %}\nChunk {{ loop.index }}:\n{{ input.trial_document_chunk }}\n{% endfor %}\nCombine all safety findings into structured output.",
        topk_config=TopKConfig(
            method="embedding",
            k=12,
            query="adverse event serious AE SAE laboratory abnormality toxicity safety signal CTCAE grade hospitalization death",
            keys=["trial_document_chunk"],
            embedding_model="text-embedding-3-small"
        ),
        model="gpt-4o"
    )
    
    result = directive.apply("gpt-4o", ops_list, "extract_clinical_findings", rewrite_embedding)
    assert isinstance(result, list)
    assert len(result) == 3  # Should be split -> topk -> reduce
    assert result[1]["type"] == "topk"
    assert result[1]["method"] == "embedding"
    assert result[1]["k"] == 12
    assert "adverse event" in result[1]["query"]
    
    # Test with FTS-based topk
    rewrite_fts = DocumentChunkingTopKInstantiateSchema(
        chunk_size=8000,
        split_key="trial_document",
        reduce_prompt="Extract safety data from clinical trial chunks:\n{% for input in inputs %}\n{{ input.trial_document_chunk }}\n{% endfor %}\nMerge all adverse events, SAEs, and lab abnormalities.",
        topk_config=TopKConfig(
            method="fts",
            k=15,
            query="CTCAE grade 3 grade 4 grade 5 death hospitalization discontinuation dose reduction SAE",
            keys=["trial_document_chunk"]
        ),
        model="gpt-4o-mini"
    )
    
    result_fts = directive.apply("gpt-4o-mini", ops_list, "extract_clinical_findings", rewrite_fts)
    assert isinstance(result_fts, list)
    assert len(result_fts) == 3  # Should be split -> topk -> reduce
    assert result_fts[1]["type"] == "topk"
    assert result_fts[1]["method"] == "fts"
    assert result_fts[1]["k"] == 15
    
    # Test with dynamic Jinja query
    ops_list_dynamic = [
        {
            "name": "extract_condition_history",
            "type": "map",
            "prompt": """Extract medical history for condition: {{ input.target_condition }}
                Patient: {{ input.patient_id }}
                Records: {{ input.medical_records }}""",
            "model": "gpt-4o",
            "output": {"schema": {"condition_history": "list[dict]"}}
        }
    ]
    
    rewrite_dynamic = DocumentChunkingTopKInstantiateSchema(
        chunk_size=5000,
        split_key="medical_records",
        reduce_prompt="Extracting {{ inputs[0].target_condition }} history for patient {{ inputs[0].patient_id }} from chunks:\n{% for input in inputs %}\n{{ input.medical_records_chunk }}\n{% endfor %}\nCompile complete condition history.",
        topk_config=TopKConfig(
            method="embedding",
            k=10,
            query="{{ input.target_condition }} diagnosis treatment medication {{ input.target_condition | lower }}",
            keys=["medical_records_chunk"]
        ),
        model="gpt-4o"
    )
    
    result_dynamic = directive.apply("gpt-4o", ops_list_dynamic, "extract_condition_history", rewrite_dynamic)
    assert isinstance(result_dynamic, list)
    assert "{{ input.target_condition }}" in result_dynamic[1]["query"]


def test_doc_chunking_topk_filter_apply():
    """Test that doc chunking with topk works for filter operations"""
    directive = DocumentChunkingTopKDirective()
    
    # Test filter operation
    ops_list_filter = [
        {
            "name": "filter_competitor_mentions",
            "type": "filter",
            "prompt": """Determine if this lengthy review mentions competitors more positively than our product.
                Our Product: {{ input.our_product }}
                Review: {{ input.review_text }}
                Return true if competitors are mentioned more favorably.""",
            "model": "gpt-4o",
            "output": {
                "schema": {
                    "mentions_competitors_favorably": "bool"
                }
            }
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import (
        DocumentChunkingTopKInstantiateSchema,
        TopKConfig
    )
    
    rewrite_filter = DocumentChunkingTopKInstantiateSchema(
        chunk_size=5000,
        split_key="review_text",
        reduce_prompt="Analyzing review for product {{ inputs[0].our_product }}. Review chunks:\n{% for input in inputs %}\n{{ input.review_text_chunk }}\n{% endfor %}\nDetermine if competitors are mentioned more positively than our product. Output true if yes, false if no.",
        topk_config=TopKConfig(
            method="embedding",
            k=10,
            query="competitor comparison versus alternative better superior inferior worse recommendation",
            keys=["review_text_chunk"]
        ),
        model="gpt-4o"
    )
    
    result_filter = directive.apply("gpt-4o", ops_list_filter, "filter_competitor_mentions", rewrite_filter)
    assert isinstance(result_filter, list)
    assert len(result_filter) == 4  # Should be split -> topk -> reduce -> code_filter
    assert result_filter[0]["type"] == "split"
    assert result_filter[1]["type"] == "topk"
    assert result_filter[2]["type"] == "reduce"
    assert result_filter[3]["type"] == "code_filter"
    assert "code_filter" in result_filter[3]["name"]
    assert "def transform" in result_filter[3]["code"]
    assert "mentions_competitors_favorably" in result_filter[3]["code"]


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
    assert "def transform" in result[0]["code"]
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
    exec(result[0]["code"], namespace)
    transformed_result = namespace["transform"](doc)
    # With 5 head words and 5 tail words, and original has more than 10 words, should be truncated
    assert "content" in transformed_result
    assert len(transformed_result["content"].split()) <= 12  # 5 + " ... " + 5 = at most 12 tokens


def test_clarify_instructions_apply():
    """Test that clarify instructions apply doesn't crash"""
    directive = ClarifyInstructionsDirective()
    
    # Simple pipeline with one map op
    ops_list = [
        {
            "name": "analyze_feedback",
            "type": "map",
            "prompt": "Analyze the feedback: {{ input.feedback }}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"sentiment": "string", "issues": "list[str]"}}
        }
    ]
    
    # Mock rewrite schema
    from docetl.reasoning_optimizer.instantiate_schemas import ClarifyInstructionsInstantiateSchema
    rewrite = ClarifyInstructionsInstantiateSchema(
        clarified_prompt="Analyze the customer feedback in {{ input.feedback }}. Determine the overall sentiment (positive, negative, neutral) and identify specific issues mentioned. Look for complaints about: product quality, delivery, customer service, pricing. Extract concrete problems rather than general dissatisfaction."
    )
    
    result = directive.apply("gpt-4o-mini", ops_list, ["analyze_feedback"], rewrite)
    
    # Should return modified ops list with updated prompt
    assert isinstance(result, list)
    assert len(result) == 1  # Same number of operations
    assert result[0]["name"] == "analyze_feedback"
    assert result[0]["type"] == "map"
    assert result[0]["prompt"] == rewrite.clarified_prompt
    assert "{{ input.feedback }}" in result[0]["prompt"]  # Should preserve input variables
    assert result[0]["model"] == "gpt-4o-mini"  # Should preserve other fields


def test_swap_with_code_apply():
    """Test that swap with code apply doesn't crash"""
    directive = SwapWithCodeDirective()
    
    # Test with reduce operation - no map needed (code reduce outputs correct schema)
    ops_list = [
        {
            "name": "count_locations",
            "type": "reduce",
            "reduce_key": "_all",
            "prompt": "Count unique locations: {% for input in inputs %}{{ input.location }}{% endfor %}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"unique_count": "int", "locations": "list[str]"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import SwapWithCodeInstantiateSchema
    rewrite_no_map = SwapWithCodeInstantiateSchema(
        code_reduce_name="code_count_locations",
        code='''def transform(inputs):
    locations = set()
    for item in inputs:
        if "location" in item:
            locations.add(item["location"])
    return {"unique_count": len(locations), "locations": sorted(list(locations))}''',
        map_prompt=None  # No map needed
    )
    
    result = directive.apply("gpt-4o-mini", ops_list, ["count_locations"], rewrite_no_map)
    assert isinstance(result, list)
    assert len(result) == 1  # Should replace reduce with code_reduce only
    assert result[0]["name"] == "count_locations"  # Should preserve original name
    assert result[0]["type"] == "code_reduce"
    assert "code" in result[0]
    assert result[0]["reduce_key"] == "_all"
    
    # Test with reduce operation + map needed (code reduce needs formatting)
    ops_list_with_map = [
        {
            "name": "summarize_locations", 
            "type": "reduce",
            "reduce_key": "_all",
            "prompt": "Summarize all locations: {% for input in inputs %}{{ input.location }}{% endfor %}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"summary": "str", "count": "int"}}
        }
    ]
    
    rewrite_with_map = SwapWithCodeInstantiateSchema(
        code_reduce_name="collect_locations",
        code='''def transform(inputs):
    locations = []
    for item in inputs:
        if "location" in item:
            locations.append(item["location"])
    return {"all_locations": locations}''',
        map_prompt="Create a summary from these locations {{ input.all_locations }}. Output: summary (descriptive text), count (number of locations)."
    )
    
    result_with_map = directive.apply("gpt-4o-mini", ops_list_with_map, ["summarize_locations"], rewrite_with_map)
    assert isinstance(result_with_map, list)
    assert len(result_with_map) == 2  # Should have code_reduce + map
    
    # Check code reduce operation
    assert result_with_map[0]["name"] == "collect_locations"
    assert result_with_map[0]["type"] == "code_reduce"
    assert "code" in result_with_map[0]
    assert result_with_map[0]["reduce_key"] == "_all"
    
    # Check map operation
    assert result_with_map[1]["name"] == "summarize_locations"  # Should preserve original name
    assert result_with_map[1]["type"] == "map"
    assert "prompt" in result_with_map[1]
    assert result_with_map[1]["model"] == "gpt-4o-mini"
    assert "output" in result_with_map[1]


def test_hierarchical_reduce_apply():
    """Test that hierarchical reduce apply doesn't crash"""
    directive = HierarchicalReduceDirective()
    
    # Test with Map for synthetic key
    ops_list = [
        {
            "name": "summarize_by_state",
            "type": "reduce",
            "reduce_key": "state",
            "prompt": "Summarize posts: {% for input in inputs %}{{ input.content }}{% endfor %}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"summary": "string"}}
        }
    ]
    
    from docetl.reasoning_optimizer.instantiate_schemas import HierarchicalReduceInstantiateSchema, MapOpConfig
    
    # Test with Map for synthetic key
    rewrite_with_map = HierarchicalReduceInstantiateSchema(
        map_config=MapOpConfig(
            name="extract_city",
            prompt="Extract city from: {{ input.content }}",
            output_keys=["city"],
            model="gpt-4o-mini"
        ),
        additional_key="city",
        reduce_1_name="summarize_by_state_city",
        reduce_1_prompt="Goal: Summarize social media posts.\n\nSummarize posts for this state and city: {% for input in inputs %}{{ input.content }}{% endfor %}",
        reduce_2_prompt="Goal: Summarize social media posts.\n\nWe have already summarized posts at the city level. Your task is to combine these city-level summaries into a state summary: {% for input in inputs %}City: {{ input.city }}, Summary: {{ input.summary }}{% endfor %}",
        model="gpt-4o-mini"
    )
    
    result_with_map = directive.apply("azure/gpt-4o-mini", ops_list, "summarize_by_state", rewrite_with_map)
    assert isinstance(result_with_map, list)
    assert len(result_with_map) == 3  # Map + 2 Reduces
    assert result_with_map[0]["type"] == "map"
    assert result_with_map[1]["type"] == "reduce"
    assert result_with_map[2]["type"] == "reduce"
    
    # Test without Map (using existing key)
    ops_list_existing_key = [
        {
            "name": "aggregate_sales",
            "type": "reduce",
            "reduce_key": "region",
            "prompt": "Aggregate sales: {% for input in inputs %}{{ input.sales_data }}{% endfor %}",
            "model": "gpt-4o-mini",
            "output": {"schema": {"total": "number"}}
        }
    ]
    
    rewrite_no_map = HierarchicalReduceInstantiateSchema(
        map_config=None,
        additional_key="store_id",  # Assume this exists in data
        reduce_1_name="aggregate_by_region_store",
        reduce_1_prompt="Goal: Aggregate sales data.\n\nAggregate sales by region and store: {% for input in inputs %}{{ input.sales_data }}{% endfor %}",
        reduce_2_prompt="Goal: Aggregate sales data.\n\nWe have already aggregated sales at the store level. Your task is to combine these store-level totals into a regional summary: {% for input in inputs %}Store: {{ input.store_id }}, Total: {{ input.total }}{% endfor %}",
        model="gpt-4o-mini"
    )
    
    result_no_map = directive.apply("azure/gpt-4o-mini", ops_list_existing_key, "aggregate_sales", rewrite_no_map)
    assert isinstance(result_no_map, list)
    assert len(result_no_map) == 2  # 2 Reduces only
    assert result_no_map[0]["type"] == "reduce"
    assert result_no_map[1]["type"] == "reduce"


def test_cascade_filtering_apply():
    """Test that cascade filtering apply correctly creates filter cascade"""
    from docetl.reasoning_optimizer.directives import CascadeFilteringDirective
    from docetl.reasoning_optimizer.instantiate_schemas import (
        CascadeFilteringInstantiateSchema,
        CodePreFilter,
        LLMPreFilter,
    )
    
    directive = CascadeFilteringDirective()
    
    ops_list = [
        {
            "name": "other_op",
            "type": "map",
            "prompt": "Do something",
            "output": {"schema": {"result": "string"}}
        },
        {
            "name": "filter_quality",
            "type": "filter",
            "model": "gpt-4o",
            "prompt": "Is this a high-quality research paper? Paper: {{ input.paper_text }}",
            "output": {"schema": {"is_quality": "boolean"}}
        }
    ]
    
    # Create test rewrite with both code and LLM pre-filters
    rewrite = CascadeFilteringInstantiateSchema(
        code_pre_filters=[
            CodePreFilter(
                name="filter_min_length",
                code="def transform(input_doc):\n    text = input_doc.get('paper_text', '')\n    return len(text.split()) >= 500",
                reasoning="Papers under 500 words are rarely high quality"
            ),
            CodePreFilter(
                name="filter_references",
                code="def transform(input_doc):\n    text = input_doc.get('paper_text', '').lower()\n    return 'references' in text or 'bibliography' in text",
                reasoning="Quality papers have references section"
            ),
        ],
        llm_pre_filters=[
            LLMPreFilter(
                name="quick_research_check",
                prompt="Is this research? {{ input.paper_text }} Answer yes/no.",
                reasoning="Non-research papers are filtered out"
            ),
            LLMPreFilter(
                name="academic_check",
                prompt="Does this paper have academic structure (abstract, methods, results)? {{ input.paper_text }} Answer yes or no.",
                reasoning="Papers without academic structure are not high quality"
            ),
        ],
        analysis_summary="Identified patterns to filter 70% of documents with cheap methods"
    )
    
    result = directive.apply("gpt-4o", ops_list, ["filter_quality"], rewrite)
    
    # Check structure: other_op, 2 code filters, 2 LLM filters (sorted by prompt length), original filter
    assert len(result) == 6
    
    # First op should be unchanged
    assert result[0]["name"] == "other_op"
    assert result[0]["type"] == "map"
    
    # Next should be code filters
    assert result[1]["type"] == "code_filter"
    assert result[1]["name"] == "filter_min_length_filter_quality"
    assert "def transform" in result[1]["code"]
    
    assert result[2]["type"] == "code_filter"
    assert result[2]["name"] == "filter_references_filter_quality"
    
    # Then LLM filters (sorted by prompt length)
    assert result[3]["type"] == "filter"
    assert result[3]["model"] == "gpt-5-nano"
    # Shorter prompt should come first
    assert "research?" in result[3]["prompt"] or "academic structure" in result[3]["prompt"]
    
    assert result[4]["type"] == "filter"
    assert result[4]["model"] == "gpt-5-nano"
    
    # Finally the original filter
    assert result[5]["name"] == "filter_quality"
    assert result[5]["type"] == "filter"
    assert result[5]["model"] == "gpt-4o"
    assert "high-quality research paper" in result[5]["prompt"]


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
    
    test_doc_summarization_apply()
    print("✅ Doc summarization apply test passed")
    
    test_isolating_subtasks_apply()
    print("✅ Isolating subtasks apply test passed")
    
    test_reduce_gleaning_apply()
    print("✅ Reduce gleaning apply test passed")
    
    test_reduce_chaining_apply()
    print("✅ Reduce chaining apply test passed")
    
    test_doc_compression_apply()
    print("✅ Doc compression apply test passed")
    
    test_deterministic_doc_compression_apply()
    print("✅ Deterministic doc compression apply test passed")
    
    test_doc_chunking_apply()
    print("✅ Doc chunking apply test passed")
    
    test_doc_chunking_topk_apply()
    print("✅ Doc chunking topk apply test passed")
    
    test_chunk_header_summary_apply()
    print("✅ Chunk header summary apply test passed")


    test_take_head_tail_apply()
    print("✅ Take head tail apply test passed")

    test_clarify_instructions_apply()
    print("✅ Clarify instructions apply test passed")

    test_swap_with_code_apply()
    print("✅ Swap with code apply test passed")

    test_hierarchical_reduce_apply()
    print("✅ Hierarchical reduce apply test passed")
    
    test_cascade_filtering_apply()
    print("✅ Cascade filtering apply test passed")

    print("\n🎉 All directive apply tests passed!")
