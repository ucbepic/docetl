# NOTE: This function is structured as if it will be a method of PlanGenerator in plan_generators.py.
# For now, it is a standalone function in this file, but 'self' is used for future compatibility.

import copy
from typing import Dict, List, Any, Tuple
from docetl.optimizers.reduce_optimizer import ReduceOptimizer

def _generate_single_chunk_size_plan(
    self,
    op_config: Dict[str, Any],
    input_data: List[Dict[str, Any]],
    user_chunk_size: int,
    user_peripheral_config_tuple: Tuple[Dict[str, Any], bool],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate a single plan for a user-specified chunk size and user-specified peripheral config.
    """
    split_result = self.config_generator._get_split_config(op_config, input_data)
    split_key = split_result["split_key"]
    content_key = f"{split_key}_chunk"
    summary_key = f"{split_key}_summary"
    doc_id_key = f"split_{op_config['name']}_id"
    subprompt_output_schema = split_result.get("subprompt_output_schema", {})
    if not subprompt_output_schema:
        subprompt_output_schema = op_config["output"]["schema"]
    split_subprompt = split_result["subprompt"]

 
    chunk_size = user_chunk_size

    avg_doc_size = sum(len(doc[split_key].split()) for doc in input_data) // len(input_data)
    avg_chunk_size = chunk_size

    def determine_metadata_with_retry():
        try:
            metadata_info = self.config_generator._determine_metadata_needs(
                op_config,
                split_subprompt,
                avg_chunk_size,
                split_key,
                input_data,
            )
            return metadata_info
        except Exception as e:
            self.console.log(
                f"[yellow]Error determining metadata needs: {e}. Retrying...[/yellow]"
            )
            try:
                # Retry once
                return self.config_generator._determine_metadata_needs(
                    op_config,
                    split_subprompt,
                    avg_chunk_size,
                    split_key,
                    input_data,
                )
            except Exception:
                # Silently fail on second attempt
                return {"needs_metadata": False}

    metadata_info = determine_metadata_with_retry()
    self.console.log(f"Needs metadata: {metadata_info['needs_metadata']}")
    if metadata_info["needs_metadata"]:
        self.console.log(
            f"Metadata prompt and output schema: {metadata_info.get('metadata_prompt', 'N/A')}; {metadata_info.get('output_schema', 'N/A')}"
        )
        self.console.log(f"Reason: {metadata_info.get('reason', 'N/A')}")
        split_subprompt = (
            "Given the following metadata about the document:\n{{ input.metadata }}\n\n"
            + split_subprompt
        )

    # Header extraction prompt
    header_extraction_prompt, header_output_schema = (self.prompt_generator._get_header_extraction_prompt(
        op_config, input_data, split_key)
    )
    if header_extraction_prompt:
        self.console.log(
            f"Inferring headers from the documents. Will apply this prompt to find headers in chunks: {header_extraction_prompt}"
        )
    else:
        self.console.log(
            "Not inferring headers from the documents. Will not apply any header extraction prompt."
        )

    # Create base operations
    base_operations = []
    if metadata_info["needs_metadata"]:
        base_operations.append(
            self.operation_creator.create_metadata_operation(
                op_config,
                metadata_info["metadata_prompt"],
                metadata_info["output_schema"],
            )
        )

    # Info extraction prompt
    peripheral_config, needs_summary = user_peripheral_config_tuple
    chunk_word_size = int(chunk_size / 2.5)
    
    largest_input = max(input_data, key=lambda x: len(x[split_key].split()))
    words = largest_input[split_key].split()
    sample_chunks = [
        words[i : i + chunk_word_size]
        for i in range(0, len(words), chunk_word_size)
    ]
    if not sample_chunks or len(sample_chunks) < 2:
        raise ValueError("Not enough words in input data to generate sample chunks")
    info_extraction_prompt = self.generate_info_extraction_prompt(
        split_subprompt, split_key, sample_chunks[0], sample_chunks[1]
    )
    self.console.log(
        "[bold]Info Extraction Prompt (Used to Summarize Peripheral Chunks):[/bold]"
    )
    self.console.log(info_extraction_prompt)

    # Synthesize the reduce operation
    sample_output = copy.deepcopy(input_data)
    max_plan = copy.deepcopy(base_operations)

    smg_ops = self.operation_creator.create_split_map_gather_operations(
        op_config,
        {"chunk_size": chunk_size},
        peripheral_config,
        split_key,
        content_key,
        info_extraction_prompt if needs_summary else None,
        "gpt-4o-mini",
        header_extraction_prompt,
        header_output_schema,
    )
    map_op = self.operation_creator.create_map_operation(
        op_config,
        subprompt_output_schema,
        split_subprompt,
    )
    max_plan.extend(smg_ops)

    sample_map_input = copy.deepcopy(input_data)
    for smg_op in max_plan:
        sample_map_input = self._run_operation(smg_op, sample_map_input)

    sample_output = self._run_operation(map_op, sample_map_input, is_build=True)
    max_plan.append(map_op)

    # Generate the combine prompt using the sample output
    combine_prompt, is_associative = self.prompt_generator._get_combine_prompt(
        op_config, sample_output
    )
    self.console.log("[bold]Combine Prompt:[/bold]")
    self.console.log(combine_prompt)

    # Create the reduce operation
    reduce_op = self.operation_creator.create_reduce_operation(
        op_config, combine_prompt, is_associative, doc_id_key
    )

    # Assume no recursive optimization of the map and reduce operations

    # Build the final plan
    plan = copy.deepcopy(base_operations)
    plan.extend(smg_ops + [map_op] + [reduce_op])
    return {f"chunk_size_{chunk_size}_plan": plan} 