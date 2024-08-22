from typing import Any, Dict, List, Optional


class OperationCreator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_parallel_map_operation(
        self, op_config: Dict[str, Any], subtasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        parallel_map_op = {
            "type": "parallel_map",
            "name": f"{op_config['name']}_parallel_map",
            "prompts": [],
            "output": op_config["output"],
            "model": op_config.get("model", self.config["default_model"]),
        }

        for subtask in subtasks:
            parallel_map_op["prompts"].append(
                {
                    "name": subtask["name"],
                    "prompt": subtask["prompt"],
                    "output_keys": subtask["output_keys"],
                }
            )

        return parallel_map_op

    def create_metadata_operation(
        self,
        op_config: Dict[str, Any],
        metadata_prompt: str,
        output_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "type": "map",
            "name": f"extract_metadata_{op_config['name']}",
            "prompt": metadata_prompt,
            "model": self.config["default_model"],
            "output": {"schema": output_schema},
        }

    def create_split_map_gather_operations(
        self,
        op_config: Dict[str, Any],
        chunk_info: Dict[str, Any],
        context_info: Dict[str, Any],
        split_key: str,
        content_key: str,
        summary_prompt: Optional[str] = None,
        summary_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pipeline = []
        chunk_size = int(chunk_info["chunk_size"] * 1.5)
        split_name = f"split_{op_config['name']}"
        split_config = {
            "type": "split",
            "name": split_name,
            "split_key": split_key,
            "chunk_size": chunk_size,
        }
        pipeline.append(split_config)

        # If there's a summary prompt, create a map config
        if summary_prompt:
            pipeline.append(
                {
                    "type": "map",
                    "name": f"summary_{split_key}_{op_config['name']}",
                    "prompt": summary_prompt,
                    "model": summary_model,
                    "output": {"schema": {f"{split_key}_summary": "string"}},
                }
            )

        gather_config = {
            "type": "gather",
            "name": f"gather_{split_key}_{op_config['name']}",
            "content_key": content_key,
            "doc_id_key": f"{split_name}_id",
            "order_key": f"{split_name}_chunk_num",
            "peripheral_chunks": {},
        }

        if "previous" in context_info:
            gather_config["peripheral_chunks"]["previous"] = context_info["previous"]

        if "next" in context_info:
            gather_config["peripheral_chunks"]["next"] = context_info["next"]

        # Add gather to the pipeline if there are peripheral chunks
        if gather_config["peripheral_chunks"]:
            pipeline.append(gather_config)

        return pipeline

    def create_map_operation(
        self, op_config: Dict[str, Any], subprompt: str
    ) -> Dict[str, Any]:
        name = f"sub{op_config['type']}_{op_config['name']}"
        output = op_config["output"]
        if op_config["type"] == "filter":
            output["schema"]["_short_explanation"] = "string"

        return {
            "type": op_config["type"],
            "name": name,
            "prompt": subprompt,
            "model": (
                op_config["model"]
                if "model" in op_config
                else self.config["default_model"]
            ),
            "output": output,
        }

    def create_unnest_operations(
        self, op_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Check if the output schema has a list type key and create an unnest operation for it
        output_list_keys = [
            key
            for key, value in op_config.get("output", {}).get("schema", {}).items()
            if isinstance(value, str) and value.startswith("list[")
        ]

        # Create an unnest operation for each list type key
        unnest_ops = []
        for unnest_key in output_list_keys:
            unnest_ops.append(
                {
                    "type": "unnest",
                    "name": f"unnest_{unnest_key}_{op_config['name']}",
                    "unnest_key": unnest_key,
                    "keep_empty": True,
                }
            )

        return unnest_ops

    def create_reduce_operation(
        self,
        op_config: Dict[str, Any],
        combine_prompt: str,
        is_commutative: bool,
        doc_id_key: str,
    ) -> Dict[str, Any]:
        name = f"subreduce_{op_config['name']}"
        return {
            "type": "reduce",
            "name": name,
            "reduce_key": doc_id_key,
            "input": op_config["output"],  # subselect keys
            "prompt": combine_prompt,
            "model": (
                op_config["model"]
                if "model" in op_config
                else self.config["default_model"]
            ),
            "output": op_config["output"],
            "pass_through": True,
            "commutative": is_commutative,
            "_intermediates": {"last_map_prompt": op_config["prompt"]},
        }
