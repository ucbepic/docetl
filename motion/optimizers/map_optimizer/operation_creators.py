from typing import Dict, Any, List


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

    def create_split_operation(
        self,
        op_config: Dict[str, Any],
        chunk_info: Dict[str, Any],
        context_info: Dict[str, Any],
        split_key: str,
        summary_prompt: str,
        summary_model: str,
    ) -> Dict[str, Any]:
        chunk_size = int(chunk_info["chunk_size"] * 1.5)
        name = f"split_{op_config['name']}"
        split_config = {
            "type": "split",
            "name": name,
            "split_key": split_key,
            "chunk_group_id_field": f"{op_config['name']}_chunk_group_id",
            "chunk_size": chunk_size,
            "peripheral_chunks": {},
            "summary_prompt": summary_prompt,
            "summary_model": summary_model,
        }

        if "previous" in context_info:
            split_config["peripheral_chunks"]["previous"] = context_info["previous"]

        if "next" in context_info:
            split_config["peripheral_chunks"]["next"] = context_info["next"]

        # Remove peripheral_chunks if it's empty
        if not split_config["peripheral_chunks"]:
            del split_config["peripheral_chunks"]

        return split_config

    def create_map_operation(
        self, op_config: Dict[str, Any], subprompt: str
    ) -> Dict[str, Any]:
        name = f"submap_{op_config['name']}"
        return {
            "type": "map",
            "name": name,
            "prompt": subprompt,
            "model": (
                op_config["model"]
                if "model" in op_config
                else self.config["default_model"]
            ),
            "output": op_config["output"],
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
        self, op_config: Dict[str, Any], combine_prompt: str, is_commutative: bool
    ) -> Dict[str, Any]:
        name = f"subreduce_{op_config['name']}"
        return {
            "type": "reduce",
            "name": name,
            "reduce_key": f"{op_config['name']}_chunk_group_id",
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
        }
