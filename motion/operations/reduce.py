from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from jinja2 import Template
from itertools import groupby
from operator import itemgetter
from motion.operations.base import BaseOperation
from motion.operations.utils import call_llm, parse_llm_response
from motion.operations.utils import validate_output
from litellm import completion_cost
import jinja2


class ReduceOperation(BaseOperation):
    def syntax_check(self) -> None:
        required_keys = ["reduce_key", "prompt", "output"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required key '{key}' in ReduceOperation configuration"
                )

        if "schema" not in self.config["output"]:
            raise ValueError("Missing 'schema' in 'output' configuration")

        if not isinstance(self.config["output"]["schema"], dict):
            raise TypeError("'schema' in 'output' configuration must be a dictionary")

        if not self.config["output"]["schema"]:
            raise ValueError("'schema' in 'output' configuration cannot be empty")

        # Check if the prompt is a valid Jinja2 template
        try:
            template = Template(self.config["prompt"])
            template_vars = template.environment.parse(self.config["prompt"]).find_all(
                jinja2.nodes.Name
            )
            template_var_names = {var.name for var in template_vars}
            if "values" not in template_var_names:
                raise ValueError("Template must include the 'values' variable")
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template in 'prompt': {str(e)}")

        # Check if the model is specified (optional)
        if "model" in self.config and not isinstance(self.config["model"], str):
            raise TypeError("'model' in configuration must be a string")

        # Check if reduce_key is a string
        if not isinstance(self.config["reduce_key"], str):
            raise TypeError("'reduce_key' must be a string")

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        reduce_key = self.config["reduce_key"]
        sorted_data = sorted(input_data, key=itemgetter(reduce_key))
        grouped_data = groupby(sorted_data, key=itemgetter(reduce_key))

        def process_group(key: Any, group: List[Dict]) -> Tuple[Optional[Dict], float]:
            group_list = list(group)
            prompt_template = Template(self.config["prompt"])
            prompt = prompt_template.render(reduce_key=key, values=group_list)
            response = call_llm(
                self.config.get("model", self.default_model),
                "reduce",
                prompt,
                self.config["output"]["schema"],
            )
            output = parse_llm_response(response)[0]
            output[reduce_key] = key
            item_cost = completion_cost(response)
            if validate_output(self.config, output, self.console):
                return output, item_cost
            return None, item_cost

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_group, key, group)
                for key, group in grouped_data
            ]
            results = []
            total_cost = 0
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing reduce items",
                leave=True,
            ):
                output, item_cost = future.result()
                total_cost += item_cost
                if output is not None:
                    results.append(output)

        return results, total_cost
