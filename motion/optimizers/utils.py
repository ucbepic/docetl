from typing import Dict, List, Any, Optional, Tuple, Union
import json
from litellm import completion, completion_cost
import os
import jinja2
from jinja2 import Environment, meta
import re


def extract_jinja_variables(template_string):
    # Create a Jinja2 environment
    env = Environment()

    # Parse the template
    ast = env.parse(template_string)

    # Find all the variables referenced in the template
    variables = meta.find_undeclared_variables(ast)

    # Use regex to find any additional variables that might be missed
    # This regex looks for {{ variable }} patterns
    regex_variables = set(re.findall(r"{{\s*(\w+)\s*}}", template_string))

    # Combine both sets of variables
    all_variables = variables.union(regex_variables)

    return list(all_variables)


SUPPORTED_OPS = ["map"]


class LLMClient:
    def __init__(self, model="gpt-4o"):
        if model == "gpt-4o":
            model = "gpt-4o-2024-08-06"
        self.model = model
        self.total_cost = 0

    def generate(self, messages, system_prompt, parameters):
        parameters["additionalProperties"] = False

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *messages,
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": parameters,
                },
            },
        )
        cost = completion_cost(response)
        self.total_cost += cost
        return response
