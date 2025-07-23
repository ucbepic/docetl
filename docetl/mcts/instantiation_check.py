import os
from typing import Any, Dict

import litellm
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def load_yaml(yaml_file_path) -> Dict[str, Any]:
    """
    Load and parse the YAML file.

    Returns:
        Parsed YAML content as a dictionary
    """
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return {}


# new_yaml_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/MCTS/execute_res/CUAD-map_2.yaml"
# new_config = load_yaml(new_yaml_path)


def llm_judge_chaining(orig_config, new_config):
    user_message = f"""
        Compare the original prompt configuration with the new decomposed version to verify that ALL specifications are preserved. You only need to focus on the "prompt" field.

        **Original Configuration:**
        {orig_config}

        **New Decomposed Configuration:**
        {new_config}

        **Evaluation Focus:**
        - Are all categories from the original prompt explicitly listed in the new configuration? (Note: It's acceptable for categories to be distributed across different operations, but the complete pipeline must include ALL categories exactly)
        - Can each step access the information it needs?
        - Does the pipeline as a whole preserve all specifications from the original?

        Reasoning about what's missing or incorrect and provide a boolean pass/fail decision.
        """

    messages = [
        {
            "role": "system",
            "content": "You are an expert judge evaluating whether a newly generated prompt configuration makes logical sense for contract analysis tasks. Your primary focus is to identify critical issues that would prevent the LLM from performing well. Your output must follow the structured output format.",
        },
        {"role": "user", "content": user_message},
    ]

    class ResponseFormat(BaseModel):
        pass_test: bool
        reason: str

    response = litellm.completion(
        model="gpt-4.1",
        messages=messages,
        api_key=os.environ.get("AZURE_API_KEY"),
        api_base=os.environ.get("AZURE_API_BASE"),
        api_version=os.environ.get("AZURE_API_VERSION"),
        azure=True,
        response_format=ResponseFormat,
    )

    print(response.choices[0].message.content)


orig_yaml_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/MCTS/execute_res/CUAD-map.yaml"
orig_config = load_yaml(orig_yaml_path)
orig_config = orig_config["operations"]

# chaining = ChainingDirective()
# new_ops_list, message_history = chaining.instantiate(operators = orig_config, target_ops = ["extract_contract_info"], agent_llm = "gpt-4.1")


# # Dump new_ops_list to a YAML file
# with open("new_ops_list_3-1.yaml", "w", encoding="utf-8") as f:
#     yaml.dump(new_ops_list, f, allow_unicode=True, sort_keys=False)


new_yaml_path = "/Users/lindseywei/Documents/DocETL-optimizer/reasoning-optimizer/MCTS/new_ops_list_3-1.yaml"
new_ops_list = load_yaml(new_yaml_path)

print("_______________________________________________")
llm_judge_chaining(orig_config, new_ops_list)
