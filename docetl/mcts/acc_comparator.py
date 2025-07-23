import json
import os
import random

import litellm
import yaml
from pydantic import BaseModel

from .Node import Node


class CompareResponseFormat(BaseModel):
    score: int
    reason: str


class AccuracyComparator:
    """
    LLM-based plan accuracy comparator
    """

    COMPARISON_SCORES = {
        -3: "plan1 is much worse than plan2",
        -1: "plan1 is slightly worse than plan2",
        0: "both plans are roughly equivalent",
        1: "plan1 is slightly better than plan2",
        3: "plan1 is much better than plan2",
    }

    def __init__(self, input_data, max_retries=3, model="gpt-4.1"):
        """
        Initialize LLM comparator

        Args:
            input_data: sample input data
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
        self.input_data = input_data
        self.model = model

    def compare(self, node1: Node, node2: Node) -> float:
        """
        Use LLM to compare the accuracy of two query plans

        Args:
            node1: First Node object containing the plan
            node2: Second Node object containing the plan

        Returns:
            float: Comparison score (-3, -1, 0, 1, 3)
        """
        try:
            system_prompt, user_prompt = self._build_comparison_prompt(
                self.input_data,
                node1.parsed_yaml,
                node1.sample_result,
                node2.parsed_yaml,
                node2.sample_result,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            # Call LLM and parse resultsta
            for attempt in range(self.max_retries):
                try:
                    response = litellm.completion(
                        model=self.model,
                        messages=messages,
                        api_key=os.environ.get("AZURE_API_KEY"),
                        api_base=os.environ.get("AZURE_API_BASE"),
                        api_version=os.environ.get("AZURE_API_VERSION"),
                        azure=True,
                        response_format=CompareResponseFormat,
                    )
                    reply = response.choices[0].message.content
                    parsed = json.loads(reply)
                    score = parsed.get("score")
                    print("COMPARATOR SCORE: ", score)
                    return float(score)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(
                            f"LLM comparison failed after {self.max_retries} attempts: {e}"
                        )
                        return 0.0  # Default to equal if all attempts fail
                    continue
            return 0.0

        except Exception as e:
            print(
                f"Error comparing plans {node1.yaml_file_path} vs {node2.yaml_file_path}: {e}"
            )
            return 0.0

    def sample_plan_results(self, plan_result, num_samples=5):
        """
        Sample up to num_samples entries from plan_result and keep only filename and clauses.
        """
        sampled = random.sample(plan_result, min(num_samples, len(plan_result)))
        sampled_simple = []
        for entry in sampled:
            sample = {
                "filename": entry.get("name") or entry.get("filename"),
                "clauses": entry.get("clauses", []),
            }
            sampled_simple.append(sample)
        return sampled_simple

    def _build_comparison_prompt(
        self, input_data, plan1_content, plan1_result, plan2_content, plan2_result
    ):
        """
        Build LLM comparison prompt
        """

        num_samples = 5
        sampled_plan1_result = self.sample_plan_results(plan1_result, num_samples)
        sampled_plan2_result = self.sample_plan_results(plan2_result, num_samples)

        system_prompt = "You are an AI assistant tasked with comparing the outputs of two query plans for a same document processing task."
        user_prompt = f"""
            Please compare the accuracy of the following two query plans given their output.

            ## Evaluation Criteria
            Please evaluate the relative quality of the two plans.

            ## Plan 1:
            ```yaml
            {yaml.dump(plan1_content, default_flow_style=False, allow_unicode=True)}
            ```

            ## Plan 2:
            ```yaml
            {yaml.dump(plan2_content, default_flow_style=False, allow_unicode=True)}
            ```

            ## Sample input:
            {json.dumps(input_data, indent=2)[:5000]}

            Compare the outputs of two plans for this input:

            ## Plan 1 output:
            {json.dumps(sampled_plan1_result, indent=2)}

            ## Plan 2 output:
            {json.dumps(sampled_plan2_result, indent=2)}

            ## Comparison Requirements
            Please compare Plan 1's accuracy relative to Plan 2 and give one of the following scores:

            - **-3**: Plan 1 is much worse than Plan 2
            - **-1**: Plan 1 is slightly worse than Plan 2
            - **0**: Both plans are roughly equivalent
            - **1**: Plan 1 is slightly better than Plan 2
            - **3**: Plan 1 is much better than Plan 2
            """
        return system_prompt, user_prompt
