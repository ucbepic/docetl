import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from litellm import model_cost

from rich.console import Console

from docetl.optimizers.utils import LLMClient, extract_jinja_variables
from docetl.utils import truncate_sample_data, count_tokens


class Evaluator:
    def __init__(
        self,
        llm_client: LLMClient,
        console: Console,
        run_operation: Callable[
            [Dict[str, Any], List[Dict[str, Any]]], List[Dict[str, Any]]
        ],
        timeout: int = 60,
        num_plans_to_evaluate_in_parallel: int = 10,
        is_filter: bool = False,
    ):
        self.llm_client = llm_client
        self.console = console
        self._run_operation = run_operation
        self.timeout = timeout
        self.num_plans_to_evaluate_in_parallel = num_plans_to_evaluate_in_parallel
        self.is_filter = is_filter

    def _pairwise_compare_plans(
        self,
        filtered_results: Dict[str, Tuple[float, float, List[Dict[str, Any]]]],
        validator_prompt: str,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        plan_names = list(filtered_results.keys())
        rankings = {plan: 0 for plan in plan_names}
        overall_prompt = op_config["prompt"]

        with ThreadPoolExecutor(
            max_workers=self.num_plans_to_evaluate_in_parallel
        ) as executor:
            futures = {}
            for i in range(len(plan_names)):
                for j in range(i + 1, len(plan_names)):
                    plan1, plan2 = plan_names[i], plan_names[j]
                    future = executor.submit(
                        self._compare_two_plans,
                        overall_prompt,
                        plan1,
                        filtered_results[plan1][2],
                        plan2,
                        filtered_results[plan2][2],
                        validator_prompt,
                        op_config,
                        input_data,
                    )
                    futures[(plan1, plan2)] = future

            for (plan1, plan2), future in futures.items():
                try:
                    comparison_result = future.result()
                    if comparison_result == plan1:
                        rankings[plan1] += 1
                    elif comparison_result == plan2:
                        rankings[plan2] += 1
                    # If comparison_result is None, it's a tie, so we don't update rankings
                except Exception as e:
                    self.console.log(
                        f"[red]Error comparing {plan1} and {plan2}: {str(e)}[/red]"
                    )

        return rankings

    def _compare_two_plans(
        self,
        overall_prompt: str,
        plan1_name: str,
        plan1_output: List[Dict[str, Any]],
        plan2_name: str,
        plan2_output: List[Dict[str, Any]],
        validator_prompt: str,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
    ) -> Optional[str]:
        system_prompt = "You are an AI assistant tasked with comparing the outputs of two ways to complete a task."

        comparisons = []
        for i in range(min(len(plan1_output), len(plan2_output), len(input_data))):
            # Extract variables from the overall prompt template using extract_jinja_variables
            variables_in_prompt = extract_jinja_variables(overall_prompt)
            variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]

            # Filter input_data to only include relevant variables
            filtered_input = {
                k: v for k, v in input_data[i].items() if k in variables_in_prompt
            }

            output_vars = op_config.get("output", {}).get("schema", {}).keys()

            filtered_plan1_output = {
                k: v for k, v in plan1_output[i].items() if k in output_vars
            }
            filtered_plan2_output = {
                k: v for k, v in plan2_output[i].items() if k in output_vars
            }

            prompt = f"""
            Overall Task Prompt:
            {overall_prompt}

            Validator Prompt:
            {validator_prompt}

            Input:
            {json.dumps(filtered_input, indent=2)}

            Compare the outputs of two plans for this input:

            Plan 1 output:
            {json.dumps(filtered_plan1_output, indent=2)}

            Plan 2 output:
            {json.dumps(filtered_plan2_output, indent=2)}

            Based on the overall task prompt, validator prompt, input, and these outputs, which plan performed better for this specific input?
            If one plan is clearly superior, return either "plan_1" or "plan_2". If they are roughly equivalent, return "tie".

            Provide your response in the following format:
            """

            parameters = {
                "type": "object",
                "properties": {
                    "better_plan": {
                        "type": "string",
                        "enum": ["plan_1", "plan_2", "tie"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["better_plan", "reason"],
            }

            response = self.llm_client.generate(
                [{"role": "user", "content": prompt}],
                system_prompt,
                parameters,
            )
            result = json.loads(response.choices[0].message.content)
            comparisons.append(result)

        # Aggregate results
        plan1_wins = sum(1 for comp in comparisons if comp["better_plan"] == "plan_1")
        plan2_wins = sum(1 for comp in comparisons if comp["better_plan"] == "plan_2")
        ties = sum(1 for comp in comparisons if comp["better_plan"] == "tie")

        comparison_details = "\n".join(
            [
                f"Input {i+1}: \"{comp['better_plan']}\" because {comp['reason']}"
                for i, comp in enumerate(comparisons)
            ]
        )

        self.console.log(
            f"[bold magenta]Pairwise Comparison: {plan1_name} vs {plan2_name}[/bold magenta]\n"
            f"[cyan]{plan1_name} wins: {plan1_wins}[/cyan]\n"
            f"[green]{plan2_name} wins: {plan2_wins}[/green]\n"
            f"[yellow]Ties: {ties}[/yellow]\n\n"
            f"Comparison Details:\n{comparison_details}"
        )

        if plan1_wins > plan2_wins:
            return plan1_name
        elif plan2_wins > plan1_wins:
            return plan2_name
        else:
            return None  # Tie

    def _evaluate_plan(
        self,
        plan_name: str,
        op_config: Dict[str, Any],
        plan: Union[Dict[str, Any], List[Dict[str, Any]]],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Evaluate a single optimization plan.

        This method executes the given plan on the input data, measures its runtime,
        and assesses the quality of the output using a custom validator prompt.

        Args:
            plan_name (str): The name of the plan being evaluated.
            op_config (Dict[str, Any]): The original operation configuration.
            plan (Union[Dict[str, Any], List[Dict[str, Any]]]): The plan to be evaluated,
                which can be a single operation or a list of operations.
            input_data (List[Dict[str, Any]]): The input data to run the plan on.
            validator_prompt (str): The prompt used to assess the quality of the output.

        Returns:
            Tuple[float, float, List[Dict[str, Any]]]: A tuple containing:
                - The average quality score of the plan's output (float)
                - The runtime of the plan (float)
                - The output data produced by the plan (List[Dict[str, Any]])

        Note:
            The quality score is calculated based on the assessment of each output
            item using the validator prompt. The scoring is as follows:
            - Satisfactory: 4 points
            - Mostly Satisfactory: 3 points
            - Partially Satisfactory: 2 points
            - Unsatisfactory: 1 point
            The final score is the average of all individual scores.
        """

        if isinstance(plan, dict):
            plan = [plan]

        output_data = input_data
        start_time = time.time()
        for op in plan:
            output_data = self._run_operation(op, output_data, is_build=True)
        runtime = time.time() - start_time

        # Reorder output_data to match input_data
        output_data = [
            next(
                output
                for output in output_data
                if output["_map_opt_id"] == inp["_map_opt_id"]
            )
            for inp in input_data
        ]

        scores = []

        for idx in range(len(input_data)):
            # Evaluate the quality of the output using the custom validator prompt
            quality = self._assess_output_quality(
                op_config, input_data, output_data, idx, validator_prompt
            )
            score_map = {
                "Satisfactory": 4,
                "Mostly Satisfactory": 3,
                "Partially Satisfactory": 2,
                "Unsatisfactory": 1,
            }
            scores.append(score_map.get(quality["quality_category"], 0))

        average_score = sum(scores) / len(scores)

        return average_score, runtime, output_data

    def _assess_operation(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Dict[str, Any]:
        system_prompt = "You are an AI assistant tasked with assessing the performance of data processing operations. Use the provided validator prompt to evaluate the operation's output."

        # Extract input variables from the prompt
        variables_in_prompt = extract_jinja_variables(op_config["prompt"])
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]
        input_sample = input_data[:2]
        output_sample = [
            next(
                (
                    output
                    for output in output_data
                    if output["_map_opt_id"] == input_item["_map_opt_id"]
                ),
                "N/A",
            )
            for input_item in input_sample
        ]

        # Get output schema
        output_schema = op_config.get("output", {}).get("schema", {})
        # Calculate available tokens for sample data
        model_input_context_length = model_cost.get(self.llm_client.model, {}).get(
            "max_input_tokens", 8192
        )
        prompt_tokens = count_tokens(
            op_config.get("prompt", "N/A"), self.llm_client.model
        )
        available_tokens = (
            model_input_context_length - prompt_tokens - 100
        ) // 4  # 100 token buffer, divide by 4 for each sample

        # Prepare and truncate sample data
        input_1 = truncate_sample_data(
            {key: input_sample[0].get(key, "N/A") for key in variables_in_prompt},
            available_tokens,
            [variables_in_prompt],
            self.llm_client.model,
        )
        output_1 = truncate_sample_data(
            {key: output_sample[0].get(key, "N/A") for key in output_schema.keys()},
            available_tokens,
            [list(output_schema.keys())],
            self.llm_client.model,
        )

        prompt = f"""Task: Assess the performance of a data processing operation based on sample input-output pairs and a custom validator prompt.

        Operation Name: {op_config['name']}
        Operation Type: {op_config['type']}
        Current Task Prompt: {op_config.get('prompt', 'N/A')}

        Sample Input-Output Pairs:
        ---Pair 1---
        {json.dumps({"input": input_1, "output": output_1}, indent=2)}
        """

        if len(input_sample) > 1:
            input_2 = truncate_sample_data(
                {key: input_sample[1].get(key, "N/A") for key in variables_in_prompt},
                available_tokens,
                [variables_in_prompt],
                self.llm_client.model,
            )
            output_2 = truncate_sample_data(
                {key: output_sample[1].get(key, "N/A") for key in output_schema.keys()},
                available_tokens,
                [list(output_schema.keys())],
                self.llm_client.model,
            )
            prompt += f"""
        ---Pair 2---
        {json.dumps({"input": input_2, "output": output_2}, indent=2)}
        """

        prompt += f"""
        Custom Validator Prompt:
        {validator_prompt}

        Based on the above information, please assess the operation's performance. Provide your assessment in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "needs_improvement": {"type": "boolean"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["needs_improvement", "reasons", "improvements"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)

    def _assess_output_quality(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        element_idx: int,
        validator_prompt: str,
    ) -> str:
        """
        Assess the quality of a single output element against its corresponding input.

        This method evaluates the quality of a specific output element by comparing it
        to its corresponding input element, using the provided validator prompt as a
        guideline for assessment.

        Args:
            op_config (Dict[str, Any]): The configuration of the operation being assessed.
            input_data (List[Dict[str, Any]]): The list of input data elements.
            output_data (List[Dict[str, Any]]): The list of output data elements.
            element_idx (int): The index of the specific input/output pair to assess.
            validator_prompt (str): The prompt used to guide the quality assessment.

        Returns:
            str: A JSON string containing the quality assessment, including a quality
                 category and a reason for the assessment.

        The quality assessment is categorized into four levels:
        1. "Unsatisfactory": The output failed to meet any validator prompt requirements.
        2. "Partially Satisfactory": The output met some, but not all, requirements.
        3. "Mostly Satisfactory": The output met most requirements with room for improvement.
        4. "Satisfactory": The output fully met all validator prompt requirements.

        This method uses the LLM client to generate the quality assessment based on
        the input-output pair and the validator prompt.
        """

        system_prompt = "You are an AI assistant tasked with evaluating the quality of data processing outputs."
        output_schema_keys = list(
            set(output_data[0].keys()) - set(input_data[0].keys())
        )
        document_id = input_data[element_idx]["_map_opt_id"]
        input_elem = input_data[element_idx]
        output_elem = [
            item for item in output_data if item["_map_opt_id"] == document_id
        ][0]
        output_elem = {key: output_elem[key] for key in output_schema_keys}

        variables_in_prompt = extract_jinja_variables(op_config["prompt"])
        variables_in_prompt = [v.replace("input.", "") for v in variables_in_prompt]

        # Filter input_data to only include relevant variables
        input_elem = {key: input_elem[key] for key in variables_in_prompt}

        prompt = f"""
        Validation Prompt:
        {validator_prompt}

        Input and Output Data Sample:
        {json.dumps({"input": input_elem, "output": output_elem}, indent=2)}

        Based on the validation prompt and the input-output data sample, assess the quality of the output.
        Categorize the quality into one of these four categories:
        1. "Unsatisfactory": The output failed to meet any of the validation prompt requirements.
        2. "Partially Satisfactory": The output met some of the validation prompt requirements but not all.
        3. "Mostly Satisfactory": The output met most of the validation prompt requirements but has some room for improvement.
        4. "Satisfactory": The output fully met the validation prompt requirements.

        Provide your response in the following format:
        """

        parameters = {
            "type": "object",
            "properties": {
                "quality_category": {
                    "type": "string",
                    "enum": [
                        "Unsatisfactory",
                        "Partially Satisfactory",
                        "Mostly Satisfactory",
                        "Satisfactory",
                    ],
                },
                "reason": {"type": "string"},
            },
            "required": ["quality_category", "reason"],
        }

        response = self.llm_client.generate(
            [
                {"role": "user", "content": prompt},
            ],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        return result
