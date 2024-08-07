import json
import random
from typing import Any, Dict, List, Callable, Tuple, Union
from motion.operations.base import BaseOperation
from rich.console import Console
from motion.optimizers.utils import LLMClient, extract_jinja_variables
from motion.operations import get_operation
from collections import Counter
from statistics import mean, median
from concurrent.futures import ThreadPoolExecutor, as_completed


class ReduceOptimizer:
    def __init__(
        self,
        config: Dict[str, Any],
        console: Console,
        llm_client: LLMClient,
        max_threads: int,
        run_operation: Callable,
        num_fold_prompts: int = 1,
        num_samples_in_validation: int = 10,
    ):
        self.config = config
        self.console = console
        self.llm_client = llm_client
        self._run_operation = run_operation
        self.max_threads = max_threads
        self.num_fold_prompts = num_fold_prompts
        self.num_samples_in_validation = num_samples_in_validation

    def optimize(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

        original_output = self._run_operation(op_config, input_data)

        # Step 1: Synthesize a validator prompt
        validator_prompt = self._generate_validator_prompt(
            op_config, input_data, original_output
        )

        # Step 2: validate the output
        validation_results = self._validate_reduce_output(
            op_config, input_data, original_output, validator_prompt
        )

        # Print the validation results
        self.console.print("[bold]Validation Results:[/bold]")
        if validation_results["needs_improvement"]:
            self.console.print(
                "\n".join(
                    [
                        f"Issues: {result['issues']} Suggestions: {result['suggestions']}"
                        for result in validation_results["validation_results"]
                    ]
                )
            )

            # Step 3: Create and evaluate multiple reduce plans
            reduce_plans = self._create_reduce_plans(op_config, input_data)
            best_plan = self._evaluate_reduce_plans(
                reduce_plans, input_data, validator_prompt
            )

            # Step 4: Run the best reduce plan
            optimized_output = self._run_operation(best_plan, input_data)

            return best_plan, optimized_output
        else:
            self.console.print("No improvements identified.")
            return op_config, original_output

    def _generate_validator_prompt(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        original_output: List[Dict[str, Any]],
    ) -> str:
        system_prompt = "You are an AI assistant tasked with creating custom validation prompts for reduce operations in data processing pipelines."

        sample_input = random.choice(input_data)
        input_keys = op_config.get("input", {}).get("schema", {})
        if input_keys:
            sample_input = {k: sample_input[k] for k in input_keys}

        reduce_key = op_config.get("reduce_key")
        if reduce_key and original_output:
            key = next(
                (item[reduce_key] for item in original_output if reduce_key in item),
                None,
            )
            sample_output = next(
                (item for item in original_output if item.get(reduce_key) == key), {}
            )
        else:
            sample_output = original_output[0] if original_output else {}

        output_keys = op_config.get("output", {}).get("schema", {})
        sample_output = {k: sample_output[k] for k in output_keys}

        prompt = f"""
        Analyze the following reduce operation and its input/output:

        Reduce Operation Prompt:
        {op_config["prompt"]}

        Sample Input:
        {json.dumps(sample_input, indent=2)}

        Sample Output:
        {json.dumps(sample_output, indent=2)}

        Create a custom validator prompt that will assess how well the reduce operation performed its intended task. The prompt should ask specific questions about the quality and completeness of the output, such as:
        1. Are all input values properly represented in the output?
        2. Is the aggregation performed correctly according to the task requirements?
        3. Is there any loss of important information during the reduction process?
        4. Does the output maintain the required structure and data types?

        Provide your response as a single string containing the custom validator prompt.
        """

        parameters = {
            "type": "object",
            "properties": {"validator_prompt": {"type": "string"}},
            "required": ["validator_prompt"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)["validator_prompt"]

    def _validate_reduce_output(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Dict[str, Any]:
        system_prompt = "You are an AI assistant tasked with validating the output of reduce operations in data processing pipelines."

        # Count occurrences of each key in input_data
        key_counts = {}
        for item in input_data:
            key = item[op_config["reduce_key"]]
            key_counts[key] = key_counts.get(key, 0) + 1

        validation_results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for _ in range(self.num_samples_in_validation):

                # Select a key weighted by its count
                selected_key = random.choices(
                    list(key_counts.keys()), weights=list(key_counts.values()), k=1
                )[0]

                # Find a sample input with the selected key
                sample_input = next(
                    item
                    for item in input_data
                    if item[op_config["reduce_key"]] == selected_key
                )

                # Find the corresponding output
                sample_output = next(
                    (
                        out
                        for out in output_data
                        if out[op_config["reduce_key"]] == selected_key
                    ),
                    None,
                )

                prompt = f"""
                {validator_prompt}

                Reduce Operation Task:
                {op_config["prompt"]}

                Input Data Sample:
                {json.dumps(sample_input, indent=2)}

                Output Data Sample:
                {json.dumps(sample_output, indent=2)}

                Based on the validator prompt and the input/output samples, assess the quality of the reduce operation output.
                Provide your assessment in the following format:
                """

                parameters = {
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "suggestions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["is_valid", "issues", "suggestions"],
                }

                futures.append(
                    executor.submit(
                        self.llm_client.generate,
                        [{"role": "user", "content": prompt}],
                        system_prompt,
                        parameters,
                    )
                )

            for future in as_completed(futures):
                response = future.result()
                validation_results.append(
                    json.loads(response.choices[0].message.content)
                )

        # Determine if optimization is needed based on validation results
        invalid_count = sum(
            1 for result in validation_results if not result["is_valid"]
        )
        needs_improvement = invalid_count > 1

        return {
            "needs_improvement": needs_improvement,
            "validation_results": validation_results,
        }

    def _create_reduce_plans(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        reduce_key = op_config["reduce_key"]
        key_counts = Counter(item[reduce_key] for item in input_data)
        values_per_key = list(key_counts.values())

        avg_values = mean(values_per_key)
        median_values = median(values_per_key)
        max_values = max(values_per_key)

        # Run the operation once on a sample of the input data
        sample_size = min(100, len(input_data))
        sample_input = random.sample(input_data, sample_size)
        sample_output = self._run_operation(op_config, sample_input)

        # Calculate compression ratio
        compression_ratio = self._calculate_compression_ratio(
            op_config, sample_input, sample_output
        )

        # Print the compression ratio
        self.console.print(
            f"[bold]Estimated Compression Ratio:[/bold] {compression_ratio:.2f}"
        )

        plans = []

        if "fold_prompt" in op_config:
            current_batch_size = op_config.get("fold_batch_size", max_values)
            batch_sizes = [
                max(1, int(current_batch_size * 0.25)),
                max(1, int(current_batch_size * 0.5)),
                max(1, int(current_batch_size * 0.75)),
                current_batch_size,
            ]
            fold_prompts = [op_config["fold_prompt"]]
        else:
            batch_sizes = [
                max(1, int(avg_values * 0.5)),
                max(1, int(avg_values)),
                max(1, int(median_values)),
                max(1, int(max_values * 0.5)),
                max_values,
            ]

            # Add compression ratio-based batch size
            compression_batch_size = max(1, int(compression_ratio * max_values))
            batch_sizes.append(compression_batch_size)

            # Remove duplicates and sort
            batch_sizes = sorted(set(batch_sizes))

            # Generate multiple fold prompts
            fold_prompts = self._synthesize_fold_prompts(
                op_config,
                sample_input,
                sample_output,
                num_prompts=self.num_fold_prompts,
            )

        for batch_size in batch_sizes:
            for fold_prompt in fold_prompts:
                plan = op_config.copy()
                plan["fold_prompt"] = fold_prompt
                plan["fold_batch_size"] = batch_size
                plans.append(plan)

        return plans

    def _calculate_compression_ratio(
        self,
        op_config: Dict[str, Any],
        sample_input: List[Dict[str, Any]],
        sample_output: List[Dict[str, Any]],
    ) -> float:
        reduce_key = op_config["reduce_key"]
        input_schema = op_config.get("input", {}).get("schema", {})
        output_schema = op_config["output"]["schema"]

        compression_ratios = {}
        for key in set(item[reduce_key] for item in sample_input):
            key_input = [item for item in sample_input if item[reduce_key] == key]
            key_output = [item for item in sample_output if item[reduce_key] == key]

            if input_schema:
                key_input_chars = sum(
                    len(json.dumps({k: item[k] for k in input_schema if k in item}))
                    for item in key_input
                )
            else:
                key_input_chars = sum(len(json.dumps(item)) for item in key_input)

            key_output_chars = sum(
                len(json.dumps({k: item[k] for k in output_schema if k in item}))
                for item in key_output
            )

            compression_ratios[key] = (
                key_output_chars / key_input_chars if key_input_chars > 0 else 1
            )

        if not compression_ratios:
            return 1

        # Calculate importance weights based on the number of items for each key
        total_items = sum(
            len([item for item in sample_input if item[reduce_key] == key])
            for key in compression_ratios
        )
        importance_weights = {
            key: len([item for item in sample_input if item[reduce_key] == key])
            / total_items
            for key in compression_ratios
        }

        # Calculate weighted average of compression ratios
        weighted_sum = sum(
            compression_ratios[key] * importance_weights[key]
            for key in compression_ratios
        )
        return weighted_sum

    def _synthesize_fold_prompts(
        self,
        op_config: Dict[str, Any],
        sample_input: List[Dict[str, Any]],
        sample_output: List[Dict[str, Any]],
        num_prompts: int = 2,
    ) -> List[str]:
        system_prompt = "You are an AI assistant tasked with creating a fold prompt for reduce operations in data processing pipelines."
        original_prompt = op_config["prompt"]

        input_schema = op_config.get("input", {}).get("schema", {})
        output_schema = op_config["output"]["schema"]
        reduce_key = op_config["reduce_key"]

        def get_random_examples():
            random_key = random.choice(
                [item[reduce_key] for item in sample_input if reduce_key in item]
            )
            input_example = random.choice(
                [item for item in sample_input if item[reduce_key] == random_key]
            )
            if input_schema:
                input_example = {
                    k: input_example[k] for k in input_schema if k in input_example
                }
            output_example = random.choice(
                [item for item in sample_output if item[reduce_key] == random_key]
            )
            output_example = {
                k: output_example[k] for k in output_schema if k in output_example
            }
            return input_example, output_example

        parameters = {
            "type": "object",
            "properties": {
                "fold_prompt": {
                    "type": "string",
                }
            },
            "required": ["fold_prompt"],
        }

        def generate_single_prompt():
            input_example, output_example = get_random_examples()
            prompt = f"""
            Original Reduce Operation Prompt:
            {original_prompt}
            
            Sample Input:
            {json.dumps(input_example, indent=2)}

            Sample Output:
            {json.dumps(output_example, indent=2)}

            Create a fold prompt for the reduce operation to run on batches of inputs. The fold prompt should:
            1. Minimally modify the original reduce prompt
            2. Describe how to combine the new values with the current reduced value
            3. Be designed to work iteratively, allowing for multiple fold operations. The first iteration will use the original prompt, and all successive iterations will use the fold prompt.

            The fold prompt should be a Jinja2 template with the following variables available:
            - {{ output }}: The current reduced value (a dictionary with the current output schema)
            - {{ values }}: A list of new values to be folded in
            - {{ reduce_key }}: The key used for grouping in the reduce operation

            Provide the fold prompt as a string.
            """
            response = self.llm_client.generate(
                [{"role": "user", "content": prompt}],
                system_prompt,
                parameters,
            )
            return json.loads(response.choices[0].message.content)["fold_prompt"]

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            fold_prompts = list(
                executor.map(lambda _: generate_single_prompt(), range(num_prompts))
            )

        return fold_prompts

    def _evaluate_reduce_plans(
        self,
        plans: List[Dict[str, Any]],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Dict[str, Any]:
        self.console.print("\n[bold]Evaluating Reduce Plans:[/bold]")
        for i, plan in enumerate(plans):
            self.console.print(f"Plan {i+1} (batch size: {plan['fold_batch_size']})")

        plan_scores = []
        plan_outputs = {}

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    self._evaluate_single_plan, plan, input_data, validator_prompt
                )
                for plan in plans
            ]
            for future in as_completed(futures):
                plan, score, output = future.result()
                plan_scores.append((plan, score))
                plan_outputs[id(plan)] = output

        # Sort plans by score in descending order, then by fold_batch_size in descending order
        sorted_plans = sorted(
            plan_scores, key=lambda x: (x[1], x[0]["fold_batch_size"]), reverse=True
        )

        self.console.print("\n[bold]Reduce Plan Scores:[/bold]")
        for i, (plan, score) in enumerate(sorted_plans):
            self.console.print(
                f"Plan {i+1} (batch size: {plan['fold_batch_size']}): {score:.2f}"
            )

        best_plan, best_score = sorted_plans[0]
        self.console.print(
            f"\n[green]Selected best plan with score: {best_score:.2f} and batch size: {best_plan['fold_batch_size']}[/green]"
        )

        # Create a new plan with merge prompt and updated parameters
        merged_plan = best_plan.copy()

        # Synthesize merge prompt if it doesn't exist
        if "merge_prompt" not in merged_plan:
            merged_plan["merge_prompt"] = self._synthesize_merge_prompt(
                merged_plan, plan_outputs[id(best_plan)]
            )
            # Print the synthesized merge prompt
            self.console.print("\n[bold]Synthesized Merge Prompt:[/bold]")
            self.console.print(merged_plan["merge_prompt"])

        # Set merge_batch_size to 2 and num_parallel_folds to 5
        merged_plan["merge_batch_size"] = 2

        # Evaluate the merged plan
        _, merged_plan_score, _, operation_instance = self._evaluate_single_plan(
            merged_plan, input_data, validator_prompt, return_instance=True
        )

        # Get the merge and fold times from the operation instance
        merge_times = operation_instance.merge_times
        fold_times = operation_instance.fold_times
        merge_avg_time = mean(merge_times) if merge_times else None
        fold_avg_time = mean(fold_times) if fold_times else None

        self.console.print(f"\n[bold]Scores:[/bold]")
        self.console.print(f"Original plan: {best_score:.2f}")
        self.console.print(f"Merged plan: {merged_plan_score:.2f}")

        # Compare scores and decide which plan to use
        if merged_plan_score >= best_score * 0.75:
            self.console.print(
                f"\n[green]Using merged plan with score: {merged_plan_score:.2f}[/green]"
            )
            if merge_avg_time and fold_avg_time:
                merged_plan["merge_time"] = merge_avg_time
                merged_plan["fold_time"] = fold_avg_time
            return merged_plan
        else:
            self.console.print(
                f"\n[yellow]Merged plan quality too low. Using original plan with score: {best_score:.2f}[/yellow]"
            )
            return best_plan

    def _evaluate_single_plan(
        self,
        plan: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
        return_instance: bool = False,
    ) -> Union[
        Tuple[Dict[str, Any], float, List[Dict[str, Any]]],
        Tuple[Dict[str, Any], float, List[Dict[str, Any]], BaseOperation],
    ]:
        output = self._run_operation(plan, input_data, return_instance)
        if return_instance:
            output, operation_instance = output
        validation_result = self._validate_reduce_output(
            plan, input_data, output, validator_prompt
        )

        # Calculate a score based on validation results
        valid_count = sum(
            1
            for result in validation_result["validation_results"]
            if result["is_valid"]
        )
        score = valid_count / len(validation_result["validation_results"])

        if return_instance:
            return plan, score, output, operation_instance
        else:
            return plan, score, output

    def _synthesize_merge_prompt(
        self, plan: Dict[str, Any], sample_outputs: List[Dict[str, Any]]
    ) -> str:
        system_prompt = "You are an AI assistant tasked with creating a merge prompt for reduce operations in data processing pipelines. The pipeline has a reduce operation, and incrementally folds inputs into a single output. We want to optimize the pipeline for speed by running multiple folds on different inputs in parallel, and then merging the fold outputs into a single output."

        output_schema = plan["output"]["schema"]
        random_output = random.choice(sample_outputs)
        random_output = {
            k: random_output[k] for k in output_schema if k in random_output
        }

        prompt = f"""
        Reduce Operation Prompt (runs on the first batch of inputs):
        {plan["prompt"]}
        
        Fold Prompt (runs on the second and subsequent batches of inputs):
        {plan["fold_prompt"]}
        
        Sample output of the fold operation (an input to the merge operation):
        {json.dumps(random_output, indent=2)}

        Create a merge prompt for the reduce operation to combine 2+ folded outputs. The merge prompt should:
        1. Give context on the task & fold operations, describing that the prompt will be used to combine multiple outputs from the fold operation (as if the original prompt was run on all inputs at once)
        2. Describe how to combine multiple folded outputs into a single output
        3. Minimally deviate from the reduce and fold prompts

        The merge prompt should be a Jinja2 template with the following variables available:
        - {{ outputs }}: A list of reduced outputs to be merged (each following the output schema). You can access the first output with {{ outputs[0] }} and the second with {{ outputs[1] }}

        Output Schema:
        {json.dumps(output_schema, indent=2)}

        Provide the merge prompt as a string.
        """

        parameters = {
            "type": "object",
            "properties": {
                "merge_prompt": {
                    "type": "string",
                }
            },
            "required": ["merge_prompt"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        return json.loads(response.choices[0].message.content)["merge_prompt"]
