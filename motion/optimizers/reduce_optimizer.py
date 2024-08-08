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
    """
    A class that optimizes reduce operations in data processing pipelines.

    This optimizer analyzes the input and output of a reduce operation, creates and evaluates
    multiple reduce plans, and selects the best plan for optimizing the operation's performance.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the optimizer.
        console (Console): Rich console object for pretty printing.
        llm_client (LLMClient): Client for interacting with a language model.
        _run_operation (Callable): Function to run an operation.
        max_threads (int): Maximum number of threads to use for parallel processing.
        num_fold_prompts (int): Number of fold prompts to generate.
        num_samples_in_validation (int): Number of samples to use in validation.
    """

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
        """
        Initialize the ReduceOptimizer.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the optimizer.
            console (Console): Rich console object for pretty printing.
            llm_client (LLMClient): Client for interacting with a language model.
            max_threads (int): Maximum number of threads to use for parallel processing.
            run_operation (Callable): Function to run an operation.
            num_fold_prompts (int, optional): Number of fold prompts to generate. Defaults to 1.
            num_samples_in_validation (int, optional): Number of samples to use in validation. Defaults to 10.
        """
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
        """
        Optimize the reduce operation based on the given configuration and input data.

        This method performs the following steps:
        1. Run the original operation
        2. Generate a validator prompt
        3. Validate the output
        4. If improvement is needed, create and evaluate multiple reduce plans
        5. Run the best reduce plan

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: A tuple containing the optimized configuration
            and the output of the optimized operation.
        """

        original_output = self._run_operation(op_config, input_data)

        # Step 1: Synthesize a validator prompt
        validator_prompt = self._generate_validator_prompt(
            op_config, input_data, original_output
        )

        # Step 2: validate the output
        validator_inputs = self._create_validation_inputs(
            input_data, op_config["reduce_key"]
        )
        validation_results = self._validate_reduce_output(
            op_config, validator_inputs, original_output, validator_prompt
        )

        # Print the validation results
        self.console.log("[bold]Validation Results:[/bold]")
        if validation_results["needs_improvement"]:
            self.console.log(
                "\n".join(
                    [
                        f"Issues: {result['issues']} Suggestions: {result['suggestions']}"
                        for result in validation_results["validation_results"]
                    ]
                )
            )

            # Step 2.5: Determine if the reduce operation is commutative
            is_commutative = self._is_commutative(op_config, input_data)

            # Step 3: Create and evaluate multiple reduce plans
            reduce_plans = self._create_reduce_plans(
                op_config, input_data, is_commutative
            )
            best_plan = self._evaluate_reduce_plans(
                reduce_plans, input_data, validator_prompt
            )

            # Step 4: Run the best reduce plan
            optimized_output = self._run_operation(best_plan, input_data)

            return best_plan, optimized_output
        else:
            self.console.log("No improvements identified.")
            return op_config, original_output

    def _is_commutative(
        self, op_config: Dict[str, Any], input_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if the reduce operation is commutative.

        This method analyzes the reduce operation configuration and a sample of the input data
        to determine if the operation is commutative (i.e., the order of combining elements
        doesn't affect the final result).

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.

        Returns:
            bool: True if the operation is determined to be commutative, False otherwise.
        """
        system_prompt = (
            "You are an AI assistant helping to optimize data processing pipelines."
        )

        # Sample a subset of input data for analysis
        sample_size = min(5, len(input_data))
        sample_input = random.sample(input_data, sample_size)

        prompt = f"""
        Analyze the following reduce operation and determine if it is commutative:

        Reduce Operation Prompt:
        {op_config['prompt']}

        Sample Input Data:
        {json.dumps(sample_input, indent=2)}

        A reduce operation is commutative if the order of combining elements doesn't affect the final result.
        For example, sum and product operations are commutative, while subtraction and division are not.

        Based on the reduce operation prompt and the sample input data, determine if this operation is likely to be commutative.
        Answer with 'yes' if order matters (non-commutative) or 'no' if order doesn't matter (commutative). 
        Explain your reasoning briefly.

        For example:
        - Merging extracted key-value pairs from documents is commutative: combining {{"name": "John", "age": 30}} with {{"city": "New York", "job": "Engineer"}} yields the same result regardless of order
        - Generating a timeline of events is non-commutative: the order of events matters for maintaining chronological accuracy.

        Consider these examples when determining if the combining operation is commutative or not. You might also have to consider the specific data.
        """

        parameters = {
            "type": "object",
            "properties": {
                "is_commutative": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": ["is_commutative", "explanation"],
        }

        response = self.llm_client.generate(
            [{"role": "user", "content": prompt}],
            system_prompt,
            parameters,
        )
        result = json.loads(response.choices[0].message.content)

        self.console.log(
            f"Result: {'Commutative' if result['is_commutative'] else 'Non-commutative'} - Commutativity analysis: {result['explanation']}"
        )
        return result["is_commutative"]

    def _generate_validator_prompt(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        original_output: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a custom validator prompt for assessing the quality of the reduce operation output.

        This method creates a prompt that will be used to validate the output of the reduce operation.
        It includes specific questions about the quality and completeness of the output.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            original_output (List[Dict[str, Any]]): Original output of the reduce operation.

        Returns:
            str: A custom validator prompt as a string.
        """
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
        validation_inputs: List[Dict[str, Any]],
        output_data: List[Dict[str, Any]],
        validator_prompt: str,
    ) -> Dict[str, Any]:
        """
        Validate the output of the reduce operation using the generated validator prompt.

        This method assesses the quality of the reduce operation output by applying the validator prompt
        to multiple samples of the input and output data.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            output_data (List[Dict[str, Any]]): Output data from the reduce operation.
            validator_prompt (str): The validator prompt generated earlier.

        Returns:
            Dict[str, Any]: A dictionary containing validation results and a flag indicating if improvement is needed.
        """
        system_prompt = "You are an AI assistant tasked with validating the output of reduce operations in data processing pipelines."

        validation_results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for sample_input in validation_inputs:
                reduce_key = op_config["reduce_key"]
                sample_output = next(
                    (
                        item
                        for item in output_data
                        if item[reduce_key] == sample_input[reduce_key]
                    ),
                    None,
                )

                if sample_output is None:
                    self.console.log(
                        f"Warning: No output found for reduce key {sample_input[reduce_key]}"
                    )
                    continue

                prompt = f"""
                {validator_prompt}

                Reduce Operation Task:
                {op_config["prompt"]}

                Input Data Sample:
                {json.dumps(sample_input, indent=2)}

                Output Data Sample:
                {json.dumps(sample_output, indent=2)}

                Based on the validator prompt and the input/output samples, assess the quality (e.g., correctness, completeness) of the reduce operation output.
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

    def _create_validation_inputs(
        self, input_data: List[Dict[str, Any]], reduce_key: str
    ) -> List[Dict[str, Any]]:
        # Group input data by reduce_key
        grouped_data = {}
        for item in input_data:
            key = item[reduce_key]
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(item)

        # Select a fixed set of samples
        samples = []
        for key, group in grouped_data.items():
            if len(group) > 1:
                sample_input = random.choice(group)
                samples.append(sample_input)

        return samples[: self.num_samples_in_validation]

    def _create_reduce_plans(
        self,
        op_config: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        is_commutative: bool,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple reduce plans based on the input data and operation configuration.

        This method generates various reduce plans by varying batch sizes and fold prompts.
        It also calculates a compression ratio, which is the ratio of input data size to output data size.
        The compression ratio weakly indicates how much the data is being condensed by the reduce operation.
        A higher compression ratio suggests that larger batch sizes may be more efficient, as more data
        can be processed in each reduce step.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            input_data (List[Dict[str, Any]]): Input data for the reduce operation.
            is_commutative (bool): Flag indicating whether the reduce operation is commutative.

        Returns:
            List[Dict[str, Any]]: A list of reduce plans, each with different batch sizes and fold prompts.
        """
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
        self.console.log(
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
                plan["commutative"] = is_commutative
                plans.append(plan)

        return plans

    def _calculate_compression_ratio(
        self,
        op_config: Dict[str, Any],
        sample_input: List[Dict[str, Any]],
        sample_output: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate the compression ratio of the reduce operation.

        This method compares the size of the input data to the size of the output data
        to determine how much the data is being compressed by the reduce operation.

        Args:
            op_config (Dict[str, Any]): Configuration for the reduce operation.
            sample_input (List[Dict[str, Any]]): Sample input data.
            sample_output (List[Dict[str, Any]]): Sample output data.

        Returns:
            float: The calculated compression ratio.
        """
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
        """
        Synthesize fold prompts for the reduce operation. We generate multiple
        fold prompts in case one is bad.

        A fold operation is a higher-order function that iterates through a data structure,
        accumulating the results of applying a given combining operation to its elements.
        In the context of reduce operations, folding allows processing of data in batches,
        which can significantly improve performance for large datasets.

        This method generates multiple fold prompts that can be used to optimize the reduce operation
        by allowing it to run on batches of inputs. It uses the language model to create prompts
        that are variations of the original reduce prompt, adapted for folding operations.

        Args:
            op_config (Dict[str, Any]): The configuration of the reduce operation.
            sample_input (List[Dict[str, Any]]): A sample of the input data.
            sample_output (List[Dict[str, Any]]): A sample of the output data.
            num_prompts (int, optional): The number of fold prompts to generate. Defaults to 2.

        Returns:
            List[str]: A list of synthesized fold prompts.

        The method performs the following steps:
        1. Sets up the system prompt and parameters for the language model.
        2. Defines a function to get random examples from the sample data.
        3. Creates a prompt template for generating fold prompts.
        4. Uses multi-threading to generate multiple fold prompts in parallel.
        5. Returns the list of generated fold prompts.
        """
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
        """
        Evaluate multiple reduce plans and select the best one.

        This method takes a list of reduce plans, evaluates each one using the input data
        and a validator prompt, and selects the best plan based on the evaluation scores.
        It also attempts to create and evaluate a merged plan that enhances the runtime performance
        of the best plan.

        A merged plan is an optimization technique applied to the best-performing plan
        that uses the fold operation. It allows the best plan to run even faster by
        executing parallel folds and then merging the results of these individual folds
        together. We default to a merge batch size of 2, but one can increase this.

        Args:
            plans (List[Dict[str, Any]]): A list of reduce plans to evaluate.
            input_data (List[Dict[str, Any]]): The input data to use for evaluation.
            validator_prompt (str): The prompt to use for validating the output of each plan.

        Returns:
            Dict[str, Any]: The best reduce plan, either the top-performing original plan
                            or a merged plan if it performs well enough.

        The method performs the following steps:
        1. Evaluates each plan using multi-threading.
        2. Sorts the plans based on their evaluation scores.
        3. Selects the best plan and attempts to create a merged plan.
        4. Evaluates the merged plan and compares it to the best original plan.
        5. Returns either the merged plan or the best original plan based on their scores.
        """
        self.console.log("\n[bold]Evaluating Reduce Plans:[/bold]")
        for i, plan in enumerate(plans):
            self.console.log(f"Plan {i+1} (batch size: {plan['fold_batch_size']})")

        plan_scores = []
        plan_outputs = {}

        # Create a fixed random sample for evaluation
        sample_size = min(100, len(input_data))
        evaluation_sample = random.sample(input_data, sample_size)

        # Create a fixed set of validation samples
        validation_inputs = self._create_validation_inputs(
            evaluation_sample, plan["reduce_key"]
        )

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    self._evaluate_single_plan,
                    plan,
                    evaluation_sample,
                    validator_prompt,
                    validation_inputs,
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

        self.console.log("\n[bold]Reduce Plan Scores:[/bold]")
        for i, (plan, score) in enumerate(sorted_plans):
            self.console.log(
                f"Plan {i+1} (batch size: {plan['fold_batch_size']}): {score:.2f}"
            )

        best_plan, best_score = sorted_plans[0]
        self.console.log(
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
            self.console.log("\n[bold]Synthesized Merge Prompt:[/bold]")
            self.console.log(merged_plan["merge_prompt"])

        # Set merge_batch_size to 2 and num_parallel_folds to 5
        merged_plan["merge_batch_size"] = 2

        # Evaluate the merged plan
        _, merged_plan_score, _, operation_instance = self._evaluate_single_plan(
            merged_plan,
            evaluation_sample,
            validator_prompt,
            validation_inputs,
            return_instance=True,
        )

        # Get the merge and fold times from the operation instance
        merge_times = operation_instance.merge_times
        fold_times = operation_instance.fold_times
        merge_avg_time = mean(merge_times) if merge_times else None
        fold_avg_time = mean(fold_times) if fold_times else None

        self.console.log(f"\n[bold]Scores:[/bold]")
        self.console.log(f"Original plan: {best_score:.2f}")
        self.console.log(f"Merged plan: {merged_plan_score:.2f}")

        # Compare scores and decide which plan to use
        if merged_plan_score >= best_score * 0.75:
            self.console.log(
                f"\n[green]Using merged plan with score: {merged_plan_score:.2f}[/green]"
            )
            if merge_avg_time and fold_avg_time:
                merged_plan["merge_time"] = merge_avg_time
                merged_plan["fold_time"] = fold_avg_time
            return merged_plan
        else:
            self.console.log(
                f"\n[yellow]Merged plan quality too low. Using original plan with score: {best_score:.2f}[/yellow]"
            )
            return best_plan

    def _evaluate_single_plan(
        self,
        plan: Dict[str, Any],
        input_data: List[Dict[str, Any]],
        validator_prompt: str,
        validation_inputs: List[Dict[str, Any]],
        return_instance: bool = False,
    ) -> Union[
        Tuple[Dict[str, Any], float, List[Dict[str, Any]]],
        Tuple[Dict[str, Any], float, List[Dict[str, Any]], BaseOperation],
    ]:
        """
        Evaluate a single reduce plan using the provided input data and validator prompt.

        This method runs the reduce operation with the given plan, validates the output,
        and calculates a score based on the validation results. The scoring works as follows:
        1. It counts the number of valid results from the validation.
        2. The score is calculated as the ratio of valid results to the total number of validation results.
        3. This produces a score between 0 and 1, where 1 indicates all results were valid, and 0 indicates none were valid.

        TODO: We should come up with a better scoring method here, maybe pairwise comparisons.

        Args:
            plan (Dict[str, Any]): The reduce plan to evaluate.
            input_data (List[Dict[str, Any]]): The input data to use for evaluation.
            validator_prompt (str): The prompt to use for validating the output.
            return_instance (bool, optional): Whether to return the operation instance. Defaults to False.

        Returns:
            Union[
                Tuple[Dict[str, Any], float, List[Dict[str, Any]]],
                Tuple[Dict[str, Any], float, List[Dict[str, Any]], BaseOperation],
            ]: A tuple containing the plan, its score, the output data, and optionally the operation instance.

        The method performs the following steps:
        1. Runs the reduce operation with the given plan on the input data.
        2. Validates the output using the validator prompt.
        3. Calculates a score based on the validation results.
        4. Returns the plan, score, output data, and optionally the operation instance.
        """
        output = self._run_operation(plan, input_data, return_instance)
        if return_instance:
            output, operation_instance = output

        validation_result = self._validate_reduce_output(
            plan, validation_inputs, output, validator_prompt
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
        """
        Synthesize a merge prompt for combining multiple folded outputs in a reduce operation.

        This method generates a merge prompt that can be used to combine the results of multiple
        parallel fold operations into a single output. It uses the language model to create a prompt
        that is consistent with the original reduce and fold prompts while addressing the specific
        requirements of merging multiple outputs.

        Args:
            plan (Dict[str, Any]): The reduce plan containing the original prompt and fold prompt.
            sample_outputs (List[Dict[str, Any]]): Sample outputs from the fold operation to use as examples.

        Returns:
            str: The synthesized merge prompt as a string.

        The method performs the following steps:
        1. Sets up the system prompt for the language model.
        2. Prepares a random sample output to use as an example.
        3. Creates a detailed prompt for the language model, including the original reduce prompt,
           fold prompt, sample output, and instructions for creating the merge prompt.
        4. Uses the language model to generate the merge prompt.
        5. Returns the generated merge prompt.
        """
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
