#!/usr/bin/env python3
"""
Simple Baseline Agent for Reasoning Experiments

This is a very basic agent that:
1. Reads operator documentation
2. Looks at sample data
3. Generates a pipeline of operators using tool calls
4. Executes the pipeline and returns results
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from litellm import completion
from docetl.runner import DSLRunner
from experiments.reasoning.evaluation.utils import run_dataset_evaluation, get_evaluate_func, dataset_accuracy_metrics
import modal
from experiments.reasoning.utils import app, volume, VOLUME_MOUNT_PATH, image



DEFAULT_MODEL = "o3"
DEFAULT_OUTPUT_DIR = "outputs/simple_baseline"

class AgentAction(BaseModel):
    """Schema for agent action decisions."""
    action: Literal["try_pipeline", "return_pipeline"] = Field(
        ..., description="The action to take"
    )
    reasoning: str = Field(
        ..., description="Explanation of why this action was chosen"
    )

class PathResolver:
    """Handles path resolution for local and Modal environments."""
    
    @staticmethod
    def resolve_in_volume(path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str((Path(VOLUME_MOUNT_PATH) / p).resolve())
    
    @staticmethod
    def get_data_path(dataset: str) -> Path:
        """Get the training data path for a dataset."""
        path_map = {
            "cuad": "experiments/reasoning/data/train/cuad.json",
            "blackvault": "experiments/reasoning/data/train/blackvault.json", 
            "game_reviews": "experiments/reasoning/data/train/game_reviews.json",
            "reviews": "experiments/reasoning/data/train/game_reviews.json",
            "sustainability": "experiments/reasoning/data/train/sustainability.json",
            "biodex": "experiments/reasoning/data/train/biodex.json",
            "medec": "experiments/reasoning/data/train/medec.json",
            "facility": "experiments/reasoning/data/train/facility.json"
        }
        
        data_path = Path(path_map.get(dataset.lower(), f"experiments/reasoning/data/train/{dataset.lower()}.json"))
        
        # Try volume mount if original doesn't exist
        if not data_path.exists():
            data_path = Path(VOLUME_MOUNT_PATH) / data_path
            
        return data_path

class PipelineExecutor:
    """Handles pipeline creation and execution."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
    
    def create_yaml(self, operators: List[Dict], dataset: str, prefix: str = "pipeline") -> str:
        """Create a pipeline YAML configuration."""
        dataset_file = f"experiments/reasoning/data/train/{dataset.lower()}.json"
        output_json = self.experiment_dir / f"{prefix}_output.json"
        
        # Add azure/ prefix to models
        for op in operators:
            if "model" in op and "azure/" not in op["model"]:
                op["model"] = "azure/" + op["model"]
        
        config = {
            "bypass_cache": True,
            "datasets": {
                "input_data": {"type": "file", "path": dataset_file}
            },
            "default_model": "azure/gpt-4o-mini",
            "operations": operators,
            "pipeline": {
                "steps": [{
                    "name": f"{dataset}_pipeline",
                    "input": "input_data", 
                    "operations": [op["name"] for op in operators]
                }],
                "output": {"type": "file", "path": str(output_json)}
            }
        }
        
        yaml_path = self.experiment_dir / f"{prefix}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        return str(yaml_path)
    
    def rewrite_for_modal(self, yaml_path: str) -> str:
        """Rewrite pipeline YAML for Modal volume mounting."""
        if not Path(VOLUME_MOUNT_PATH).exists():
            return yaml_path
            
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        base_mount = Path(VOLUME_MOUNT_PATH)
        output_root = base_mount / "outputs" / self.experiment_dir.name
        
        # Update output paths
        pipeline_cfg = cfg.get("pipeline", {})
        if isinstance(pipeline_cfg, dict):
            out = pipeline_cfg.get("output")
            if isinstance(out, dict) and isinstance(out.get("path"), str):
                original_name = Path(out["path"]).name
                output_root.mkdir(parents=True, exist_ok=True)
                out["path"] = str(output_root / original_name)
                if isinstance(out.get("intermediate_dir"), str):
                    out["intermediate_dir"] = str(output_root / "intermediates")

        # Save rewritten YAML
        tmp_dir = base_mount / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        new_yaml_path = tmp_dir / f"{Path(yaml_path).stem}_modal.yaml"
        with open(new_yaml_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        return str(new_yaml_path)
    
    def execute(self, operators: List[Dict], dataset: str, prefix: str = "test_pipeline") -> Dict[str, Any]:
        """Execute a pipeline and return results."""
        try:
            yaml_path = self.create_yaml(operators, dataset, prefix)
            yaml_path = self.rewrite_for_modal(yaml_path)
            
            runner = DSLRunner.from_yaml(yaml_path)
            runner.load()
            
            if runner.last_op_container:
                data, _, _ = runner.last_op_container.next()
                runner.save(data)
            
            # Get output path and sample outputs
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            output_path = config['pipeline']['output']['path']
            
            sample_outputs = []
            try:
                with open(output_path, 'r') as f:
                    outputs = json.load(f)
                    sample_outputs = outputs[:3] if isinstance(outputs, list) else [outputs]
            except:
                pass
            
            return {
                "success": True,
                "cost": runner.total_cost,
                "sample_outputs": sample_outputs,
                "error": None,
                "output_path": output_path,
                "yaml_path": yaml_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "cost": 0.0,
                "sample_outputs": [],
                "error": str(e),
                "output_path": None,
                "yaml_path": yaml_path if 'yaml_path' in locals() else None
            }

class DataAnalyzer:
    """Handles data analysis and documentation loading."""
    
    @staticmethod
    def load_documentation(doc_path: str = "experiments/reasoning/data/operators_documentation.txt") -> str:
        """Load operator documentation."""
        doc_file = Path(doc_path)
        if not doc_file.exists():
            doc_file = Path(VOLUME_MOUNT_PATH) / doc_path
        
        with open(doc_file, 'r') as f:
            return f.read()
    
    @staticmethod
    def load_sample_data(dataset: str, limit: int = 5) -> List[Dict]:
        """Load sample data for analysis."""
        data_path = PathResolver.get_data_path(dataset)
        
        if not data_path.exists():
            return []
            
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                return data[:limit] if isinstance(data, list) else [data]
        except Exception:
            return []
    
    @staticmethod
    def analyze_sample_data(sample_data: List[Dict]) -> str:
        """Analyze sample data structure."""
        if not sample_data:
            return "No sample data provided"
        
        first_sample = sample_data[0]
        analysis = [
            f"Number of samples: {len(sample_data)}",
            f"Keys in first sample: {list(first_sample.keys())}"
        ]
        
        # Analyze field types
        for key, value in first_sample.items():
            if isinstance(value, str):
                analysis.append(f"- {key}: string (length: {len(value)})")
            elif isinstance(value, list):
                analysis.append(f"- {key}: list (length: {len(value)})")
            elif isinstance(value, dict):
                analysis.append(f"- {key}: dict (keys: {list(value.keys())})")
            else:
                analysis.append(f"- {key}: {type(value).__name__}")
        
        # Add truncated sample
        sample_json = json.dumps(first_sample, indent=2)
        if len(sample_json) > 2000:
            sample_json = sample_json[:2000] + "\n... (truncated)"
        analysis.append(f"\nFirst sample content:\n{sample_json}")
        
        return "\n".join(analysis)

class AgentCommunicator:
    """Handles LLM communication and JSON parsing."""
    
    def __init__(self, model: str):
        self.model = model
    
    def safe_json_parse(self, response_content: str, fallback: Dict = None) -> Dict:
        """Safely parse JSON response with fallback."""
        try:
            return json.loads(response_content)
        except Exception:
            return fallback or {}
    
    def get_action_decision(self, messages: List[Dict]) -> Optional[AgentAction]:
        """Get agent action decision."""
        try:
            response = completion(
                model=self.model,
                messages=messages,
                azure=True,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                response_format={"type": "json_object"}
            )
            
            decision_json = self.safe_json_parse(response.choices[0].message.content)
            return AgentAction(**decision_json)
        except Exception:
            return None
    
    def get_operators(self, messages: List[Dict], request_msg: str) -> List[Dict]:
        """Get operators from agent."""
        try:
            response = completion(
                model=self.model,
                messages=messages + [{"role": "user", "content": request_msg}],
                azure=True,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                response_format={"type": "json_object"}
            )
            
            result = self.safe_json_parse(response.choices[0].message.content)
            return result.get("operators", [])
        except Exception:
            return []

class SimpleBaselineAgent:
    """Simplified baseline agent that uses tool calling to generate pipelines."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.communicator = AgentCommunicator(model)
        self.documentation = None
        self.original_config = None
        
    def load_resources(self, dataset: str):
        """Load documentation and original pipeline."""
        self.documentation = DataAnalyzer.load_documentation()
        
        # Load original pipeline
        pipeline_path = Path(f"experiments/reasoning/pipelines/{dataset.lower()}.yaml")
        if not pipeline_path.exists():
            pipeline_path = Path(VOLUME_MOUNT_PATH) / f"experiments/reasoning/pipelines/{dataset.lower()}.yaml"
        
        if pipeline_path.exists():
            with open(pipeline_path, 'r') as f:
                self.original_yaml = f.read()
                self.original_config = yaml.safe_load(self.original_yaml)
    
    def _run_individual_evaluation(self, dataset: str, output_path: str, iteration_id: int) -> Dict[str, Any]:
        """Run evaluation for a single pipeline output and return formatted metrics."""
        try:
            evaluate_func = get_evaluate_func(dataset)
            if not evaluate_func:
                return {"accuracy_msg": "No evaluation function available", "accuracy_val": None, "eval_metrics": {}}
            
            # Resolve output path for Modal environment if needed
            resolved_output_path = output_path
            if not Path(output_path).exists() and Path(VOLUME_MOUNT_PATH).exists():
                # Try to find the file in the volume mount
                potential_path = Path(VOLUME_MOUNT_PATH) / Path(output_path).name
                if potential_path.exists():
                    resolved_output_path = str(potential_path)
            
            # Call the evaluation function with proper parameters
            eval_metrics = evaluate_func(f"simple_baseline_iter_{iteration_id}", resolved_output_path)
            
            # Extract the main accuracy metric for this dataset
            dataset_accuracy_metrics = {
                "cuad": "avg_f1",
                "blackvault": "avg_distinct_locations", 
                "game_reviews": "combined_accuracy_score",
                "medec": "combined_score",
                "sustainability": "combined_score",
                "biodex": "avg_rp_at_5",
                "facility": "combined_score"
            }
            
            metric_key = dataset_accuracy_metrics.get(dataset.lower(), "accuracy")
            accuracy_val = eval_metrics.get(metric_key, 0.0)
            
            # Format accuracy message based on dataset
            if dataset.lower() == "cuad":
                accuracy_msg = f"F1: {accuracy_val:.4f}, Precision: {eval_metrics.get('avg_precision', 0):.4f}, Recall: {eval_metrics.get('avg_recall', 0):.4f}"
            elif dataset.lower() == "blackvault":
                accuracy_msg = f"Avg Distinct Locations: {accuracy_val:.4f}, Total Docs: {eval_metrics.get('total_documents', 0)}"
            elif dataset.lower() == "biodex":
                accuracy_msg = f"Avg RP@10: {accuracy_val:.4f}, Avg RP@5: {eval_metrics.get('avg_rp_at_5', 0):.4f}, Term Recall: {eval_metrics.get('avg_term_recall', 0):.4f}"
            elif dataset.lower() == "sustainability":
                accuracy_msg = f"Economic Activity Acc: {accuracy_val:.4f}, Company Name Acc: {eval_metrics.get('company_name_accuracy', 0):.4f}"
            elif dataset.lower() == "facility":
                accuracy_msg = f"Combined Score: {accuracy_val:.4f}, Urgency: {eval_metrics.get('urgency_accuracy', 0):.4f}, Sentiment: {eval_metrics.get('sentiment_accuracy', 0):.4f}, Categories: {eval_metrics.get('categories_accuracy', 0):.4f}"
            else:
                accuracy_msg = f"Accuracy: {accuracy_val:.4f}"
            
            return {
                "accuracy_msg": accuracy_msg,
                "accuracy_val": accuracy_val,
                "eval_metrics": eval_metrics
            }
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {
                "accuracy_msg": f"Evaluation failed: {str(e)}",
                "accuracy_val": None,
                "eval_metrics": {}
            }

    def create_system_prompt(self, dataset: str, baseline_cost: float = None, baseline_accuracy: float = None) -> str:
        """Create system prompt for the agent."""
        
        # Define evaluation metrics explanation for each dataset
        dataset_metrics_info = {
            "cuad": "F1 score, Precision, and Recall for legal clause extraction",
            "blackvault": "Average distinct locations per document for UFO sighting analysis",
            "biodex": "Rank Precision at 5/10 and Term Recall for biochemical reaction prediction",
            "sustainability": "Economic activity accuracy and company name accuracy for sustainability analysis",
            "game_reviews": "Combined accuracy score for game review sentiment analysis",
            "medec": "Combined score for medical entity classification",
            "facility": "Combined score (urgency, sentiment, and categories accuracy) for facility support message classification"
        }
        
        metrics_info = dataset_metrics_info.get(dataset.lower(), "Accuracy metrics specific to the dataset")
        
        return f"""You are a pipeline optimization agent that improves DocETL data processing pipelines.

        You must always respond with valid JSON. You have access to the following actions:
        1. try_pipeline: Test a pipeline configuration and see the results (cost, accuracy, and sample outputs)
        2. return_pipeline: Return the final optimized pipeline configuration

        Always respond in JSON format with:
        {{"action": "try_pipeline", "reasoning": "explanation of why you want to test this pipeline"}}
        OR
        {{"action": "return_pipeline", "reasoning": "explanation of why this is the final pipeline"}}

        When asked for operators, respond with:
        {{"operators": [list of operator dictionaries]}}

        AVAILABLE OPERATORS DOCUMENTATION:
        {self.documentation}

        CURRENT PIPELINE (baseline to improve upon):
        {self.original_yaml}

        BASELINE RESULTS:
        - Cost: ${baseline_cost if baseline_cost is not None else 'N/A'}
        - Accuracy: {baseline_accuracy if baseline_accuracy is not None else 'N/A'}

        EVALUATION METRICS:
        Your pipeline results will be evaluated using: {metrics_info}
        Each pipeline test will provide detailed evaluation metrics to help you optimize.

        YOUR TASK: Improve the pipeline's accuracy by optimizing operators, prompts, models, or adding new operations.

        OPTIMIZATION STRATEGIES:
        1. **Prompt Engineering**: Refine operator prompts for better extraction/classification
        2. **Model Selection**: Try different models from the available list for better performance
        3. **Operator Addition**: Add preprocessing, filtering, or post-processing operators
        4. **Jinja Templating**: Use flexible templating to read more/less context from documents

        AVAILABLE MODELS (use with 'azure/' prefix):
        - azure/gpt-4o-mini
        - azure/gpt-5-nano
        - azure/gpt-5

        Your goal is to beat the baseline accuracy of {baseline_accuracy if baseline_accuracy is not None else 'N/A'}."""

    def run_agent_loop(self, dataset: str, experiment_dir: Path, ground_truth_path: str = None,
                      baseline_cost: float = None, baseline_accuracy: float = None, 
                      baseline_operators: List[Dict] = None, max_iterations: int = 10,
                      all_iteration_results: Optional[List[Dict]] = None, iteration_counter: int = 1) -> tuple[List[Dict], int]:
        """Run the agent optimization loop."""
        
        sample_data = DataAnalyzer.load_sample_data(dataset)
        data_analysis = DataAnalyzer.analyze_sample_data(sample_data)
        executor = PipelineExecutor(experiment_dir)
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": self.create_system_prompt(dataset, baseline_cost, baseline_accuracy)},
            {"role": "user", "content": f"""Dataset: {dataset}

            DATA ANALYSIS:
            {data_analysis}

            Task: Generate a pipeline to process this {dataset} data. You can see the original pipeline YAML above as a baseline. 

            You should:
            1. First try the original pipeline configuration to establish a baseline
            2. Then try to improve it if possible
            3. Return your best pipeline configuration

            Start by trying the original pipeline."""}
            ]
        
        # Track best results
        best_pipeline = baseline_operators or []
        best_cost = baseline_cost or float('inf')
        best_accuracy = baseline_accuracy or -1.0
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Agent iteration {iteration + 1}/{max_iterations}")
            
            # Get agent decision
            decision = self.communicator.get_action_decision(messages)
            if not decision:
                continue
            
            # Print agent's decision and reasoning
            print(f"🤖 Agent Decision: {decision.action}")
            print(f"💭 Reasoning: {decision.reasoning}")
                
            messages.append({"role": "assistant", "content": json.dumps(decision.model_dump())})
            
            if decision.action == "try_pipeline":
                operators = self.communicator.get_operators(
                    messages, 
                    "Provide the operators for the pipeline you want to try. Return JSON in format: {\"operators\": [...]}"
                )
                
                if not operators and self.original_config:
                    operators = self.original_config.get('operations', [])
                
                # Execute and evaluate
                result = self._test_pipeline(operators, dataset, executor, iteration_counter, 
                                           ground_truth_path, experiment_dir, all_iteration_results)
                iteration_counter += 1
                
                # Update best pipeline
                if result["success"] and result.get("accuracy_val") is not None:
                    if result["accuracy_val"] > best_accuracy + 1e-6:
                        best_accuracy = result["accuracy_val"]
                        best_cost = result["cost"]
                        best_pipeline = operators
                
                # Add result to conversation
                formatted_result = self._format_test_result(result)
                messages.append({"role": "user", "content": formatted_result})
                
                # Print iteration results
                print(f"\n📋 Iteration {iteration + 1} Results:")
                formatted_result = self._format_test_result(result)
                print(formatted_result)
                
            elif decision.action == "return_pipeline":
                final_operators = self.communicator.get_operators(
                    messages,
                    "Provide the final operators for the pipeline you want to return. Return JSON in format: {\"operators\": [...]}"
                )
                
                return final_operators or best_pipeline, iteration_counter
        
        return best_pipeline or (self.original_config.get('operations', []) if self.original_config else []), iteration_counter
    
    def _test_pipeline(self, operators: List[Dict], dataset: str, executor: PipelineExecutor, 
                      iteration_id: int, ground_truth_path: str, experiment_dir: Path,
                      all_iteration_results: List[Dict]) -> Dict[str, Any]:
        """Test a pipeline and return results with evaluation."""
        
        print(f"🧪 Testing pipeline with {len(operators)} operators...")
        
        test_result = executor.execute(operators, dataset, f"iteration_{iteration_id}")
        
        # Initialize evaluation results
        accuracy_msg = "N/A"
        accuracy_val = None
        eval_metrics = {}
        
        # Run evaluation immediately after pipeline execution
        if test_result["success"] and test_result.get("output_path"):
            eval_result = self._run_individual_evaluation(dataset, test_result["output_path"], iteration_id)
            accuracy_msg = eval_result["accuracy_msg"]
            accuracy_val = eval_result["accuracy_val"]
            eval_metrics = eval_result["eval_metrics"]
            
            # Add to iteration results for batch evaluation later
            if all_iteration_results is not None:
                all_iteration_results.append({
                    "file_path": test_result["output_path"],
                    "cost": test_result["cost"],
                    "node_id": str(iteration_id)
                })
        
        return {
            **test_result,
            "accuracy_msg": accuracy_msg,
            "accuracy_val": accuracy_val,
            "eval_metrics": eval_metrics
        }
    
    def _format_test_result(self, result: Dict) -> str:
        """Format test result for agent feedback."""
        eval_metrics = result.get('eval_metrics', {})
        sample_outputs = result.get('sample_outputs', [])
        
        # Base message
        message_parts = [
            f"Pipeline test results:",
            f"- Success: {result['success']}",
            f"- Cost: ${result['cost']:.4f}",
            f"- Accuracy: {result['accuracy_msg']}",
            f"- Error: {result.get('error', 'None')}"
        ]
        
        # Add sample outputs with actual content
        if sample_outputs:
            message_parts.append(f"- Sample outputs ({len(sample_outputs)} items):")
            for i, output in enumerate(sample_outputs[:3]):  # Show up to 3 samples
                if isinstance(output, dict):
                    # Truncate the output for readability
                    output_str = json.dumps(output, indent=2)
                    if len(output_str) > 800:
                        output_str = output_str[:800] + "\n... (truncated)"
                    message_parts.append(f"  Sample {i+1}:")
                    # Indent each line of the output
                    for line in output_str.split('\n'):
                        message_parts.append(f"    {line}")
                else:
                    # Handle non-dict outputs
                    output_str = str(output)
                    if len(output_str) > 400:
                        output_str = output_str[:400] + "... (truncated)"
                    message_parts.append(f"  Sample {i+1}: {output_str}")
        else:
            message_parts.append("- Sample outputs: No outputs generated")
        
        # Add detailed evaluation metrics if available
        if eval_metrics:
            message_parts.append("\nDetailed Evaluation Metrics:")
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    if key.startswith('avg_') or 'accuracy' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower():
                        message_parts.append(f"- {key}: {value:.4f}")
                    else:
                        message_parts.append(f"- {key}: {value}")
                elif isinstance(value, list) and len(value) <= 5:  # Show short lists
                    message_parts.append(f"- {key}: {value}")
                elif not isinstance(value, (list, dict)):  # Show simple values
                    message_parts.append(f"- {key}: {value}")
        
        message_parts.append("\nBased on these results, you can either try another pipeline configuration or return the best one you've found.")
        
        return "\n".join(message_parts)

def run_simple_baseline_experiment(dataset: str, output_dir: str = None, model: str = DEFAULT_MODEL,
                                 experiment_name: str = None, ground_truth_path: str = None,
                                 original_query_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run the simple baseline experiment for a dataset."""
    
    # Setup
    if output_dir is None:
        output_dir = os.environ.get('EXPERIMENT_OUTPUT_DIR', DEFAULT_OUTPUT_DIR)
    if experiment_name is None:
        experiment_name = f"simple_baseline_{dataset}"
    
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Running Simple Baseline Experiment")
    print(f"Dataset: {dataset}, Model: {model}, Output: {exp_dir}")
    
    # Initialize agent and executor
    agent = SimpleBaselineAgent(model=model)
    agent.load_resources(dataset)
    executor = PipelineExecutor(exp_dir)
    
    # Use original query result if available, otherwise run baseline
    all_iteration_results = []
    iteration_counter = 0
    baseline_accuracy = None
    
    # Get baseline operations for agent loop
    baseline_ops = agent.original_config.get('operations', []) if agent.original_config else []
    
    if original_query_result and original_query_result["success"]:
        print("✅ Using pre-executed original query result")
        # Use the pre-executed original query result
        baseline_cost = original_query_result["cost"]
        
        # Copy the original output file to our experiment directory for consistency
        baseline_json_path = exp_dir / "baseline_output.json"
        if original_query_result["output_file_path"]:
            import shutil
            try:
                shutil.copy2(original_query_result["output_file_path"], baseline_json_path)
                
                # Run evaluation on the copied baseline
                baseline_eval = agent._run_individual_evaluation(dataset, str(baseline_json_path), 0)
                baseline_accuracy = baseline_eval["accuracy_val"]
                
                # Create baseline result for consistency
                baseline_result_with_eval = {
                    "success": True,
                    "cost": baseline_cost,
                    "output_path": str(baseline_json_path),
                    "accuracy_msg": baseline_eval["accuracy_msg"],
                    "accuracy_val": baseline_eval["accuracy_val"],
                    "eval_metrics": baseline_eval["eval_metrics"]
                }
                
                # Format and display baseline results
                baseline_formatted = agent._format_test_result(baseline_result_with_eval)
                print(f"🏁 Baseline execution results:")
                print(baseline_formatted)
                
                all_iteration_results.append({
                    "file_path": str(baseline_json_path),
                    "cost": baseline_cost,
                    "node_id": str(iteration_counter)
                })
                iteration_counter += 1
                
            except Exception as e:
                print(f"⚠️  Could not copy/evaluate original output file: {e}")
                baseline_accuracy = None
                baseline_cost = original_query_result["cost"]
        else:
            baseline_cost = original_query_result["cost"]
            print(f"✅ Baseline cost: ${baseline_cost:.4f}")
    else:
        print("▶️  Running baseline - original query result not available")
        # Run baseline as before
        baseline_result = executor.execute(baseline_ops, dataset, "baseline")
        
        if baseline_result["success"] and baseline_result.get("output_path"):
            # Run evaluation on baseline
            baseline_eval = agent._run_individual_evaluation(dataset, baseline_result["output_path"], 0)
            baseline_accuracy = baseline_eval["accuracy_val"]
            
            # Update baseline result with evaluation metrics for consistent formatting
            baseline_result_with_eval = {
                **baseline_result,
                "accuracy_msg": baseline_eval["accuracy_msg"],
                "accuracy_val": baseline_eval["accuracy_val"],
                "eval_metrics": baseline_eval["eval_metrics"]
            }
            
            # Format and display baseline results
            baseline_formatted = agent._format_test_result(baseline_result_with_eval)
            print(f"🏁 Baseline execution results:")
            print(baseline_formatted)
            
            all_iteration_results.append({
                "file_path": baseline_result["output_path"],
                "cost": baseline_result["cost"],
                "node_id": str(iteration_counter)
            })
            iteration_counter += 1
            baseline_cost = baseline_result["cost"]
        else:
            print(f"❌ Baseline execution failed: {baseline_result.get('error', 'Unknown error')}")
            baseline_cost = 0.0

    # Agent optimization loop
    operators, iteration_counter = agent.run_agent_loop(
        dataset=dataset,
        experiment_dir=exp_dir,
        ground_truth_path=ground_truth_path,
        baseline_cost=baseline_cost,
        baseline_accuracy=baseline_accuracy,
        baseline_operators=baseline_ops,
        all_iteration_results=all_iteration_results,
        iteration_counter=iteration_counter
    )
    
    # Execute final pipeline
    print("🚀 Executing final pipeline...")
    final_yaml = executor.create_yaml(operators, dataset, "final_pipeline")
    final_yaml = executor.rewrite_for_modal(final_yaml)
    
    try:
        runner = DSLRunner.from_yaml(final_yaml)
        runner.load()
        
        if runner.last_op_container:
            data, _, _ = runner.last_op_container.next()
            runner.save(data)
        
        with open(final_yaml, 'r') as f:
            config = yaml.safe_load(f)
        output_path = config['pipeline']['output']['path']
        
        all_iteration_results.append({
            "file_path": output_path,
            "cost": runner.total_cost,
            "node_id": str(iteration_counter)
        })
        
        results = {
            "success": True,
            "cost": runner.total_cost,
            "output_file": output_path,
            "pipeline_yaml": final_yaml
        }
    except Exception as e:
        results = {
            "success": False,
            "error": str(e),
            "cost": 0.0,
            "pipeline_yaml": final_yaml
        }
    
    # Final evaluation
    if all_iteration_results:
        print(f"📊 Running evaluation on {len(all_iteration_results)} iterations...")
        eval_results, pareto_auc = run_dataset_evaluation(
            dataset=dataset,
            nodes_or_files=all_iteration_results,
            output_path=exp_dir,
            ground_truth_path=ground_truth_path,
            method_name="simple_baseline"
        )
        results.update({"evaluation": eval_results, "pareto_auc": pareto_auc})
    
    # Save results
    results.update({
        "dataset": dataset,
        "operators": operators,
        "experiment_name": experiment_name
    })
    
    print(f"\n✅ Experiment complete! Results saved to: {exp_dir}")
    return results

# Modal functions
@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 2
)
def run_simple_baseline_remote(dataset: str, output_dir: str = None, model: str = DEFAULT_MODEL,
                              experiment_name: str = None, ground_truth_path: str = None,
                              original_query_result: Dict[str, Any] | None = None):
    """Modal remote function for running simple baseline."""
    os.environ["EXPERIMENT_OUTPUT_DIR"] = str(Path(VOLUME_MOUNT_PATH) / "outputs")
    
    resolved_output_dir = PathResolver.resolve_in_volume(output_dir) if output_dir else None
    if resolved_output_dir is None:
        resolved_output_dir = os.environ["EXPERIMENT_OUTPUT_DIR"]
    
    results = run_simple_baseline_experiment(
        dataset=dataset,
        output_dir=resolved_output_dir,
        model=model,
        experiment_name=experiment_name,
        ground_truth_path=ground_truth_path,
        original_query_result=original_query_result
    )
    
    volume.commit()
    return results

@app.local_entrypoint()
def modal_main_simple_baseline(dataset: str, experiment_name: str | None = None,
                              output_dir: str | None = None, model: str = DEFAULT_MODEL,
                              ground_truth: str | None = None,
                              original_query_result: Dict[str, Any] | None = None):
    """Modal entrypoint for simple baseline."""
    run_simple_baseline_remote.remote(
        dataset=dataset,
        output_dir=output_dir,
        model=model,
        experiment_name=experiment_name,
        ground_truth_path=ground_truth,
        original_query_result=original_query_result
    )

def main():
    """Local main function."""
    parser = argparse.ArgumentParser(description="Run simple baseline agent")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["cuad", "reviews", "blackvault", "sustainability", "biodex", "medec", "facility"])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth file")
    
    args = parser.parse_args()
    
    results = run_simple_baseline_experiment(
        dataset=args.dataset,
        output_dir=args.output_dir,
        model=args.model,
        experiment_name=args.experiment_name,
        ground_truth_path=args.ground_truth
    )
    
    if results["success"]:
        print(f"✅ Success! Cost: ${results['cost']:.4f}")
        if "evaluation" in results:
            print(f"Evaluation metrics saved")
    else:
        print(f"❌ Failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()