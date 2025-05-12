import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
from rich.console import Console
from rich.table import Table
from rich import box
from dotenv import load_dotenv

# Import DSLRunner
from docetl.runner import DSLRunner

# Constants for the experiment
MODELS = ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
EXTRACTION_METHODS = ["line_number", "regex"]
# API base URL for Ollama models
OLLAMA_API_BASE = "https://ucbepic--ollama-service-ollamaservice-server.modal.run"
LOGICAL_FALLACY_PROMPT = """
Identify and extract all instances of logical fallacies in the provided text.
Focus on clear examples of:
1. Ad Hominem: Attacking the person instead of addressing their argument
2. Straw Man: Misrepresenting an opponent's argument to make it easier to attack
3. False Dichotomy: Presenting only two options when others exist
4. Appeal to Authority: Using the opinion of an authority figure as evidence
5. Slippery Slope: Arguing that a small step will lead to extreme consequences
6. Post Hoc Fallacy: Assuming correlation implies causation
7. Circular Reasoning: Using the conclusion as a premise
8. Bandwagon Fallacy: Appealing to popularity or the majority
9. Red Herring: Introducing irrelevant information to distract
10. Hasty Generalization: Drawing conclusions from inadequate samples

For each extracted fallacy, include enough context to understand the fallacy.
"""

# Load environment variables
load_dotenv()

def load_debate_data(filepath: str) -> List[Dict[str, Any]]:
    """Load presidential debate data"""
    with open(filepath, 'r') as f:
        debates = json.load(f)
    
    # Process the debate data to create proper documents
    documents = []
    for debate in debates:
        documents.append({
            "title": debate.get("title", "Presidential Debate"),
            "date": debate.get("date", "Unknown Date"),
            "content": debate.get("content", "")
        })
    
    # # Take first 10 documents
    # documents = documents[:10]
    
    return documents

def run_extraction_with_method(
    model: str,
    method: str, 
    documents: List[Dict[str, Any]], 
    max_workers: int = 64
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """
    Run the extraction operation with a specific model and method.
    
    Args:
        model: The LLM model to use
        method: The extraction method to use ('line_number' or 'regex')
        documents: List of documents to process
        max_workers: Maximum number of worker threads
    
    Returns:
        Tuple containing processed documents, total cost, and metrics
    """
    from docetl.operations.extract import ExtractOperation
    
    # Create a runner configuration
    runner_config = {
        "default_model": model,
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": "/tmp/logical_fallacy_extraction.json"}},
    }
    
    # Add API base URL for Ollama models
    if "ollama" in model:
        runner_config["default_lm_api_base"] = OLLAMA_API_BASE
    
    # Create a real DSLRunner instance
    runner = DSLRunner(
        runner_config,
        max_threads=max_workers,
    )
    
    # Configure the extraction operation
    config = {
        "name": f"fallacy_extract_{model.replace('/', '_')}_{method}",
        "type": "extract",
        "prompt": LOGICAL_FALLACY_PROMPT,
        "document_keys": ["content"],
        "model": model,
        "extraction_method": method,
        "format_extraction": False,  # We want a list to count items
    }
    
    # Add specific completion parameters for Ollama models
    if "ollama" in model:
        config["litellm_completion_kwargs"] = {
            "api_base": OLLAMA_API_BASE
        }
    
    # Create the extraction operation
    op = ExtractOperation(
        runner=runner, 
        config=config, 
        default_model=model, 
        max_threads=max_workers,
        console=Console()
    )
    
    # Measure execution time
    start_time = time.time()
    results, cost = op.execute(documents)
    runtime = time.time() - start_time
    
    # Calculate metrics
    fallacy_counts = []
    avg_lengths = []
    all_fallacies = []
    
    for item in results:
        extraction_key_suffix = f"_extracted_fallacy_extract_{model.replace('/', '_')}_{method}"
        extracted_key = f"content{extraction_key_suffix}"
        fallacies = item.get(extracted_key, [])
        
        fallacy_counts.append(len(fallacies))
        all_fallacies.extend(fallacies)
        
        # Calculate average length
        if fallacies:
            avg_length = sum(len(f) for f in fallacies) / len(fallacies)
            avg_lengths.append(avg_length)
        else:
            avg_lengths.append(0)
    
    # Create metrics dictionary
    metrics = {
        "total_fallacies": sum(fallacy_counts),
        "avg_fallacies_per_doc": sum(fallacy_counts) / len(documents) if documents else 0,
        "max_fallacies_in_doc": max(fallacy_counts) if fallacy_counts else 0,
        "avg_fallacy_length": sum(avg_lengths) / len(avg_lengths) if avg_lengths else 0,
        "runtime": runtime,
        "examples": all_fallacies[:5]  # Store first 5 examples for display
    }
    
    return results, cost, metrics

def run_experiment(debates_file: str, max_workers: int = 64):
    """Run the main experiment comparing different models and extraction methods"""
    console = Console()
    
    # Load the debate data
    documents = load_debate_data(debates_file)
    console.print(f"[bold]Loaded {len(documents)} debate documents[/bold]")
    
    # Nested dictionary to store results by model and method
    results = {model: {} for model in MODELS}
    
    for model in MODELS:
        console.print(f"\n[bold green]Testing model: {model}[/bold green]")
        
        for method in EXTRACTION_METHODS:
            console.print(f"\n[bold blue]  Testing extraction method: {method}[/bold blue]")
            
            # Extract fallacies using the current model and method
            console.print(f"  Processing documents with {model} using {method} method...")
            result_docs, cost, metrics = run_extraction_with_method(
                model, method, documents, max_workers
            )
            
            # Store results
            results[model][method] = {
                **metrics,
                "total_cost": cost
            }
            
            # Display summary
            console.print(f"  Found {metrics['total_fallacies']} fallacies")
            console.print(f"  Average {metrics['avg_fallacies_per_doc']:.2f} fallacies per document")
            console.print(f"  Average fallacy length: {metrics['avg_fallacy_length']:.1f} characters")
            console.print(f"  Runtime: {metrics['runtime']:.2f} seconds")
            console.print(f"  Cost: ${cost:.4f}")
    
    return results

def format_results_table(results: Dict) -> Table:
    """Format results using Rich table"""
    table = Table(
        title="Logical Fallacy Extraction Experiment Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    # Add columns
    table.add_column("Model", style="bold")
    table.add_column("Method", style="bold magenta")
    table.add_column("Total Fallacies", justify="right")
    table.add_column("Avg per Doc", justify="right")
    table.add_column("Max in Doc", justify="right")
    table.add_column("Avg Length", justify="right")
    table.add_column("Runtime (s)", justify="right")
    table.add_column("Total Cost ($)", justify="right")
    
    # Add rows for each model and method
    for model in MODELS:
        for i, method in enumerate(EXTRACTION_METHODS):
            metrics = results[model][method]
            
            # First row for this model gets the model name, others get empty string to avoid repetition
            model_display = model if i == 0 else ""
            
            table.add_row(
                model_display,
                method,
                str(metrics["total_fallacies"]),
                f"{metrics['avg_fallacies_per_doc']:.2f}",
                str(metrics["max_fallacies_in_doc"]),
                f"{metrics['avg_fallacy_length']:.1f}",
                f"{metrics['runtime']:.2f}",
                f"${metrics['total_cost']:.6f}",
            )
        
        # Add a divider between models (except after the last model)
        if model != MODELS[-1]:
            table.add_section()
    
    return table

def print_comparative_conclusion(results: Dict):
    """Print a conclusion comparing the extraction methods across models"""
    console = Console()
    
    console.print("\n[bold]Comparative Conclusion:[/bold]")
    
    # Find the best method for each model
    for model in MODELS:
        line_count = results[model]["line_number"]["total_fallacies"]
        regex_count = results[model]["regex"]["total_fallacies"]
        
        if line_count > regex_count:
            console.print(f"[bold]For {model}:[/bold] Line Number method found more fallacies ({line_count} vs {regex_count})")
        elif regex_count > line_count:
            console.print(f"[bold]For {model}:[/bold] Regex method found more fallacies ({regex_count} vs {line_count})")
        else:
            console.print(f"[bold]For {model}:[/bold] Both methods found the same number of fallacies ({line_count})")
    
    # Find the best overall model
    best_model = None
    best_method = None
    best_count = -1
    
    for model in MODELS:
        for method in EXTRACTION_METHODS:
            count = results[model][method]["total_fallacies"]
            if count > best_count:
                best_count = count
                best_model = model
                best_method = method
    
    console.print(f"\n[bold green]Overall best combination:[/bold green] {best_model} with {best_method} method ({best_count} fallacies)")

if __name__ == "__main__":
    console = Console()
    
    # Run experiment with presidential debate data
    console.print("[bold green]Starting Logical Fallacy Extraction Experiment[/bold green]")
    results = run_experiment("example_data/debates/data.json")
    
    # Print rich table
    console.print("\n[bold]Results Table:[/bold]")
    console.print(format_results_table(results))
    
    # Print example fallacies from each model and method
    console.print("\n[bold]Example Fallacies Extracted:[/bold]")
    for model in MODELS:
        console.print(f"\n[bold green]{model}[/bold green]")
        for method in EXTRACTION_METHODS:
            console.print(f"  [bold magenta]{method.upper()} Method Examples:[/bold magenta]")
            for i, fallacy in enumerate(results[model][method]["examples"][:3]):  # Limit to 3 examples
                console.print(f"  {i+1}. [yellow]{fallacy}[/yellow]")
    
    # Print comparative conclusion
    print_comparative_conclusion(results) 