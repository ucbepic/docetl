import os
import time
import random
import json
from typing import List, Dict, Set
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv
import concurrent.futures
from threading import Lock
from rich.console import Console
from rich.table import Table
from rich import box
import litellm

# litellm.set_verbose=True

# Load environment variables
load_dotenv()

# Constants for the experiment
FRUITS_VEGETABLES = [
    "apple", "banana", "carrot", "durian", "eggplant",
    "fig", "grape", "honeydew", "iceberg lettuce", "jackfruit",
    "kale", "lemon", "mango", "nectarine", "orange",
    "papaya", "quince", "radish", "spinach", "tomato",
    "apricot", "blackberry", "cucumber", "dragonfruit", "endive",
    "fennel", "grapefruit", "horseradish", "indian gooseberry", "jicama",
    "kohlrabi", "lime", "mushroom", "napa cabbage", "okra",
    "pear", "quinoa", "raspberry", "squash", "turnip"
]

# Models to test
MODELS = [
    "azure/gpt-4o-mini",
    "deepseek/deepseek-chat",
    # "lm_studio/hugging-quants/llama-3.2-3b-instruct",
    # "lm_studio/qwen2.5-7b-instruct-1m",
]
SYSTEM_PROMPT = (
    "You are a helpful assistant, helping the user make sense of their data. "
    "The dataset description is: a collection of presidential debate transcripts. "
    "You will be performing a map operation (1 input:1 output). "
    "The user will specify a task for you to perform on the provided data, as precisely and "
    "exhaustively (i.e., high recall) as possible. The result should be a structured "
    "output that you will send back to the user, with the `send_output` function. "
    "Do not influence your answers too much based on the `send_output` function "
    "parameter names; just use them to send the result back to the user."
)
STRUCTURED_SYSTEM_PROMPT = (
    "You are a helpful assistant, helping the user make sense of their data. "
    "The dataset description is: a collection of presidential debate transcripts. "
    "You will be performing a map operation (1 input:1 output). "
    "The user will specify a task for you to perform on the provided data, as precisely and "
    "exhaustively (i.e., high recall) as possible. The result should be a structured "
    "output that you will send back to the user, in JSON format. Do not influence your answers "
    "too much based on the JSON schema names. "
    "The JSON schema is: {schema}"
)

PROMPT_TEMPLATE = (
    "I have injected several fruit and vegetable names into this transcript. "
    "Your task is to find and list all fruits and vegetables mentioned. "
    "Only include items that are actually fruits or vegetables, "
    "not metaphors or company names: {text}"
)

class FoundItems(BaseModel):
    fruits_and_vegetables: List[str]

def load_and_augment_debates(filepath: str, num_samples: int = 20, frac_doc_content: float = 0.5) -> List[Dict[str, any]]:
    """Load debates and augment them with fruits/vegetables"""
    with open(filepath, 'r') as f:
        debates = json.load(f)
    
    # Randomly sample debates if there are more than we need
    if len(debates) > num_samples:
        debates = random.sample(debates, num_samples)
    
    augmented_data = []
    for debate in debates:
        # Get the original content
        content = debate['content']
        
        # Take only the first frac_doc_content of the content
        content = content[:int(len(content) * frac_doc_content)]
        
        words = content.split()
        ground_truth = set()
        
        # Insert random fruits/vegetables
        num_insertions = random.randint(1, 3)
        for _ in range(num_insertions):
            item = random.choice(FRUITS_VEGETABLES)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, item)
            ground_truth.add(item)
        
        augmented_data.append({
            "text": " ".join(words),
            "inserted_items": list(ground_truth)
        })
    
    return augmented_data

def evaluate_structured_output(model: str, text: str) -> tuple[Set[str], float, float]:
    """Evaluate using structured output approach"""
    start_time = time.time()
    
    messages = [{
        "role": "system",
        "content": STRUCTURED_SYSTEM_PROMPT.format(schema=FoundItems.model_json_schema())
    }, {
        "role": "user",
        "content": PROMPT_TEMPLATE.format(text=text)
    }]
    
    response = None
    json_schema_object = {
      "type": "json_schema",
      "json_schema": {
        "name": "send_output",
        "strict": "true",
        "schema": {
          "type": "object",
          "properties": {
            "fruits_and_vegetables": {
              "type": "array",
              "items": {
                "type": "string"
              }
            }
          },
          "required": ["fruits_and_vegetables"]
        }
      },
      "temperature": 0
    }
    if "gpt" in model:
        json_schema_object = FoundItems
    if "deepseek" in model:
        json_schema_object = {"type": "json_object"}
    
    try:
        response = completion(
            model=model,
            messages=messages,
            response_format=json_schema_object,
            num_retries=3,
            temperature=1.0,
            max_tokens=500,
        )
        extracted_items = set(json.loads(response.choices[0].message.content)["fruits_and_vegetables"])
        cost = response._hidden_params["response_cost"]
    except Exception as e:
        print(f"Error with structured output for {model}: {e}; {response}")
        extracted_items = set()
        cost = 0.0
    
    runtime = time.time() - start_time
    return extracted_items, runtime, cost

def evaluate_tool_calling(model: str, text: str) -> tuple[Set[str], float, float]:
    """Evaluate using tool calling approach"""
    start_time = time.time()
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(text=text)
        }
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "send_output",
            "description": "Send output back to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "fruits_and_vegetables": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["fruits_and_vegetables"]
            }
        },
        "additionalProperties": False
    }]

    
    try:
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "send_output"}},
            num_retries=3,
            max_tokens=500,
            temperature=1.0,
        )
        
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            extracted_items = set(
                json.loads(tool_calls[0].function.arguments)["fruits_and_vegetables"]
            )
        else:
            extracted_items = set()
        cost = response._hidden_params["response_cost"]
    except Exception as e:
        print(f"Error with tool calling for {model}: {e}")
        extracted_items = set()
        cost = 0.0
    
    runtime = time.time() - start_time
    return extracted_items, runtime, cost

def calculate_metrics(extracted: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    if not extracted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not extracted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    true_positives = len(extracted.intersection(ground_truth))
    precision = true_positives / len(extracted) if extracted else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def process_document(args) -> Dict[str, any]:
    """Process a single document with both approaches"""
    model, doc, i, total = args
    print(f"Processing document {i+1}/{total}")
    
    # Test structured output
    extracted_structured, runtime_structured, cost_structured = evaluate_structured_output(
        model, doc["text"]
    )
    metrics_structured = calculate_metrics(
        extracted_structured, set(doc["inserted_items"])
    )
    
    # Test tool calling
    extracted_tool, runtime_tool, cost_tool = evaluate_tool_calling(
        model, doc["text"]
    )
    metrics_tool = calculate_metrics(
        extracted_tool, set(doc["inserted_items"])
    )
    
    return {
        "structured": {
            **metrics_structured,
            "runtime": runtime_structured,
            "cost": cost_structured if cost_structured else 0.0
        },
        "tool": {
            **metrics_tool,
            "runtime": runtime_tool,
            "cost": cost_tool if cost_tool else 0.0
        }
    }

def run_experiment(debates_file: str, num_samples: int = 20, max_workers: int = 64):
    """Run the main experiment with parallel processing across different document fractions"""
    fractions = [0.1]
    results = {
        model: {
            fraction: {"structured": {}, "tool": {}} 
            for fraction in fractions
        } for model in MODELS
    }
    results_lock = Lock()
    
    for fraction in fractions:
        print(f"\nTesting with document fraction: {fraction}")
        # Load and augment real debate data with current fraction
        documents = load_and_augment_debates(debates_file, num_samples, fraction)
        
        for model in MODELS:
            print(f"Testing model: {model}")
            
            # Prepare arguments for parallel processing
            args_list = [(model, doc, i, len(documents)) 
                        for i, doc in enumerate(documents)]
            
            # Process documents in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {executor.submit(process_document, args): args 
                               for args in args_list}
                
                for future in concurrent.futures.as_completed(future_to_doc):
                    try:
                        doc_results = future.result()
                        
                        # Thread-safe results aggregation
                        with results_lock:
                            for approach in ["structured", "tool"]:
                                for metric, value in doc_results[approach].items():
                                    results[model][fraction][approach][metric] = results[model][fraction][approach].get(
                                        metric, []
                                    ) + [value]
                            
                            # Save intermediate results
                            with open('experiments/results.json', 'w') as f:
                                json.dump(results, f, indent=2)
                                
                    except Exception as e:
                        args = future_to_doc[future]
                        print(f"Error processing document {args[2]+1}: {e}")
    
    return results

def format_results_table(results: Dict) -> Table:
    """Format results using Rich table"""
    table = Table(
        title="Experiment Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    # Add columns
    table.add_column("Model", style="bold")
    table.add_column("Doc %", justify="right")
    table.add_column("Approach", style="magenta")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Avg Runtime", justify="right")
    table.add_column("Avg Cost ($)", justify="right")
    
    for model in results:
        for fraction in sorted(results[model].keys()):
            for approach in ["structured", "tool"]:
                metrics = results[model][fraction][approach]
                table.add_row(
                    model,
                    f"{fraction*100:>3.0f}%",
                    approach,
                    f"{sum(metrics['precision']) / len(metrics['precision']):.3f}",
                    f"{sum(metrics['recall']) / len(metrics['recall']):.3f}",
                    f"{sum(metrics['f1']) / len(metrics['f1']):.3f}",
                    f"{sum(metrics['runtime']) / len(metrics['runtime']):.3f}s",
                    f"${sum(metrics['cost']) / len(metrics['cost']):.4f}",
                )
            # Add a divider after each fraction except the last one
            if fraction != max(results[model].keys()):
                table.add_row()
        # Add a section divider after each model except the last one
        if model != list(results.keys())[-1]:
            table.add_section()
    
    return table

if __name__ == "__main__":
    # Run experiment with real debate data
    results = run_experiment(
        debates_file="example_data/debates/data.json",
        num_samples=30
    )
    
    # Print rich table
    console = Console()
    console.print("\nResults Table:")
    console.print(format_results_table(results))
