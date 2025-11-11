"""
Medical Complaints Map-Reduce Precision Experiment

This experiment demonstrates how lower map precision (extracting more context)
can lead to higher reduce quality in medical complaint summarization.

The experiment:
1. Extracts primary complaints from medical transcripts with varying levels of precision
   - High precision: strict extraction of only chief complaints
   - Low precision: extract complaints with surrounding context
2. Varies the fraction of extracted complaints passed to the reduce operation
3. Reduces/summarizes all complaints into a comprehensive summary
4. Tracks:
   - Map precision: How accurately complaints are extracted
   - Reduce quality: Quality of the final summary using ROUGE and BLEU metrics

The hypothesis is that lower map precision (more context) leads to better
reduce quality because the LLM has more information to synthesize.
"""

import os
import json
import time
import random
from typing import Any, List, Dict
from rich.console import Console
from rich.table import Table
from rich import box
from dotenv import load_dotenv

# Import DocETL components
from docetl.runner import DSLRunner
from docetl.operations.map import MapOperation
from docetl.operations.reduce import ReduceOperation

# For metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    print("Warning: Install rouge-score and nltk for metrics: pip install rouge-score nltk")
    rouge_scorer = None

# Load environment variables
load_dotenv()

# Constants
MEDICAL_DATA_PATH = "docs/assets/medical_transcripts.json"
GROUND_TRUTH_MODEL = "gpt-4o"  # High quality model for ground truth
EXPERIMENT_MODEL = "gpt-4o-mini"  # Faster model for experiments
MAX_WORKERS = 64

# Extraction prompts with different precision levels
EXTRACTION_PROMPTS = {
    "high_precision": """
Extract ONLY the primary chief complaint from this medical transcript.
Be very strict - only extract the main reason the patient is visiting.
Do not include any additional context or secondary complaints.

Transcript: {{ input.src }}

Return only the chief complaint as a brief string (1-2 sentences max).
""",
    "medium_precision": """
Extract the primary complaints from this medical transcript.
Include the chief complaint and any closely related symptoms or concerns.

Transcript: {{ input.src }}

Return the complaints with minimal context.
""",
    "low_precision": """
Extract all complaints and concerns mentioned in this medical transcript.
Include the chief complaint, related symptoms, and relevant context from the patient's history.
Include any information that helps understand the patient's condition.

Transcript: {{ input.src }}

Return comprehensive information about the patient's complaints and concerns.
"""
}

# Reduce prompt for summarization
REDUCE_PROMPT = """
You are a medical professional tasked with creating a comprehensive chief complaint summary.

Here are the extracted complaints from multiple medical transcripts:
{% for input in inputs %}
{{ loop.index }}. {{ input.extracted_complaints }}
{% endfor %}

Create a comprehensive and well-structured summary that:
1. Identifies the most common chief complaints
2. Notes any patterns in symptoms or presentations
3. Highlights important medical context

Provide a clear, professional medical summary.
"""


def load_medical_data(filepath: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load medical transcript data"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    return data


def extract_complaints_with_precision(
    documents: List[Dict[str, Any]],
    precision_level: str,
    runner: DSLRunner,
    max_workers: int = MAX_WORKERS
) -> tuple[List[Dict[str, Any]], float]:
    """
    Extract complaints from medical transcripts with specified precision level.

    Args:
        documents: List of medical transcript documents
        precision_level: One of 'high_precision', 'medium_precision', 'low_precision'
        runner: DSLRunner instance
        max_workers: Maximum worker threads

    Returns:
        Tuple of (processed documents, cost)
    """
    from docetl.operations.map import MapOperation

    prompt = EXTRACTION_PROMPTS[precision_level]

    config = {
        "name": f"extract_complaints_{precision_level}",
        "type": "map",
        "prompt": prompt,
        "output": {
            "schema": {
                "extracted_complaints": "string"
            }
        },
        "model": EXPERIMENT_MODEL,
    }

    op = MapOperation(
        runner=runner,
        config=config,
        default_model=EXPERIMENT_MODEL,
        max_threads=max_workers,
        console=Console()
    )

    start_time = time.time()
    results, cost = op.execute(documents)
    runtime = time.time() - start_time

    return results, cost, runtime


def sample_complaints(
    documents: List[Dict[str, Any]],
    fraction: float
) -> List[Dict[str, Any]]:
    """
    Sample a fraction of the extracted complaints.

    Args:
        documents: Documents with extracted complaints
        fraction: Fraction of documents to keep (0.0 to 1.0)

    Returns:
        Sampled documents
    """
    if fraction >= 1.0:
        return documents

    sample_size = max(1, int(len(documents) * fraction))
    return random.sample(documents, sample_size)


def reduce_complaints(
    documents: List[Dict[str, Any]],
    runner: DSLRunner,
    max_workers: int = MAX_WORKERS
) -> tuple[List[Dict[str, Any]], float]:
    """
    Reduce/summarize all complaints into a comprehensive summary.

    Args:
        documents: Documents with extracted complaints
        runner: DSLRunner instance
        max_workers: Maximum worker threads

    Returns:
        Tuple of (reduced documents, cost)
    """
    from docetl.operations.reduce import ReduceOperation

    config = {
        "name": "summarize_complaints",
        "type": "reduce",
        "reduce_key": "_all",  # Reduce all documents together
        "prompt": REDUCE_PROMPT,
        "output": {
            "schema": {
                "comprehensive_summary": "string"
            }
        },
        "model": EXPERIMENT_MODEL,
    }

    op = ReduceOperation(
        runner=runner,
        config=config,
        default_model=EXPERIMENT_MODEL,
        max_threads=max_workers,
        console=Console()
    )

    start_time = time.time()
    results, cost = op.execute(documents)
    runtime = time.time() - start_time

    return results, cost, runtime


def calculate_map_precision(
    extracted_documents: List[Dict[str, Any]],
    ground_truth_documents: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate precision metrics for the map operation.

    Compares extracted complaints against ground truth chief complaints
    using ROUGE scores.

    Args:
        extracted_documents: Documents with extracted complaints
        ground_truth_documents: Original documents with ground truth

    Returns:
        Dictionary with precision metrics
    """
    # Calculate average length regardless of rouge_scorer availability
    avg_length = sum(len(doc.get('extracted_complaints', '')) for doc in extracted_documents) / len(extracted_documents) if extracted_documents else 0.0

    if rouge_scorer is None:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "avg_length": avg_length
        }

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for extracted, ground_truth in zip(extracted_documents, ground_truth_documents):
        # Extract the chief complaint section from ground truth
        gt_text = ground_truth.get('tgt', '')

        # Try to extract just the chief complaint section
        if 'CHIEF COMPLAINT' in gt_text:
            chief_complaint = gt_text.split('CHIEF COMPLAINT')[1].split('\n\n')[0:2]
            chief_complaint = '\n'.join(chief_complaint).strip()
        else:
            # Use first paragraph as fallback
            chief_complaint = gt_text.split('\n\n')[0].strip()

        extracted_text = extracted.get('extracted_complaints', '')

        if extracted_text and chief_complaint:
            scores = scorer.score(chief_complaint, extracted_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
        "avg_length": avg_length
    }


def generate_ground_truth_summary(
    documents: List[Dict[str, Any]],
    runner: DSLRunner,
    max_workers: int = MAX_WORKERS
) -> str:
    """
    Generate a high-quality ground truth summary using GPT-4o.

    Args:
        documents: Documents to summarize
        runner: DSLRunner instance
        max_workers: Maximum worker threads

    Returns:
        Ground truth summary string
    """
    from docetl.operations.reduce import ReduceOperation

    console = Console()
    console.print("[bold cyan]Generating ground truth summary with GPT-4o...[/bold cyan]")

    config = {
        "name": "ground_truth_summary",
        "type": "reduce",
        "reduce_key": "_all",
        "prompt": REDUCE_PROMPT,
        "output": {
            "schema": {
                "comprehensive_summary": "string"
            }
        },
        "model": GROUND_TRUTH_MODEL,
    }

    op = ReduceOperation(
        runner=runner,
        config=config,
        default_model=GROUND_TRUTH_MODEL,
        max_threads=max_workers,
        console=console
    )

    # Extract complaints with low precision (maximum context) for ground truth
    extract_config = {
        "name": "ground_truth_extract",
        "type": "map",
        "prompt": EXTRACTION_PROMPTS["low_precision"],
        "output": {
            "schema": {
                "extracted_complaints": "string"
            }
        },
        "model": GROUND_TRUTH_MODEL,
    }

    from docetl.operations.map import MapOperation
    extract_op = MapOperation(
        runner=runner,
        config=extract_config,
        default_model=GROUND_TRUTH_MODEL,
        max_threads=max_workers,
        console=console
    )

    # Extract with high quality model
    extracted_docs, _ = extract_op.execute(documents)

    # Generate summary
    results, _ = op.execute(extracted_docs)

    if results:
        return results[0].get('comprehensive_summary', '')
    return ""


def calculate_reduce_quality(
    reduced_result: Dict[str, Any],
    ground_truth_summary: str
) -> Dict[str, float]:
    """
    Calculate quality metrics for the reduce operation.

    Compares the comprehensive summary against GPT-4o generated ground truth.

    Args:
        reduced_result: Single result from reduce operation with comprehensive summary
        ground_truth_summary: Ground truth summary generated by GPT-4o

    Returns:
        Dictionary with quality metrics
    """
    if rouge_scorer is None:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

    summary = reduced_result.get('comprehensive_summary', '')

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ground_truth_summary, summary)

    # Calculate BLEU score
    try:
        reference_tokens = [ground_truth_summary.split()]
        candidate_tokens = summary.split()
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
    except:
        bleu_score = 0.0

    return {
        "rouge1": rouge_scores['rouge1'].fmeasure,
        "rouge2": rouge_scores['rouge2'].fmeasure,
        "rougeL": rouge_scores['rougeL'].fmeasure,
        "bleu": bleu_score
    }


def run_experiment(
    data_path: str,
    precision_levels: List[str] = ["high_precision", "medium_precision", "low_precision"],
    sample_fractions: List[float] = [0.25, 0.5, 0.75, 1.0],
    data_limit: int = 20,  # Limit data for faster experimentation
    max_workers: int = MAX_WORKERS
):
    """
    Run the main experiment comparing precision vs quality tradeoffs.

    Args:
        data_path: Path to medical transcripts JSON file
        precision_levels: List of precision levels to test
        sample_fractions: List of fractions to sample for reduce operation
        data_limit: Limit number of documents to process
        max_workers: Maximum worker threads
    """
    console = Console()

    # Load data
    console.print(f"[bold]Loading medical data from {data_path}[/bold]")
    documents = load_medical_data(data_path, limit=data_limit)
    console.print(f"Loaded {len(documents)} medical transcripts")

    # Create runner
    runner_config = {
        "default_model": EXPERIMENT_MODEL,
        "operations": [],
        "pipeline": {"steps": [], "output": {"path": "/tmp/medical_complaints.json"}},
    }
    runner = DSLRunner(runner_config, max_threads=max_workers)

    # Generate ground truth summary with high-quality model
    console.print(f"\n[bold cyan]Step 1: Generating ground truth summary with {GROUND_TRUTH_MODEL}[/bold cyan]")
    ground_truth_summary = generate_ground_truth_summary(documents, runner, max_workers)
    console.print(f"[bold green]✓ Ground truth generated ({len(ground_truth_summary)} chars)[/bold green]")
    console.print(f"\n[italic]Ground truth preview:[/italic]\n{ground_truth_summary[:300]}...\n")

    # Store results
    results = {
        "ground_truth_summary": ground_truth_summary
    }

    for precision_level in precision_levels:
        console.print(f"\n[bold green]Testing precision level: {precision_level}[/bold green]")

        # Extract complaints with this precision level
        console.print(f"  Extracting complaints...")
        extracted_docs, extract_cost, extract_time = extract_complaints_with_precision(
            documents, precision_level, runner, max_workers
        )

        # Calculate map precision
        map_metrics = calculate_map_precision(extracted_docs, documents)
        console.print(f"  Map Precision - ROUGE-L: {map_metrics['rougeL']:.3f}, Avg Length: {map_metrics['avg_length']:.1f}")

        results[precision_level] = {
            "extract_cost": extract_cost,
            "extract_time": extract_time,
            "map_precision": map_metrics,
            "sample_results": {}
        }

        # Test different sampling fractions
        for fraction in sample_fractions:
            console.print(f"\n[bold blue]  Testing sample fraction: {fraction}[/bold blue]")

            # Sample complaints
            sampled_docs = sample_complaints(extracted_docs, fraction)
            console.print(f"    Sampled {len(sampled_docs)} documents")

            # Reduce/summarize
            console.print(f"    Reducing/summarizing complaints...")
            reduced_results, reduce_cost, reduce_time = reduce_complaints(
                sampled_docs, runner, max_workers
            )

            # Calculate reduce quality (compare against ground truth summary)
            if reduced_results:
                reduce_metrics = calculate_reduce_quality(reduced_results[0], ground_truth_summary)
                console.print(f"    Reduce Quality - ROUGE-L: {reduce_metrics['rougeL']:.3f}, BLEU: {reduce_metrics['bleu']:.3f}")

                results[precision_level]["sample_results"][fraction] = {
                    "reduce_cost": reduce_cost,
                    "reduce_time": reduce_time,
                    "reduce_quality": reduce_metrics,
                    "num_samples": len(sampled_docs),
                    "summary": reduced_results[0].get('comprehensive_summary', '')[:200] + '...'
                }

    return results


def format_results_table(results: Dict[str, Any]) -> Table:
    """Format experiment results as a Rich table"""
    table = Table(
        title="Medical Complaints Precision vs Quality Experiment",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    # Add columns
    table.add_column("Precision Level", style="bold")
    table.add_column("Map ROUGE-L", justify="right")
    table.add_column("Avg Extract Length", justify="right")
    table.add_column("Sample Fraction", justify="right")
    table.add_column("Reduce ROUGE-L", justify="right")
    table.add_column("Reduce BLEU", justify="right")
    table.add_column("Total Cost ($)", justify="right")

    # Add rows
    for precision_level, data in results.items():
        map_precision = data["map_precision"]

        for i, (fraction, sample_data) in enumerate(data["sample_results"].items()):
            reduce_quality = sample_data["reduce_quality"]
            total_cost = data["extract_cost"] + sample_data["reduce_cost"]

            # Only show precision level on first row for this precision level
            precision_display = precision_level if i == 0 else ""
            map_rouge_display = f"{map_precision['rougeL']:.3f}" if i == 0 else ""
            avg_len_display = f"{map_precision['avg_length']:.0f}" if i == 0 else ""

            table.add_row(
                precision_display,
                map_rouge_display,
                avg_len_display,
                f"{fraction:.2f}",
                f"{reduce_quality['rougeL']:.3f}",
                f"{reduce_quality['bleu']:.3f}",
                f"${total_cost:.4f}"
            )

        # Add section divider
        table.add_section()

    return table


def print_analysis(results: Dict[str, Any]):
    """Print analysis of the precision-quality tradeoff"""
    console = Console()

    console.print("\n[bold]Analysis: Map Precision vs Reduce Quality[/bold]")

    # For each sample fraction, compare across precision levels
    sample_fractions = list(next(iter(results.values()))["sample_results"].keys())

    for fraction in sample_fractions:
        console.print(f"\n[bold cyan]Sample Fraction: {fraction}[/bold cyan]")

        precision_data = []
        for precision_level in results.keys():
            sample_data = results[precision_level]["sample_results"][fraction]
            map_rouge = results[precision_level]["map_precision"]["rougeL"]
            reduce_rouge = sample_data["reduce_quality"]["rougeL"]
            reduce_bleu = sample_data["reduce_quality"]["bleu"]

            precision_data.append({
                "level": precision_level,
                "map_rouge": map_rouge,
                "reduce_rouge": reduce_rouge,
                "reduce_bleu": reduce_bleu
            })

        # Sort by map precision (ascending)
        precision_data.sort(key=lambda x: x["map_rouge"])

        console.print("  Precision vs Quality:")
        for data in precision_data:
            console.print(f"    {data['level']:20s} | Map ROUGE-L: {data['map_rouge']:.3f} -> Reduce ROUGE-L: {data['reduce_rouge']:.3f}, BLEU: {data['reduce_bleu']:.3f}")

        # Check if lower map precision leads to higher reduce quality
        if len(precision_data) >= 2:
            lowest_precision = precision_data[0]
            highest_precision = precision_data[-1]

            if lowest_precision["reduce_rouge"] > highest_precision["reduce_rouge"]:
                console.print(f"  [bold green]✓ Lower map precision leads to higher reduce quality![/bold green]")
                console.print(f"    Improvement: {(lowest_precision['reduce_rouge'] - highest_precision['reduce_rouge']) / highest_precision['reduce_rouge'] * 100:.1f}%")
            else:
                console.print(f"  [yellow]✗ Higher map precision leads to higher reduce quality[/yellow]")


if __name__ == "__main__":
    console = Console()

    console.print("[bold green]Starting Medical Complaints Precision Experiment[/bold green]\n")

    # Run experiment
    results = run_experiment(
        data_path=MEDICAL_DATA_PATH,
        precision_levels=["high_precision", "medium_precision", "low_precision"],
        sample_fractions=[0.5, 1.0],  # Test with half and all samples
        data_limit=15,  # Limit for faster experimentation
        max_workers=64
    )

    # Display results
    console.print("\n[bold]Results Table:[/bold]")
    console.print(format_results_table(results))

    # Print analysis
    print_analysis(results)

    # Print example summaries
    console.print("\n[bold]Example Summaries (Full Sample):[/bold]")
    for precision_level in results.keys():
        if 1.0 in results[precision_level]["sample_results"]:
            summary = results[precision_level]["sample_results"][1.0]["summary"]
            console.print(f"\n[bold green]{precision_level}:[/bold green]")
            console.print(f"  {summary}")
