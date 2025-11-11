"""
Medical Complaints Map-Reduce Precision Experiment (Using Pandas API)

This experiment tests how extraction precision affects reduce quality in map-reduce pipelines.

Ground Truth:
- Map operation: Extract chief complaints using expensive model (gpt-4o)
- Reduce operation: Summarize using expensive model (gpt-4o)

Experiment Conditions:
- Map operation: Extract with varying precision levels using cheap model (gpt-4o-mini)
- Reduce operation: Summarize using expensive model (gpt-4o) - ALWAYS expensive

Metrics:
- Map precision: Compare cheap model extractions to expensive model ground truth
- Reduce quality: Compare experiment summaries to ground truth summary using ROUGE

Uses the DocETL pandas accessor API for cleaner code.
"""

import json
from typing import Any, List, Dict
from rich.console import Console
from rich.table import Table
from rich import box
from dotenv import load_dotenv
import pandas as pd

# Import to register the semantic accessor
import docetl.apis.pd_accessors

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
    rouge_scorer = None

# For semantic similarity
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert-score not available. Install with: pip install bert-score")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    # We'll use OpenAI embeddings since we're already using their models
    from openai import OpenAI
    openai_client = OpenAI()
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Constants
MEDICAL_DATA_PATH = "docs/assets/medical_transcripts.json"
EXPENSIVE_MODEL = "gpt-4o"  # Expensive model for ground truth
CHEAP_MODEL = "gpt-4o-mini"  # Cheap model for experiments

# Ground truth extraction prompt (simple, no explicit precision mention)
GROUND_TRUTH_MAP_PROMPT = """
Extract the chief complaints from this medical transcript.

Transcript: {{ input.src }}

Return the chief complaints.
"""

# Extraction prompts with different precision levels for experiments
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
You are a medical professional tasked with synthesizing a highly detailed and specific chief complaint summary report.

Below are the extracted complaints from multiple medical transcripts:
{% for input in inputs %}
{{ loop.index }}. {{ input.extracted_complaints }}
{% endfor %}

Please perform the following tasks in your summary:
1. Clearly list at least 10 of the most frequently occurring chief complaints across these cases. For each, provide a brief explanation if possible.
2. If fewer than 10 distinct complaints are present, list as many as are found, but explicitly state this.
3. Quantify the frequency of each chief complaint (e.g., "Headache – 4 cases, Cough – 3 cases").
4. Identify and briefly describe at least 3 significant patterns in the symptoms, combinations of complaints, or clinical presentations across the cases.
5. Note and explain any recurring secondary symptoms or contextual factors (such as duration, severity, or relevant patient history) that could impact diagnosis or management.
6. Highlight any outlier or unusual complaints that appear only once, if any.
7. Organize your report into clear sections with informative headings: "Most Common Complaints", "Symptom Patterns", "Relevant Context & Secondary Findings", and "Outlier Complaints".
8. Provide a closing synthesis—a concise, professional paragraph summarizing key findings, suitable for a medical audience.

Be as clear, precise, and comprehensive as possible. Use bullet points and tables if helpful. Write in a professional, clinical style.
"""


def load_medical_data(filepath: str, limit: int = None) -> pd.DataFrame:
    """Load medical transcript data into a DataFrame"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    return pd.DataFrame(data)


def generate_ground_truth(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Generate ground truth using expensive model for both map and reduce.

    Args:
        df: DataFrame with medical transcripts

    Returns:
        Tuple of (ground truth map DataFrame, ground truth summary string)
    """
    console = Console()
    console.print(f"[bold cyan]Generating ground truth: Map with {EXPENSIVE_MODEL}, Reduce with {EXPENSIVE_MODEL}[/bold cyan]")

    # Extract with EXPENSIVE_MODEL using simple extraction prompt
    df_gt_map = df.semantic.map(
        prompt=GROUND_TRUTH_MAP_PROMPT,
        output={"schema": {"extracted_complaints": "string"}},
        model=EXPENSIVE_MODEL
    )

    # Aggregate with EXPENSIVE_MODEL
    result_df = df_gt_map.semantic.agg(
        reduce_prompt=REDUCE_PROMPT,
        output={"schema": {"comprehensive_summary": "string"}},
        reduce_keys=["_all"],
        reduce_kwargs={"model": EXPENSIVE_MODEL}
    )

    gt_summary = result_df.iloc[0]['comprehensive_summary']

    return df_gt_map, gt_summary




def calculate_map_precision(
    df_experiment: pd.DataFrame,
    df_ground_truth: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate precision metrics for the map operation by comparing to ground truth.

    Args:
        df_experiment: DataFrame with extracted complaints from experiment (cheap model)
        df_ground_truth: DataFrame with ground truth extracted complaints (expensive model)

    Returns:
        Dictionary with precision metrics
    """
    # Calculate average length
    avg_length = df_experiment['extracted_complaints'].str.len().mean()

    # Calculate number of tokens (rough estimate using word count)
    avg_tokens = df_experiment['extracted_complaints'].str.split().str.len().mean()

    # For precision, we want to measure what fraction of the experiment's extraction
    # is relevant (i.e., matches the ground truth)
    # Precision = (# relevant extracted) / (# total extracted)
    # We'll use the overlap with ground truth as a proxy

    precision_scores = []

    for idx, row in df_experiment.iterrows():
        gt_text = df_ground_truth.loc[idx, 'extracted_complaints']
        extracted_text = row['extracted_complaints']

        if not extracted_text or not gt_text:
            continue

        # Split into tokens
        gt_tokens = set(gt_text.lower().split())
        extracted_tokens = set(extracted_text.lower().split())

        if len(extracted_tokens) > 0:
            # Precision: what fraction of extracted tokens are in ground truth
            precision = len(gt_tokens & extracted_tokens) / len(extracted_tokens)
            precision_scores.append(precision)

    return {
        "avg_length": avg_length,
        "avg_tokens": avg_tokens,
        "precision": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    }


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Get embedding for a text using OpenAI API"""
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using embeddings.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score (0-1)
    """
    if not EMBEDDINGS_AVAILABLE:
        return 0.0

    try:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)

        # Calculate cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0


def llm_judge_quality(summary: str, ground_truth: str, model: str = "gpt-4o-mini") -> Dict[str, float]:
    """
    Use an LLM to judge the quality of a summary compared to ground truth.

    Args:
        summary: The summary to evaluate
        ground_truth: The ground truth summary
        model: The model to use for judging

    Returns:
        Dictionary with quality scores
    """
    if not EMBEDDINGS_AVAILABLE:
        return {"factual_accuracy": 0.0, "completeness": 0.0, "overall_quality": 0.0}

    judge_prompt = f"""You are an expert medical reviewer evaluating the quality of a chief complaint summary.

GROUND TRUTH SUMMARY (reference):
{ground_truth}

CANDIDATE SUMMARY (to evaluate):
{summary}

Please evaluate the candidate summary on the following criteria, giving a score from 0-10 for each:

1. FACTUAL ACCURACY: Are the complaints and their frequencies correct compared to the ground truth? Do they match the actual data?
2. COMPLETENESS: Does it cover all the important complaints, patterns, and context mentioned in the ground truth?
3. SPECIFICITY: Does it provide specific, actionable information (frequencies, patterns, context) rather than vague generalizations?

Respond ONLY with a JSON object in this exact format (no other text):
{{"factual_accuracy": <score>, "completeness": <score>, "specificity": <score>}}"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0
        )

        result_text = response.choices[0].message.content.strip()
        # Parse JSON response
        import json as json_lib
        scores = json_lib.loads(result_text)

        # Normalize to 0-1 range
        return {
            "factual_accuracy": scores["factual_accuracy"] / 10.0,
            "completeness": scores["completeness"] / 10.0,
            "specificity": scores["specificity"] / 10.0,
            "overall_quality": (scores["factual_accuracy"] + scores["completeness"] + scores["specificity"]) / 30.0
        }
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return {"factual_accuracy": 0.0, "completeness": 0.0, "specificity": 0.0, "overall_quality": 0.0}


def calculate_reduce_quality(
    summary: str,
    ground_truth_summary: str
) -> Dict[str, float]:
    """
    Calculate quality metrics for the reduce operation.

    Args:
        summary: Generated summary
        ground_truth_summary: Ground truth summary from GPT-4o

    Returns:
        Dictionary with quality metrics
    """
    metrics = {}

    # LLM-as-judge (most discriminating for structured outputs)
    if EMBEDDINGS_AVAILABLE:
        judge_scores = llm_judge_quality(summary, ground_truth_summary)
        metrics.update(judge_scores)
    else:
        metrics["factual_accuracy"] = 0.0
        metrics["completeness"] = 0.0
        metrics["specificity"] = 0.0
        metrics["overall_quality"] = 0.0

    # Calculate BERTScore (semantic similarity)
    if BERTSCORE_AVAILABLE:
        try:
            P, R, F1 = bert_score([summary], [ground_truth_summary], lang="en", verbose=False)
            metrics["bertscore_f1"] = float(F1[0])
            metrics["bertscore_precision"] = float(P[0])
            metrics["bertscore_recall"] = float(R[0])
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            metrics["bertscore_f1"] = 0.0
            metrics["bertscore_precision"] = 0.0
            metrics["bertscore_recall"] = 0.0
    else:
        metrics["bertscore_f1"] = 0.0
        metrics["bertscore_precision"] = 0.0
        metrics["bertscore_recall"] = 0.0

    # Calculate ROUGE scores (for reference)
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(ground_truth_summary, summary)
        metrics["rouge1"] = rouge_scores['rouge1'].fmeasure
        metrics["rouge2"] = rouge_scores['rouge2'].fmeasure
        metrics["rougeL"] = rouge_scores['rougeL'].fmeasure
    else:
        metrics["rouge1"] = 0.0
        metrics["rouge2"] = 0.0
        metrics["rougeL"] = 0.0

    # Calculate BLEU score (for reference)
    try:
        reference_tokens = [ground_truth_summary.split()]
        candidate_tokens = summary.split()
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
    except:
        bleu_score = 0.0

    metrics["bleu"] = bleu_score

    return metrics


def run_experiment(
    data_path: str = MEDICAL_DATA_PATH,
    precision_levels: List[str] = ["high_precision", "medium_precision", "low_precision"],
    data_limit: int = None
):
    """
    Run the main experiment comparing precision vs quality tradeoffs.

    Args:
        data_path: Path to medical transcripts JSON file
        precision_levels: List of precision levels to test
        data_limit: Limit number of documents to process
    """
    console = Console()

    # Load data
    console.print(f"[bold]Loading medical data from {data_path}[/bold]")
    df = load_medical_data(data_path, limit=data_limit)
    console.print(f"Loaded {len(df)} medical transcripts\n")

    # Generate ground truth with expensive model
    console.print(f"[bold cyan]Step 1: Generating ground truth with {EXPENSIVE_MODEL}[/bold cyan]")
    df_gt_map, ground_truth_summary = generate_ground_truth(df.copy())
    console.print(f"[bold green]✓ Ground truth generated ({len(ground_truth_summary)} chars)[/bold green]")
    console.print(f"\n[italic]Ground truth summary preview:[/italic]\n{ground_truth_summary[:300]}...\n")

    # Store results
    results = {"ground_truth_summary": ground_truth_summary}

    console.print(f"\n[bold cyan]Step 2: Running experiments (Map: {CHEAP_MODEL}, Reduce: {EXPENSIVE_MODEL})[/bold cyan]\n")

    for precision_level in precision_levels:
        console.print(f"[bold green]Testing precision level: {precision_level}[/bold green]")

        # Extract with CHEAP_MODEL
        df_fresh = df.copy()
        df_extracted = df_fresh.semantic.map(
            prompt=EXTRACTION_PROMPTS[precision_level],
            output={"schema": {"extracted_complaints": "string"}},
            model=CHEAP_MODEL
        )

        # Calculate map precision against ground truth
        map_metrics = calculate_map_precision(df_extracted, df_gt_map)
        console.print(f"  Map Precision: {map_metrics['precision']:.3f}, Avg Length: {map_metrics['avg_length']:.1f}, Avg Tokens: {map_metrics['avg_tokens']:.1f}")

        # Reduce/summarize with EXPENSIVE_MODEL
        console.print(f"  Reducing/summarizing complaints...")
        df_result = df_extracted.semantic.agg(
            reduce_prompt=REDUCE_PROMPT,
            output={"schema": {"comprehensive_summary": "string"}},
            reduce_keys=["_all"],
            reduce_kwargs={"model": EXPENSIVE_MODEL}
        )

        summary = df_result.iloc[0]['comprehensive_summary']

        # Calculate reduce quality against ground truth
        reduce_metrics = calculate_reduce_quality(summary, ground_truth_summary)

        # Show LLM judge scores (most important)
        if EMBEDDINGS_AVAILABLE:
            console.print(f"  Reduce Quality - Overall: {reduce_metrics['overall_quality']:.3f} (Acc: {reduce_metrics['factual_accuracy']:.3f}, Comp: {reduce_metrics['completeness']:.3f}, Spec: {reduce_metrics['specificity']:.3f})")

        # Show other metrics
        if BERTSCORE_AVAILABLE:
            console.print(f"  BERTScore F1: {reduce_metrics['bertscore_f1']:.3f}, ROUGE-L: {reduce_metrics['rougeL']:.3f}\n")
        else:
            console.print(f"  ROUGE-L: {reduce_metrics['rougeL']:.3f}\n")

        # Get costs
        extract_cost = df_extracted.semantic.total_cost
        reduce_cost = df_result.semantic.total_cost

        results[precision_level] = {
            "map_precision": map_metrics,
            "reduce_quality": reduce_metrics,
            "summary": summary,
            "extract_cost": extract_cost,
            "reduce_cost": reduce_cost
        }

    return results


def format_results_table(results: Dict[str, Any]) -> Table:
    """Format experiment results as a Rich table"""
    table = Table(
        title=f"Medical Complaints Precision Experiment\nGT: Map={EXPENSIVE_MODEL}, Reduce={EXPENSIVE_MODEL} | Exp: Map={CHEAP_MODEL}, Reduce={EXPENSIVE_MODEL}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    # Add columns
    table.add_column("Precision Level", style="bold")
    table.add_column("Map Precision", justify="right")
    table.add_column("Avg Tokens", justify="right")

    # LLM judge columns (primary metrics)
    if EMBEDDINGS_AVAILABLE:
        table.add_column("Overall Quality", justify="right")
        table.add_column("Factual Acc", justify="right")
        table.add_column("Completeness", justify="right")
        table.add_column("Specificity", justify="right")

    # Secondary metrics
    if BERTSCORE_AVAILABLE:
        table.add_column("BERTScore F1", justify="right")
    table.add_column("ROUGE-L", justify="right")

    # Add rows
    precision_levels = [k for k in results.keys() if k != "ground_truth_summary"]

    for precision_level in precision_levels:
        data = results[precision_level]
        map_precision = data["map_precision"]
        reduce_quality = data["reduce_quality"]

        row = [
            precision_level,
            f"{map_precision['precision']:.3f}",
            f"{map_precision['avg_tokens']:.0f}",
        ]

        # Add LLM judge scores
        if EMBEDDINGS_AVAILABLE:
            row.extend([
                f"{reduce_quality['overall_quality']:.3f}",
                f"{reduce_quality['factual_accuracy']:.3f}",
                f"{reduce_quality['completeness']:.3f}",
                f"{reduce_quality['specificity']:.3f}"
            ])

        # Add secondary metrics
        if BERTSCORE_AVAILABLE:
            row.append(f"{reduce_quality['bertscore_f1']:.3f}")
        row.append(f"{reduce_quality['rougeL']:.3f}")

        table.add_row(*row)

    return table


def print_analysis(results: Dict[str, Any]):
    """Print analysis of the precision-quality tradeoff"""
    console = Console()

    console.print("\n[bold]Analysis: Map Precision vs Reduce Quality[/bold]\n")

    # Compare across precision levels
    precision_levels = [k for k in results.keys() if k != "ground_truth_summary"]

    precision_data = []
    for precision_level in precision_levels:
        data = results[precision_level]
        map_precision = data["map_precision"]["precision"]
        reduce_quality = data["reduce_quality"]

        # Use LLM judge as primary metric if available
        if EMBEDDINGS_AVAILABLE:
            primary_metric = reduce_quality["overall_quality"]
            primary_metric_name = "Overall Quality"
            factual_acc = reduce_quality["factual_accuracy"]
            completeness = reduce_quality["completeness"]
            specificity = reduce_quality["specificity"]
        elif BERTSCORE_AVAILABLE:
            primary_metric = reduce_quality["bertscore_f1"]
            primary_metric_name = "BERTScore F1"
            factual_acc = completeness = specificity = 0.0
        else:
            primary_metric = reduce_quality["rougeL"]
            primary_metric_name = "ROUGE-L"
            factual_acc = completeness = specificity = 0.0

        precision_data.append({
            "level": precision_level,
            "map_precision": map_precision,
            "primary_metric": primary_metric,
            "primary_metric_name": primary_metric_name,
            "factual_acc": factual_acc,
            "completeness": completeness,
            "specificity": specificity,
            "reduce_rouge": reduce_quality["rougeL"]
        })

    # Sort by map precision (descending - higher precision = more restrictive)
    precision_data.sort(key=lambda x: x["map_precision"], reverse=True)

    console.print("Precision vs Quality:")
    for data in precision_data:
        if EMBEDDINGS_AVAILABLE:
            console.print(f"  {data['level']:20s} | Map Prec: {data['map_precision']:.3f} -> Quality: {data['primary_metric']:.3f} (Acc: {data['factual_acc']:.3f}, Comp: {data['completeness']:.3f}, Spec: {data['specificity']:.3f})")
        else:
            console.print(f"  {data['level']:20s} | Map Precision: {data['map_precision']:.3f} -> {data['primary_metric_name']}: {data['primary_metric']:.3f}")

    # Check if lower map precision leads to higher reduce quality
    if len(precision_data) >= 2:
        highest_map_precision = precision_data[0]  # Most restrictive extraction
        lowest_map_precision = precision_data[-1]  # Least restrictive extraction

        if lowest_map_precision["primary_metric"] > highest_map_precision["primary_metric"]:
            console.print(f"\n[bold green]✓ Lower map precision (more context) leads to higher reduce quality![/bold green]")
            improvement = (lowest_map_precision['primary_metric'] - highest_map_precision['primary_metric']) / highest_map_precision['primary_metric'] * 100
            console.print(f"  {precision_data[0]['primary_metric_name']} Improvement: {improvement:.1f}%")

            # Show breakdown if using LLM judge
            if EMBEDDINGS_AVAILABLE:
                console.print(f"  Breakdown:")
                console.print(f"    Factual Accuracy: {highest_map_precision['factual_acc']:.3f} -> {lowest_map_precision['factual_acc']:.3f}")
                console.print(f"    Completeness: {highest_map_precision['completeness']:.3f} -> {lowest_map_precision['completeness']:.3f}")
                console.print(f"    Specificity: {highest_map_precision['specificity']:.3f} -> {lowest_map_precision['specificity']:.3f}")
        else:
            console.print(f"\n[yellow]✗ Higher map precision (less context) leads to higher reduce quality[/yellow]")


if __name__ == "__main__":
    console = Console()

    console.print(f"[bold green]Medical Complaints Precision Experiment[/bold green]")
    console.print(f"[cyan]Ground Truth: Map={EXPENSIVE_MODEL}, Reduce={EXPENSIVE_MODEL}[/cyan]")
    console.print(f"[cyan]Experiments: Map={CHEAP_MODEL} (varying precision), Reduce={EXPENSIVE_MODEL}[/cyan]")
    console.print(f"[italic]Testing how cheap model's extraction precision affects expensive model's reduce quality[/italic]\n")

    # Run experiment
    results = run_experiment(
        data_path=MEDICAL_DATA_PATH,
        precision_levels=["high_precision", "medium_precision", "low_precision"],
        data_limit=None  # Use full dataset
    )

    # Display results
    console.print("\n[bold]Results Table:[/bold]")
    console.print(format_results_table(results))

    # Print analysis
    print_analysis(results)

    # Print example summaries
    console.print("\n[bold]Example Summaries:[/bold]")
    console.print(f"\n[bold cyan]Ground Truth (Map={EXPENSIVE_MODEL}, Reduce={EXPENSIVE_MODEL}):[/bold cyan]")
    console.print(f"  {results['ground_truth_summary'][:300]}...")

    for precision_level in ["high_precision", "medium_precision", "low_precision"]:
        summary = results[precision_level]["summary"]
        console.print(f"\n[bold green]{precision_level} (Map={CHEAP_MODEL}, Reduce={EXPENSIVE_MODEL}):[/bold green]")
        console.print(f"  {summary[:300]}...")
