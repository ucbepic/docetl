"""
Debug script to see actual extractions vs ground truth for medical complaints
"""

import json
from docetl.runner import DSLRunner
from docetl.operations.map import MapOperation
from rich.console import Console
from rich.table import Table
from rich import box

# Load first 5 documents
data = json.load(open('docs/assets/medical_transcripts.json'))[:5]

# Create runner
runner = DSLRunner({
    'default_model': 'gpt-4o-mini',
    'operations': [],
    'pipeline': {'steps': [], 'output': {'path': '/tmp/test.json'}}
}, max_threads=64)

# High precision extraction
config = {
    'name': 'test_extract',
    'type': 'map',
    'prompt': '''Extract ONLY the primary chief complaint from this medical transcript.
Be very strict - only extract the main reason the patient is visiting.
Do not include any additional context or secondary complaints.

Transcript: {{ input.src }}

Return only the chief complaint as a brief string (1-2 sentences max).''',
    'output': {'schema': {'extracted_complaints': 'string'}},
    'model': 'gpt-4o-mini'
}

console = Console()
console.print("[bold]Extracting complaints...[/bold]")
op = MapOperation(runner=runner, config=config, default_model='gpt-4o-mini', max_threads=64, console=console)
results, cost = op.execute(data)

# Create comparison table
table = Table(
    title="Extraction vs Ground Truth Comparison",
    box=box.ROUNDED,
    show_header=True,
    header_style="bold cyan"
)

table.add_column("Doc", style="bold", width=3)
table.add_column("Ground Truth", style="green", width=30)
table.add_column("Extracted", style="yellow", width=30)
table.add_column("Match?", justify="center", width=8)

for i, (result, original) in enumerate(zip(results, data)):
    # Extract ground truth chief complaint
    gt_text = original.get('tgt', '')
    if 'CHIEF COMPLAINT' in gt_text:
        gt_complaint = gt_text.split('CHIEF COMPLAINT')[1].split('\n\n')[0:2]
        gt_complaint = '\n'.join(gt_complaint).strip()
    else:
        gt_complaint = "N/A"

    extracted = result.get('extracted_complaints', '')

    # Check if they match (case insensitive, stripped)
    match = "✓" if gt_complaint.lower().strip() == extracted.lower().strip() else "✗"

    table.add_row(
        str(i+1),
        gt_complaint[:50] + "..." if len(gt_complaint) > 50 else gt_complaint,
        extracted[:50] + "..." if len(extracted) > 50 else extracted,
        match
    )

console.print(table)

# Show full details for mismatches
console.print("\n[bold]Detailed Mismatches:[/bold]")
for i, (result, original) in enumerate(zip(results, data)):
    gt_text = original.get('tgt', '')
    if 'CHIEF COMPLAINT' in gt_text:
        gt_complaint = gt_text.split('CHIEF COMPLAINT')[1].split('\n\n')[0:2]
        gt_complaint = '\n'.join(gt_complaint).strip()
    else:
        gt_complaint = "N/A"

    extracted = result.get('extracted_complaints', '')

    if gt_complaint.lower().strip() != extracted.lower().strip():
        console.print(f"\n[bold red]Document {i+1}:[/bold red]")
        console.print(f"[green]Ground Truth:[/green] {gt_complaint}")
        console.print(f"[yellow]Extracted:[/yellow] {extracted}")
