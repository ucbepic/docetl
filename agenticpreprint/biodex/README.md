# BioDex

## Setup Requirements

- You need to have an OpenAI API key in your `.env` file.

## Usage Instructions

The data we use is the first 250 articles from the dataset on Huggingface.

### Build the Optimized Plan

Run the following command to build the optimizer:

```bash
docetl build agenticpreprint/biodex/docetl_base.yaml
```

This will use GPT-4o for document rewrites and GPT-4o-mini as the LLM judge model.
Note that it will require your manual input; to confirm any blocking rules generated. For the plans in the repo I have just said "yes" to accept all 5 blocking rules generated.

An older version of the codebase only picked up until the first 2 blocking rules, automatically (so old plans don't have as many blocking rules). But we tried to make it more "human-in-the-loop" as per user request.

### Execute a Plan

To run a plan, use:

```bash
docetl run <plan_file>
```

Note that you will be prompted to confirm whether you want to execute all the LLM calls in the comparison operation, because there are so many (12k) calls to execute. You will have to say yes.

### Print Results (Don't need to run above plans)

To print results, run

```bash
python agenticpreprint/biodex/compute_rp.py
```

This reads from the 5 optimized trials as of 3-14; it prints the min, max, and average RP@x's.

## Notes

- You may run into OpenAI rate limits during execution.

## Plan Versions

- `agenticpreprint/biodex/optimized_aug2024.yaml` and `agenticpreprint/biodex/optimized_jan2025.yaml` are older versions of the plan.
- The files `docet_base_opt_0` through `docet_base_opt_4` were generated on March 14, 2025.
- While the plans may have been generated at different times, all outputs in the repo were generated on March 14, 2025.

## Pipeline Analysis

### Results

```
+----------------------------------------+--------+---------+---------+
| Model                                  |   RP@5 |   RP@10 |   RP@25 |
+========================================+========+=========+=========+
| DocETL_0                               | 0.2199 |  0.2720 |  0.3044 |
+----------------------------------------+--------+---------+---------+
| DocETL_1                               | 0.1810 |  0.1601 |  0.1562 |
+----------------------------------------+--------+---------+---------+
| DocETL_2                               | 0.2209 |  0.2596 |  0.2934 |
+----------------------------------------+--------+---------+---------+
| DocETL_3                               | 0.2187 |  0.2450 |  0.2609 |
+----------------------------------------+--------+---------+---------+
| DocETL_4                               | 0.2379 |  0.2390 |  0.2344 |
+----------------------------------------+--------+---------+---------+
| Combined DocETL (3-14-2025 plans only) | 0.2160 |  0.2360 |  0.2509 |
+----------------------------------------+--------+---------+---------+
| Lotus                                  | 0.2115 |  0.2162 |  0.2199 |
+----------------------------------------+--------+---------+---------+
| docetl_aug2024                         | 0.2741 |  0.3195 |  0.3706 |
+----------------------------------------+--------+---------+---------+
| docetl_jan2025                         | 0.2573 |  0.3150 |  0.3765 |
+----------------------------------------+--------+---------+---------+
```

### Qualitative Analysis

#### Old vs. New Pipelines

The aug2024 and jan2025 pipelines have pretty similar prompts with wording differences. The critical thing is that their comparison prompts don't get rewritten to use the map operation output. This "mistake" by the LLM agent actually gives the best results.

For DocETL_0 through DocETL_4, all pipelines get a new map operation, and all comparison prompts get changed to either:
- Use the map operation's output alongside the full article text
- Use the map output instead of the article text

When using map outputs instead of full article text in the comparison, the results tank (e.g., see DocETL_1). When using map outputs alongside full text, results are better but still  worse than the old pipelines.

The map operations that get synthesized either:
- Summarize the article
- Extract reactions from the article

The synthesized map operations are not "grounded" with any reactions.

We also do not employ any prompting strategies.

#### Discussion

It's strange that adding more information (through map operations) actually **hurts** performance. There is a 7 point difference at RP@25 between the best old pipeline (jan2025 at 0.3765; January 2025) and the best new one (DocETL_0 at 0.3044). But this highlights the unpredictable nature of LLM agents. Also, the dataset is labeled very poorly.

But overall, it's pretty wild that removing one instruction from the prompt can give such a huge performance boost. 

