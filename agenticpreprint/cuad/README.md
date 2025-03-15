# CUAD Experiment

The data we use is the first 50 articles from the dataset.

## Plan Files Overview

- **agenticpreprint/cuad/docetl_base.yaml** - The original plan configuration file.

- **agenticpreprint/cuad/docetl_preprint_opt.yaml** - The optimized plan that was reported in our Fall 2024 preprint.

- **agenticpreprint/cuad/docetl_base_opt.yaml** - The optimized plan generated on March 7th 2025 using the command `docetl build agenticpreprint/cuad/docetl_base.yaml`.

All outputs in the repo were generated on March 7, 2025.

## Setup Requirements

- You need to have an OpenAI API key in your `.env` file.

## Usage Instructions

### Build the Optimized Plan

To build an optimized plan from the base configuration:

```bash
docetl build agenticpreprint/cuad/docetl_base.yaml
```

**Important Note**: The optimization process for this experiment is extremely slow, even with recursive rewrites turned off. It took approximately 1 hour to build the plan.

### Execute a Plan

To run a plan, use:

```bash
docetl run <plan_file>
```

Replace `<plan_file>` with one of the YAML files listed above.

## Potential Issues

- You can easily run into OpenAI rate limits during the optimization process.
- Consider running the optimization in smaller batches if you encounter rate limits.
