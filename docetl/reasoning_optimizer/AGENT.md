# DocETL Reasoning Optimizer - Agent Guide

## Overview
The reasoning optimizer uses a graph-search algorithm with LLM-based directives to rewrite and optimize DocETL pipelines. This directory contains the directive system for pipeline transformations.

## Creating New Directives - Interactive Process

As an AI agent, I can guide you through creating new rewrite directives step-by-step. Here's the process:

### 1. Initial Consultation
- **What I'll ask**: Describe what transformation you want the directive to perform
- **What you provide**: High-level description of the directive's purpose and when to use it
- **What I'll do**: Analyze existing directives and suggest the best approach

### 2. Schema Design Phase
- **What I'll ask**: Confirm the configuration parameters needed for the directive
- **What I'll propose**: The instantiate schema structure (Pydantic models) that the LLM agent will output
- **What you confirm**: Whether the schema captures all necessary parameters

### 3. Directive Specification
- **What I'll propose**:
  - `name`: Technical identifier for the directive
  - `formal_description`: Brief transformation pattern (e.g., "Op => Code Map -> Op")
  - `nl_description`: Natural language explanation
  - `when_to_use`: Specific use cases
- **What you confirm**: Whether these descriptions accurately capture the directive's purpose

### 4. Example Creation
- **What I'll propose**: Example showing original operation and expected instantiate schema output
- **What you confirm**: Whether the example demonstrates the directive correctly

### 5. Test Case Design
- **What I'll propose**: Test cases with input operations and expected behaviors
- **What you confirm**: Whether test cases cover important scenarios

### 6. Implementation Review
- **What I'll show**: Complete directive implementation including:
  - Schema classes in `instantiate_schemas.py`
  - Directive class with all required methods
  - Registration in `__init__.py`
  - Apply tests in `tests/reasoning_optimizer/test_directive_apply.py`
- **What you confirm**: Final approval before implementation

## Existing Directive Patterns

### Single Operation Modification
- **Gleaning**: Adds validation loops (`validation_prompt`, `num_rounds`)
- **Change Model**: Switches LLM model (`model`)
- **Deterministic Doc Compression**: Adds regex-based preprocessing

### Operation Replacement
- **Chaining**: Replaces complex operation with sequential simpler ones
- **Isolating Subtasks**: Breaks operation into independent parallel tasks

### Pipeline Preprocessing
- **Doc Summarization**: Adds document summarization before main processing

## Key Files and Structure
- `directives/`: Individual directive implementations
- `instantiate_schemas.py`: Pydantic schemas for LLM outputs
- `agent.py`: Core MCTS agent that applies directives
- `op_descriptions.py`: Operation type descriptions for the agent

## Testing Commands

### Instantiation Tests
- Single directive: `python experiments/reasoning/run_tests.py --directive=directive_name`
- All directives: `python experiments/reasoning/run_tests.py`

### Apply Tests
- All directive apply methods: `python tests/reasoning_optimizer/test_directive_apply.py`

### Full MCTS
- Complete optimization: `python experiments/reasoning/run_mcts.py`

## Workflow for New Directive
1. Describe desired transformation → I analyze and propose approach
2. Confirm schema design → I implement Pydantic models
3. Confirm descriptions → I write directive metadata
4. Confirm examples → I create demonstration cases
5. Confirm test cases → I design validation scenarios
6. Review implementation → I write complete directive code
7. Test and iterate → We verify functionality together

**Ready to create a new directive? Describe what transformation you want to implement!**
