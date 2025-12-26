# Directives - How to Add a New Directive

This guide explains how to add a new directive to the reasoning optimizer. Directives are transformations that can be applied to pipeline operations to improve their effectiveness.

## What is a Directive?

A directive is a transformation rule that modifies pipeline operations. For example:
- **Chaining**: Breaks complex operations into sequential steps
- **Gleaning**: Adds validation loops to improve output quality
- **Change Model**: Switches the LLM model for better performance
- **Doc Summarization**: Adds preprocessing to summarize long documents

## Key Concepts

### What is an Instantiate Schema?

An **instantiate schema** is a Pydantic model that defines the structured output an agent (LLM) must produce to apply a directive. It acts as the "configuration blueprint" that tells the system exactly how to transform an operation.

For example, when applying the **gleaning** directive:
1. The agent looks at the original operation
2. The agent outputs a `GleaningInstantiateSchema` containing:
   ```python
   {
     "gleaning_config": {
       "validation_prompt": "Check that the output has at least 2 insights...",
       "num_rounds": 3,
       "model": "gpt-4o-mini"
     }
   }
   ```
3. The directive uses this schema to add gleaning configuration to the operation

The instantiate schema ensures the agent provides all required parameters in the correct format for the directive to work.

### Workflow Overview

1. **Agent Analysis**: Agent examines the original operation and determines how to apply the directive
2. **Schema Generation**: Agent outputs structured configuration using the instantiate schema format
3. **Directive Application**: The directive's `apply()` method uses this configuration to transform the pipeline
4. **Validation**: Schema validators ensure the configuration is valid before application

## Quick Start - Adding a New Directive

### 1. Define Your Instantiate Schema

First, add your schema classes to `docetl/reasoning_optimizer/instantiate_schemas.py`. The instantiate schema defines what the agent must output:

```python
class MyDirectiveConfig(BaseModel):
    """Configuration parameters for your directive."""
    param1: str = Field(..., description="Description of param1")
    param2: int = Field(default=3, description="Description of param2")
    model: str = Field(default="gpt-4o-mini", description="The LLM model to use")

class MyDirectiveInstantiateSchema(BaseModel):
    """
    Schema that the agent must output to instantiate this directive.
    This is what gets returned by the LLM when asked to apply the directive.
    """
    my_directive_config: MyDirectiveConfig = Field(
        ..., description="The configuration to apply to the target operation"
    )

    # Add validators if needed
    @field_validator("my_directive_config")
    @classmethod
    def validate_config(cls, v):
        # Add validation logic here
        return v
```

### 2. Create Your Directive Class

Create a new file in this directory (e.g., `my_directive.py`):

```python
import json
import os
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import MyDirectiveInstantiateSchema
from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase

class MyDirective(Directive):
    name: str = Field(default="my_directive", description="The name of the directive")
    formal_description: str = Field(default="Op => Modified_Op")
    nl_description: str = Field(
        default="Natural language description of what this directive does"
    )
    when_to_use: str = Field(
        default="When to apply this directive (specific use cases)"
    )

    # This tells the system what schema format the agent should output
    instantiate_schema_type: Type[BaseModel] = Field(default=MyDirectiveInstantiateSchema)

    example: str = Field(
        default="""
        Original Op (MapOpConfig):
        - name: example_op
          type: map
          prompt: |
            Example prompt: {{ input.document }}
          output:
            schema:
              result: "string"

        Example InstantiateSchema (what the agent should output):
        MyDirectiveConfig(
            param1="example_value",
            param2=5,
            model="gpt-4o-mini"
        )
        """
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="basic_functionality",
                description="Should apply directive transformation correctly",
                input_config={
                    "name": "test_op",
                    "type": "map",
                    "prompt": "Test prompt: {{ input.text }}",
                    "output": {"schema": {"result": "string"}},
                },
                target_ops=["test_op"],
                expected_behavior="Should modify the operation with directive configuration",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, MyDirective)

    def __hash__(self):
        return hash("MyDirective")
```

### 3. Implement Required Methods

```python
    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt that asks the agent to output the instantiate schema.
        This prompt explains to the LLM what configuration it needs to generate.
        """
        return (
            f"You are an expert at [specific domain expertise for your directive].\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a MyDirectiveConfig "
            f"that specifies [specific instructions for what the directive should do].\n\n"
            f"The agent must output the configuration in this exact format:\n"
            f"- param1: [explanation of how to set this]\n"
            f"- param2: [explanation of how to set this]\n"
            f"- model: [which model to use]\n\n"
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the InstantiateSchema (MyDirectiveConfig object) "
            f"that specifies how to apply this directive to the original operation."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        agent_llm: str,
        message_history: list = [],
    ) -> tuple:
        """
        Call the LLM to generate the instantiate schema.
        The LLM will output structured data matching MyDirectiveInstantiateSchema.
        """

        message_history.extend([
            {
                "role": "user",
                "content": self.to_string_for_instantiate(original_op),
            },
        ])

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):
            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                azure=True,
                response_format=MyDirectiveInstantiateSchema,  # Forces structured output
            )

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                if "my_directive_config" not in parsed_res:
                    raise ValueError("Response missing required key 'my_directive_config'")

                config = parsed_res["my_directive_config"]
                schema = MyDirectiveInstantiateSchema(my_directive_config=config)
                message_history.append({
                    "role": "assistant",
                    "content": resp.choices[0].message.content
                })
                return schema, message_history
            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts.")

    def apply(
        self, ops_list: List[Dict], target_op: str, rewrite: MyDirectiveInstantiateSchema
    ) -> List[Dict]:
        """
        Apply the directive using the instantiate schema configuration.
        The 'rewrite' parameter contains the agent's generated configuration.
        """
        new_ops_list = deepcopy(ops_list)

        # Find the target operation
        pos_to_replace = [i for i, op in enumerate(ops_list) if op["name"] == target_op][0]
        target_operator = new_ops_list[pos_to_replace]

        # Apply transformation using the agent's configuration
        target_operator["my_directive_param"] = rewrite.my_directive_config.param1
        target_operator["model"] = rewrite.my_directive_config.model

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        **kwargs,
    ) -> tuple:
        """
        Main method that orchestrates directive instantiation:
        1. Get agent to generate instantiate schema
        2. Apply the transformation using that schema
        """
        assert len(target_ops) == 1, "This directive requires exactly one target op"

        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Step 1: Agent generates the instantiate schema
        rewrite, message_history = self.llm_instantiate(
            target_op_config, agent_llm, message_history
        )

        # Step 2: Apply transformation using the schema
        return self.apply(operators, target_ops[0], rewrite), message_history
```

### 4. Register Your Directive

Add your directive to `__init__.py`:

```python
from .my_directive import MyDirective

ALL_DIRECTIVES = [
    ChainingDirective(),
    GleaningDirective(),
    ChangeModelDirective(),
    DocSummarizationDirective(),
    MyDirective(),  # Add your directive here
]
```

### 5. Update Test Runner

Add your directive to `tests/reasoning_optimizer/test_runner.py` in two places:

1. **Import section**:
```python
from docetl.reasoning_optimizer.directives import (
    ChainingDirective,
    GleaningDirective,
    ChangeModelDirective,
    DocSummarizationDirective,
    IsolatingSubtasksDirective,
    MyDirective,  # Add your directive here
    TestResult
)
```

2. **Both directive lists**:
```python
# In run_all_directive_tests function
directives = [
    ChainingDirective(),
    GleaningDirective(),
    ChangeModelDirective(),
    DocSummarizationDirective(),
    IsolatingSubtasksDirective(),
    MyDirective()  # Add your directive here
]

# In run_specific_directive_test function
directive_map = {
    "chaining": ChainingDirective(),
    "gleaning": GleaningDirective(),
    "change_model": ChangeModelDirective(),
    "doc_summarization": DocSummarizationDirective(),
    "isolating_subtasks": IsolatingSubtasksDirective(),
    "my_directive": MyDirective()  # Add your directive here
}
```

## Real Examples of Instantiate Schemas

### Gleaning Directive
```python
# What the agent outputs:
{
  "gleaning_config": {
    "validation_prompt": "Check that the output contains at least 2 insights and each has supporting actions",
    "num_rounds": 3,
    "model": "gpt-4o-mini"
  }
}

# How it's applied: Adds gleaning configuration to the operation
target_operator["gleaning"] = {
    "validation_prompt": rewrite.gleaning_config.validation_prompt,
    "num_rounds": rewrite.gleaning_config.num_rounds,
    "model": rewrite.gleaning_config.model,
}
```

### Chaining Directive
```python
# What the agent outputs:
{
  "new_ops": [
    {
      "name": "extract_conditions",
      "prompt": "Identify new medical conditions from: {{ input.summary }}",
      "output_keys": ["conditions"],
      "model": "gpt-4o-mini"
    },
    {
      "name": "extract_treatments",
      "prompt": "Extract treatments for conditions {{ input.conditions }} from {{ input.summary }}",
      "output_keys": ["treatments"],
      "model": "gpt-4o-mini"
    }
  ]
}

# How it's applied: Replaces one operation with multiple chained operations
```

### Change Model Directive
```python
# What the agent outputs:
{
  "change_model_config": {
    "model": "gpt-4o"
  }
}

# How it's applied: Changes the model field
target_operator["model"] = rewrite.change_model_config.model
```

## Testing Your Directive

### Individual Directive Testing

Test your directive by running its test cases:

```python
from docetl.reasoning_optimizer.directives import MyDirective

directive = MyDirective()
test_results = directive.run_tests(agent_llm="gpt-4o-mini")

for result in test_results:
    print(f"{result.test_name}: {'PASS' if result.passed else 'FAIL'}")
    print(f"Reason: {result.reason}")
```

### Command Line Testing

Run directive instantiation tests from the command line:

```bash
# Test a specific directive
python experiments/reasoning/run_tests.py --directive=isolating_subtasks

# Test all directive instantiation tests
python experiments/reasoning/run_tests.py
```

### Apply Method Testing

Test that directive `apply()` methods work correctly:

```bash
# Test all directive apply methods
python tests/reasoning_optimizer/test_directive_apply.py
```

This ensures the `apply()` method doesn't crash when given realistic pipeline configurations and rewrite schemas.

### Integration Testing

Full pipeline integration testing can be done via `experiments/reasoning/run_mcts.py`.

## Common Patterns

### 1. Single Operation Modification
Adds configuration to existing operation:
- **Gleaning**: Adds validation config
- **Change Model**: Modifies model parameter

### 2. Operation Replacement
Replaces one operation with multiple:
- **Chaining**: Creates sequence of simpler operations

### 3. Pipeline Preprocessing
Adds operations at pipeline start:
- **Doc Summarization**: Adds summarization step before main processing

## File Structure Summary

```
docetl/reasoning_optimizer/
├── directives/
│   ├── my_directive.py          # Your directive implementation
│   ├── __init__.py              # Register in ALL_DIRECTIVES
│   └── README.md               # This file
└── instantiate_schemas.py       # Define your schema classes

experiments/reasoning/           # Testing framework
└── run_mcts.py                 # Full pipeline testing
```

## Best Practices

1. **Schema First**: Design the instantiate schema before implementing the directive - it defines the interface
2. **Clear Agent Instructions**: The `to_string_for_instantiate()` method should clearly explain what the agent needs to output
3. **Validation**: Use Pydantic validators in your schema to catch invalid configurations
4. **Error Handling**: Handle LLM failures gracefully with retry logic
5. **Comprehensive Testing**: Test edge cases where the agent might output invalid configurations
6. **Documentation**: Clearly document what your instantiate schema fields mean and how they're used

The instantiate schema is the critical bridge between the agent's reasoning and your directive's implementation - design it carefully!
