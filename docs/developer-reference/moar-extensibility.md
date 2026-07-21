# Extending MOAR with rewrite directives

A rewrite directive teaches MOAR one abstract way to transform a pipeline. The
directive does not hardcode the rewrite for a particular pipeline. It describes
when a transformation is useful, defines the structured configuration that the
rewrite agent must generate, validates that configuration, and applies it to the
pipeline.

The built-in `chaining` directive is a compact example. It replaces one complex
operation with a sequence of simpler map operations:

```text
Op  =>  Map* -> Op
```

The rewrite agent decides what the intermediate tasks and prompts should be.
DocETL constructs the resulting pipeline, rejects invalid candidates, and MOAR
measures whether the candidate improves the cost and accuracy frontier.

This guide follows the implementation in
[`docetl/reasoning_optimizer/directives/chaining.py`](https://github.com/ucbepic/docetl/blob/main/docetl/reasoning_optimizer/directives/chaining.py).

## Directive lifecycle

MOAR uses a directive in five stages:

1. The search agent reads `name`, `formal_description`, `nl_description`, and
   `when_to_use`, then selects a directive and one or more target operations.
2. The directive asks the rewrite agent for a typed instantiate schema.
3. The schema and directive code validate the proposed configuration.
4. `apply()` returns a new operation list, and DocETL statically validates the
   complete candidate pipeline before executing it.
5. MOAR runs the candidate and evaluates its accuracy and cost.

Keep the boundary between stages explicit. The schema describes what the agent
may propose; `apply()` is deterministic pipeline construction.

## Example: decompose a complex extraction

Suppose a map operation must identify newly diagnosed conditions and associate
treatments with them in one call:

```yaml
- name: extract_new_treatments
  type: map
  prompt: |
    From this discharge summary, extract every treatment
    prescribed specifically for a newly diagnosed condition.

    {{ input.summary }}
  output:
    schema:
      treatments: list[str]
```

The operation combines two dependent decisions:

1. Which conditions are newly diagnosed?
2. Which treatments were prescribed for those conditions?

Chaining can ask the agent to produce a decomposition such as:

![Before and after the chaining rewrite](../assets/moar-chaining-rewrite.svg)

Both plans accept `summary` and produce `treatments`. The rewritten plan makes
`new_conditions` an explicit intermediate result that the second map can use.

The developer implements the general strategy. The rewrite agent supplies the
task-specific steps.

## 1. Define the instantiate schema

Add the Pydantic models that describe the agent's output to
`docetl/reasoning_optimizer/instantiate_schemas.py`:

```python
class MapOpConfig(BaseModel):
    name: str
    prompt: str
    output_keys: list[str]


class ChainingInstantiateSchema(BaseModel):
    new_ops: list[MapOpConfig]
```

For the medical operation, a valid agent response can be represented as:

```python
ChainingInstantiateSchema(
    new_ops=[
        MapOpConfig(
            name="identify_new_conditions",
            prompt="""
            Read this discharge summary:

            {{ input.summary }}

            List only conditions explicitly described as newly diagnosed.
            """,
            output_keys=["new_conditions"],
        ),
        MapOpConfig(
            name="extract_treatments",
            prompt="""
            Read this discharge summary:

            {{ input.summary }}

            Newly diagnosed conditions:
            {{ input.new_conditions }}

            List the treatments prescribed for each new condition.
            """,
            output_keys=["treatments"],
        ),
    ]
)
```

Use narrow fields with descriptions that a model can follow. Do not ask the
agent to emit an entire pipeline when the directive needs only two prompts and
their output keys.

## 2. Validate the agent's proposal

Validation should enforce the semantic invariants of the rewrite before any
candidate is executed. Chaining currently checks three conditions:

- Every generated map prompt contains an `{{ input.<key> }}` reference.
- Every input key used by the original operation appears in at least one new
  prompt.
- The last map declares exactly the original operation's output keys.

For the example, `summary` is a required original input and `treatments` is the
required final output. A chain ending at `new_conditions` is therefore invalid.

`{{ input.new_conditions }}` is valid in a later map because the earlier map
produces `new_conditions`. The complete candidate is also passed through
DocETL's static plan validation after `apply()`. If a new directive has stricter
dependency rules, validate them in its instantiate schema rather than relying
only on execution to expose an error.

The agent can generate malformed or incomplete configurations even with
structured output enabled. Return the validation error to the agent and retry a
small, bounded number of times. `MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS` is the
shared limit used by the built-in directives.

## 3. Describe the abstract strategy

Create a subclass of `Directive` under
`docetl/reasoning_optimizer/directives/`. The four descriptive fields are part
of the search interface, not only documentation:

```python
class ChainingDirective(Directive):
    name: str = Field(default="chaining")
    formal_description: str = Field(default="Op => Map* -> Op")
    nl_description: str = Field(
        default=(
            "Decompose a complex operation into a sequence by inserting one "
            "or more Map steps that rewrite the input for the next operation."
        )
    )
    when_to_use: str = Field(
        default=(
            "When the original task is too complex for one step and should be "
            "split into a series of dependent steps."
        )
    )
    instantiate_schema_type: Type[BaseModel] = ChainingInstantiateSchema
```

Write `when_to_use` so the search agent can decide among competing directives.
State the observable property of a suitable target, such as dependent reasoning
steps, rather than claiming that the directive always improves accuracy.

The base class also requires an `example` and may include `test_cases`. The
example becomes part of the rewrite agent's context, so use a complete example
that preserves inputs, outputs, and prompt instructions.

## 4. Instantiate the directive

Implement `to_string_for_instantiate()` and `llm_instantiate()` to obtain the
typed configuration:

```python
response = completion(
    model=agent_llm,
    messages=messages,
    response_format=ChainingInstantiateSchema,
)

payload = json.loads(response.choices[0].message.content)
rewrite = ChainingInstantiateSchema(**payload)
ChainingInstantiateSchema.validate_chain(
    new_ops=rewrite.new_ops,
    required_input_keys=expected_input_keys,
    expected_output_keys=expected_output_keys,
)
```

`llm_instantiate()` should return the schema, updated message history, and the
LLM call cost. MOAR includes the call cost in the total search cost.

Implement `instantiate()` as the orchestration boundary:

1. Check the number and types of target operations.
2. Read the target's required input and output keys.
3. Call `llm_instantiate()`.
4. Call the deterministic `apply()` method.
5. Return `(new_ops, message_history, call_cost)`.

MOAR passes additional keyword arguments to every directive, including the
optimization goal, dataset, allowed models, and input path. Accept `**kwargs`
when a directive does not use all of them.

## 5. Apply the rewrite

`apply()` should not call a model. It should copy the operation list, construct
the replacement operations from the validated schema, and return the new list.
The essential chaining implementation is:

```python
def apply(self, global_default_model, ops_list, target_op, rewrite):
    new_ops_list = deepcopy(ops_list)
    position = next(
        i for i, op in enumerate(new_ops_list) if op["name"] == target_op
    )
    original = new_ops_list[position]
    model = original.get("model", global_default_model)

    replacement = []
    for index, step in enumerate(rewrite.new_ops):
        output = (
            original["output"]
            if index == len(rewrite.new_ops) - 1
            else {
                "schema": {key: "string" for key in step.output_keys}
            }
        )
        replacement.append(
            {
                "name": step.name,
                "type": "map",
                "prompt": step.prompt,
                "model": model,
                "litellm_completion_kwargs": {"temperature": 0},
                "output": output,
            }
        )

    new_ops_list[position : position + 1] = replacement
    return new_ops_list
```

The last map reuses the original output block so the rewritten pipeline
preserves the public output contract, including its field types.

## 6. Register the directive

Export and instantiate the directive in
`docetl/reasoning_optimizer/directives/__init__.py`:

```python
from .my_directive import MyDirective

ALL_DIRECTIVES = [
    # ...
    MyDirective(),
]
```

Adding an instance to `ALL_DIRECTIVES` makes the directive available to the
accuracy search and adds it to `DIRECTIVE_REGISTRY`. Also update `__all__`.

Some directives require additional registration:

| Registry | Add the directive when |
| --- | --- |
| `ALL_COST_DIRECTIVES` | Its description should be offered during cost optimization. |
| `MULTI_INSTANCE_DIRECTIVES` | MOAR should ask for two distinct instantiations per selection. |
| `DIRECTIVE_GROUPS` | A poor result should suppress a family of equivalent rewrites for the same operation. |

Only add a directive to the cost list when its expected effect on cost is
clear. A rewrite that merely adds model calls is normally an accuracy directive.

## Test the directive

Use separate test layers so a model-dependent test is not mistaken for a
deterministic correctness test.

### Schema tests

Test valid and invalid configurations directly. Include missing original input
keys, missing final outputs, empty chains, duplicate operation names, and
references to unavailable intermediate keys when the directive forbids them.

```python
def test_chaining_rejects_wrong_final_output():
    rewrite = ChainingInstantiateSchema(
        new_ops=[
            MapOpConfig(
                name="identify_new_conditions",
                prompt="Read {{ input.summary }}",
                output_keys=["new_conditions"],
            )
        ]
    )

    with pytest.raises(ValueError, match="do not match"):
        ChainingInstantiateSchema.validate_chain(
            rewrite.new_ops,
            required_input_keys=["summary"],
            expected_output_keys=["treatments"],
        )
```

### Deterministic apply tests

Construct the schema by hand, call `apply()`, and assert the full operation
shape. Verify operation order, names, models, intermediate schemas, and exact
preservation of the final output schema. Add the test to
`tests/reasoning_optimizer/test_directive_apply.py`.

```bash
uv run pytest tests/reasoning_optimizer/test_directive_apply.py -k chaining
```

### Pipeline validation and multi-step tests

A locally plausible operation list can still break step references or later
rewrites. Add a test to `tests/test_moar_multistep.py` that applies the rewrite,
updates the pipeline, runs static plan validation, and, where relevant, applies
a second directive to an inserted operation.

```bash
uv run pytest tests/test_moar_multistep.py -k chaining
```

### Live agent tests

`Directive.run_tests()` calls a live model to instantiate the directive and a
second live model call to judge the result. Use it as an integration check, not
as the only correctness test:

```bash
uv run python -m tests.reasoning_optimizer.test_runner \
  --directive chaining \
  --model gpt-4.1
```

The command requires credentials for the selected model and may vary between
runs. Record the model, prompt version, fixture, and repeated-run pass rate when
comparing directive changes.

## Confirm that MOAR uses the directive

Registration, selection, successful construction, and measured improvement are
different claims. Check each one explicitly:

| Claim | Check |
| --- | --- |
| Registered | Print the names in `ALL_DIRECTIVES` or `DIRECTIVE_REGISTRY`. |
| Offered to search | Find the directive in the action statistics in `moar_tree_log.txt`. |
| Selected | Look for `Directive: <name>, Target ops: [...]` in the console, or a nonzero use count in the tree log. |
| Constructed | Open the generated `<pipeline-name>_<node-id>.yaml` and inspect the rewritten operations. |
| Helpful | Compare the candidate's evaluation metric and cost with its parent; check whether it reaches the Pareto frontier. |

Confirm registration without making an LLM call:

```bash
uv run python -c \
  "from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES; print([d.name for d in ALL_DIRECTIVES])"
```

Then run a small MOAR experiment with an explicit `save_dir`. The search log is
written to `<save_dir>/moar_tree_log.txt`:

```bash
rg -n "my_directive|Action: my_directive" results/moar_tree_log.txt
```

A zero use count means the directive was available but was not selected. A
nonzero count means it was selected, but not necessarily that candidate
construction or execution succeeded. Inspect the console output and generated
YAML as well.

For a deterministic MOAR integration test, construct `MOARSearch` with
`available_actions={MyDirective()}` and a fixed evaluation fixture. Restricting
the action set prevents the search agent from choosing a different directive.
Keep an unrestricted benchmark too, because a good directive description must
compete successfully with the rest of the registry.

## Implementation checklist

| Part | Chaining example |
| --- | --- |
| Pattern | One complex operation |
| Replacement | A sequence of map operations |
| Selection metadata | Use for dependent reasoning steps |
| Instantiate schema | A list of map names, prompts, and output keys |
| Validation | Preserve original inputs and final output contract |
| Apply | Replace one operation with the generated sequence |
| Deterministic tests | Schema, operation shape, and static plan validity |
| Empirical tests | Selection rate, execution success, accuracy, and cost |

The central interface is small: a developer specifies an abstract strategy and
its invariants, an agent adapts the strategy to a particular pipeline, and MOAR
tests the resulting candidate against the user's evaluation function.
