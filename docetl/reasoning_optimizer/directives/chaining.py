import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Type

from litellm import completion
from pydantic import BaseModel, Field

from docetl.reasoning_optimizer.instantiate_schemas import ChainingInstantiateSchema

from .base import MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS, Directive, DirectiveTestCase


class ChainingDirective(Directive):
    name: str = Field(default="chaining", description="The name of the directive")
    formal_description: str = Field(default="Op => Map* -> Op")
    nl_description: str = Field(
        default="Decompose a complex operation into a sequence by inserting one or more Map steps that rewrite the input for the next operation. Each Map step outputs a 'result' string, and the downstream operation uses this result in its prompt."
    )
    when_to_use: str = Field(
        default="When the original task is too complex for one step and should be split into a series (e.g., first extract key facts, then generate a summary based on those facts)."
    )
    instantiate_schema_type: Type[BaseModel] = ChainingInstantiateSchema

    example: str = Field(
        default=(
            "Original Op (MapOpConfig):\n"
            "- name: extract_newly_prescribed_treatments\n"
            "  type: map\n"
            "  prompt: |\n"
            "    For a hospital discharge summary, extract every treatment that was prescribed specifically for newly diagnosed conditions.\n"
            '    Discharge summary: "{{ input.summary }}"\n'
            "  output:\n"
            "    schema:\n"
            "      treatments: list[str]\n"
            "\n"
            "Example InstantiateSchema (must refer to the same input document keys as the original Op, and subsequent Map operators must refer to the output of the previous Map operator):\n"
            "[\n"
            "  MapOpConfig(\n"
            "    name='identify_new_conditions',\n"
            "    prompt='''Review the following hospital discharge summary:\n"
            "{{ input.summary }}\n"
            "Identify all medical conditions that are explicitly marked as new diagnoses (e.g., 'new diagnosis of atrial fibrillation', 'recent onset heart failure').\n"
            "Return a list of newly diagnosed conditions.''',\n"
            "    output_keys=['new_conditions'],\n"
            "  ),\n"
            "  MapOpConfig(\n"
            "    name='extract_treatments_for_new_conditions',\n"
            "    prompt='''For each newly diagnosed condition listed below, extract every treatment or medication prescribed for that specific condition from the discharge summary.\n"
            "Discharge summary: {{ input.summary }}\n"
            "Newly diagnosed conditions: {{ input.new_conditions }}\n"
            "Return a list of prescribed treatments or medications for each condition.''',\n"
            "    output_keys=['treatments'],\n"
            "  ),\n"
            "]"
        ),
    )

    test_cases: List[DirectiveTestCase] = Field(
        default_factory=lambda: [
            DirectiveTestCase(
                name="complex_contract_extraction",
                description="Should decompose complex contract extraction into separate ops for each term type and a final unification op",
                input_config={
                    "name": "extract_contract_terms",
                    "type": "map",
                    "prompt": "Extract all payment terms, liability clauses, and termination conditions from: {{ input.contract }}",
                    "output": {"schema": {"terms": "list[str]"}},
                },
                target_ops=["extract_contract_terms"],
                expected_behavior="Should create one op for each of: payment terms, liability clauses, and termination conditions, then a final op to unify all results into 'terms'",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="medical_treatment_analysis",
                description="Should chain complex medical analysis into steps",
                input_config={
                    "name": "analyze_patient_care",
                    "type": "map",
                    "prompt": "From this medical record, identify all diagnoses, treatments prescribed, and patient outcomes: {{ input.medical_record }}",
                    "output": {"schema": {"analysis": "string"}},
                },
                target_ops=["analyze_patient_care"],
                expected_behavior="Should decompose into separate steps for diagnoses identification, treatment extraction, and outcome analysis",
                should_pass=True,
            ),
            DirectiveTestCase(
                name="research_paper_analysis",
                description="Should chain research paper analysis into structured steps",
                input_config={
                    "name": "analyze_research_paper",
                    "type": "map",
                    "prompt": "From this research paper, extract the methodology, key findings, limitations, and future work directions: {{ input.paper }}",
                    "output": {"schema": {"analysis": "string"}},
                },
                target_ops=["analyze_research_paper"],
                expected_behavior="Should decompose into separate steps for methodology extraction, findings identification, limitation analysis, and future work extraction",
                should_pass=True,
            ),
        ]
    )

    def __eq__(self, other):
        return isinstance(other, ChainingDirective)

    def __hash__(self):
        return hash("ChainingDirective")

    def to_string_for_instantiate(self, original_op: Dict) -> str:
        """
        Generate a prompt for an agent to instantiate this directive.

        Args:
            original_op (str): The YAML or string representation of the original operation.

        Returns:
            str: The agent prompt for instantiating the directive.
        """
        return (
            f"You are an expert at decomposing complex data processing operations into modular steps.\n\n"
            f"Original Operation:\n"
            f"{str(original_op)}\n"
            f"Directive: {self.name}\n"
            f"Your task is to instantiate this directive by generating a list of new Map operators (as MapOpConfig objects) that decompose the original operation into a sequence of simpler steps. "
            f"Each Map step should output a 'result' or other relevant keys, and downstream steps should use the outputs of previous steps as their input. "
            f"Ensure that the chain of Map operators together accomplishes the intent of the original operation, but in a more modular and stepwise fashion.\n\n"
            f"""Key Issues to ensure:\n
                1. Ensure the new prompts don't reference categories, lists, or criteria without providing them or making them available from previous steps.\n
                2. Every detail, category, instruction, requirement, and definition from the original must be present in the new configuration.\n
                3. Confirm that each step can access all required information either from the original document or from outputs of preceding steps.\n"""
            f"Example:\n"
            f"{self.example}\n\n"
            f"Please output only the InstantiateSchema (a list of MapOpConfig objects) for the new chain, referring to the same input document keys as the original operation and chaining outputs appropriately."
        )

    def llm_instantiate(
        self,
        original_op: Dict,
        expected_input_keys: List[str],
        expected_output_keys: List[str],
        agent_llm: str,
        message_history: list = [],
    ):
        """
        Use LLM to instantiate this directive by decomposing the original operation.

        Args:
            original_op (Dict): The original operation.
            expected_input_keys (List[str]): A list of input keys that the operation is expected to reference in its prompt. Each key should correspond to a field in the input document that must be used by the operator.
            expected_output_keys (List[str]): A list of output keys that the last operation is expected to produce.
            agent_llm (str): The LLM model to use.
            message_history (List, optional): Conversation history for context.

        Returns:
            ChainingInstantiateSchema: The structured output from the LLM.
        """

        message_history.extend(
            [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for document processing pipelines.",
                },
                {
                    "role": "user",
                    "content": self.to_string_for_instantiate(original_op),
                },
            ]
        )

        for _ in range(MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS):

            resp = completion(
                model=agent_llm,
                messages=message_history,
                api_key=os.environ.get("AZURE_API_KEY"),
                api_base=os.environ.get("AZURE_API_BASE"),
                api_version=os.environ.get("AZURE_API_VERSION"),
                # api_key=os.environ["GEMINI_API_KEY"],
                azure=True,
                response_format=ChainingInstantiateSchema,
            )
            call_cost = resp._hidden_params["response_cost"]

            try:
                parsed_res = json.loads(resp.choices[0].message.content)
                if "new_ops" not in parsed_res:
                    raise ValueError(
                        "Response from LLM is missing required key 'new_ops'"
                    )
                new_ops = parsed_res["new_ops"]
                schema = ChainingInstantiateSchema(new_ops=new_ops)
                # Validate the chain with required input/output keys
                ChainingInstantiateSchema.validate_chain(
                    new_ops=schema.new_ops,
                    required_input_keys=expected_input_keys,
                    expected_output_keys=expected_output_keys,
                )
                message_history.append(
                    {"role": "assistant", "content": resp.choices[0].message.content}
                )
                return schema, message_history, call_cost
            except Exception as err:
                error_message = f"Validation error: {err}\nPlease try again."
                message_history.append({"role": "user", "content": error_message})

        raise Exception(
            f"Failed to instantiate directive after {MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS} attempts."
        )

    def apply(
        self,
        global_default_model,
        ops_list: List[Dict],
        target_op: str,
        rewrite: ChainingInstantiateSchema,
    ) -> List[Dict]:
        """
        Apply the directive to the pipeline config.
        """
        # Create a copy of the pipeline config
        new_ops_list = deepcopy(ops_list)

        # Find position of the target ops to replace

        for i, op in enumerate(ops_list):
            if op["name"] == target_op:
                pos_to_replace = i
                orig_op = op
                break

        # pos_to_replace = [i for i, op in enumerate(ops_list) if op["name"] == target_op][0]

        # Create the new ops from the rewrite
        new_ops = []

        defualt_model = global_default_model
        if "model" in orig_op:
            defualt_model = orig_op["model"]

        for i, op in enumerate(rewrite.new_ops):
            if i < len(rewrite.new_ops) - 1:
                new_ops.append(
                    {
                        "name": op.name,
                        "type": "map",
                        "prompt": op.prompt,
                        "model": defualt_model,
                        "litellm_completion_kwargs": {"temperature": 0},
                        "output": {"schema": {key: "string" for key in op.output_keys}},
                    }
                )
            else:
                # Last op in the chain
                new_ops.append(
                    {
                        "name": op.name,
                        "type": "map",
                        "prompt": op.prompt,
                        "model": defualt_model,
                        "litellm_completion_kwargs": {"temperature": 0},
                        "output": new_ops_list[pos_to_replace]["output"],
                    }
                )

        # Remove the target op and insert the new ops
        new_ops_list.pop(pos_to_replace)
        new_ops_list[pos_to_replace:pos_to_replace] = new_ops

        return new_ops_list

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        optimize_goal="acc",
        global_default_model: str = None,
        **kwargs,
    ):
        """
        Instantiate the directive for a list of operators.
        """
        # Assert that there is only one target op
        assert (
            len(target_ops) == 1
        ), "There must be exactly one target op to instantiate this chaining directive"
        target_op_config = [op for op in operators if op["name"] == target_ops[0]][0]

        # Get the expected input/output keys
        expected_output_keys = list(target_op_config["output"]["schema"].keys())

        # Extract expected input keys from the target op's prompt template
        prompt_template = target_op_config["prompt"]
        # Find all occurrences of {{ input.key }} in the prompt
        input_key_pattern = r"\{\{\s*input\.([^\}\s]+)\s*\}\}"
        expected_input_keys = list(set(re.findall(input_key_pattern, prompt_template)))

        print("input key: ", expected_input_keys)
        print("output key: ", expected_output_keys)

        # Instantiate the directive
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_op_config,
            expected_input_keys,
            expected_output_keys,
            agent_llm,
            message_history,
        )

        # Apply the rewrite to the operators
        new_ops_plan = self.apply(
            global_default_model, operators, target_ops[0], rewrite
        )
        return new_ops_plan, message_history, call_cost
