import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from litellm import completion
from pydantic import BaseModel, Field

# Configuration constants
MAX_DIRECTIVE_INSTANTIATION_ATTEMPTS = 3
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_MAX_TPM = 5000000
DEFAULT_OUTPUT_DIR = "./outputs"


class TestResult(BaseModel):
    test_name: str
    passed: bool
    reason: str
    actual_output: Any
    execution_error: Optional[str] = None


class DirectiveTestCase(BaseModel):
    name: str
    description: str
    input_config: Dict[str, Any] | List[Dict[str, Any]]
    target_ops: List[str]
    expected_behavior: str
    should_pass: bool = True


class Directive(BaseModel, ABC):
    name: str = Field(..., description="The name of the directive")
    formal_description: str = Field(
        ..., description="A description of the directive; e.g., map => map -> map"
    )
    nl_description: str = Field(
        ..., description="An english description of the directive"
    )
    when_to_use: str = Field(..., description="When to use the directive")
    instantiate_schema_type: BaseModel = Field(
        ...,
        description="The schema the agent must conform to when instantiating the directive",
    )
    example: str = Field(..., description="An example of the directive being used")
    test_cases: List[DirectiveTestCase] = Field(
        default_factory=list, description="Test cases for this directive"
    )

    def to_string_for_plan(self) -> str:
        """Serialize directive for prompts."""
        parts = [
            f"### {self.name}",
            f"**Format:** {self.formal_description}",
            f"**Description:** {self.nl_description}",
            f"**When to Use:** {self.when_to_use}",
        ]
        return "\n\n".join(parts)

    @abstractmethod
    def to_string_for_instantiate(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def llm_instantiate(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs) -> list:
        pass

    @abstractmethod
    def instantiate(self, *args, **kwargs):
        pass

    def run_tests(self, agent_llm: str = "gpt-4o-mini") -> List[TestResult]:
        """Run all test cases for this directive using LLM judge"""
        import json
        import os
        import tempfile

        results = []

        for test_case in self.test_cases:
            try:
                # Create fake sample data and pipeline for directives that need them
                sample_data = [
                    {
                        "text": "Sample document 1",
                        "feedback": "Great product!",
                        "doc1": "Document A",
                        "doc2": "Document B",
                    },
                    {
                        "text": "Sample document 2",
                        "feedback": "Could be better",
                        "doc1": "Report 1",
                        "doc2": "Report 2",
                    },
                    {
                        "text": "Sample document 3",
                        "feedback": "Excellent service",
                        "doc1": "Policy A",
                        "doc2": "Policy B",
                    },
                ]

                fake_pipeline = {
                    "operations": (
                        [test_case.input_config]
                        if isinstance(test_case.input_config, dict)
                        else test_case.input_config
                    ),
                    "name": "test_pipeline",
                }

                # Create temporary input file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(sample_data, f, indent=2)
                    temp_file_path = f.name

                try:
                    # 1. Execute the directive
                    actual_output, _ = self.instantiate(
                        operators=(
                            [test_case.input_config]
                            if isinstance(test_case.input_config, dict)
                            else test_case.input_config
                        ),
                        target_ops=test_case.target_ops,
                        agent_llm=agent_llm,
                        input_file_path=temp_file_path,
                        pipeline_code=fake_pipeline,
                    )

                    # 2. Use LLM judge to evaluate
                    judge_result = self._llm_judge_test(
                        test_case=test_case,
                        actual_output=actual_output,
                        agent_llm=agent_llm,
                    )

                    results.append(
                        TestResult(
                            test_name=test_case.name,
                            passed=judge_result["passed"],
                            reason=judge_result["reason"],
                            actual_output=actual_output,
                        )
                    )
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass

            except Exception as e:
                # Handle execution errors
                expected_to_fail = not test_case.should_pass
                results.append(
                    TestResult(
                        test_name=test_case.name,
                        passed=expected_to_fail,
                        reason=f"Execution {'failed as expected' if expected_to_fail else 'failed unexpectedly'}: {str(e)}",
                        actual_output=None,
                        execution_error=str(e),
                    )
                )

        return results

    def _llm_judge_test(
        self, test_case: DirectiveTestCase, actual_output: Any, agent_llm: str
    ) -> Dict[str, Any]:
        """Use LLM as judge to evaluate test results"""

        user_message = f"""
        Evaluate whether the directive execution meets the expected behavior.

        **Test Case:** {test_case.name}
        **Description:** {test_case.description}
        **Expected Behavior:** {test_case.expected_behavior}
        **Should Pass:** {test_case.should_pass}

        **Original Input Configuration:**
        {test_case.input_config}

        **Actual Output from Directive:**
        {actual_output}

        **Evaluation Criteria:**
        - If should_pass=True: Does the output demonstrate the expected behavior?
        - If should_pass=False: Did the directive appropriately reject/not modify the input?
        - Does the transformation make logical sense?
        - Are all required elements preserved?

        Provide your reasoning and a boolean pass/fail decision.
        """

        messages = [
            {
                "role": "system",
                "content": f"You are an expert judge evaluating whether a {self.name} directive execution meets the specified criteria. Focus on logical correctness and adherence to expected behavior.",
            },
            {"role": "user", "content": user_message},
        ]

        class JudgeResponse(BaseModel):
            passed: bool
            reason: str

        response = completion(
            model=agent_llm,
            messages=messages,
            response_format=JudgeResponse,
            azure=True,
        )

        # Parse the JSON response

        parsed_content = json.loads(response.choices[0].message.content)

        return {"passed": parsed_content["passed"], "reason": parsed_content["reason"]}
