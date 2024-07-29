from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class Agent:
    def __init__(self, llm: BaseLLM, valid_outputs: List[str], max_retries: int = 3):
        self.llm = llm
        self.valid_outputs = [vo.lower() for vo in valid_outputs]
        self.base_instruction = (
            "Explain your thought process. Then, return one of the following valid answers on a new line: "
            + ", ".join(valid_outputs)
            + "."
        )
        self.max_retries = max_retries

    def execute(self, prompt: str) -> Tuple[str, str]:
        for attempt in range(self.max_retries):
            # Take the prompt and add the base instruction
            full_prompt = prompt + "\n\n" + self.base_instruction
            raw_response = self.llm.generate(full_prompt)
            # Get the last non-empty line of the response
            response = next(
                (
                    line.strip()
                    for line in reversed(raw_response.split("\n"))
                    if line.strip()
                ),
                "",
            )
            if response.lower() in self.valid_outputs:
                return response, raw_response
            if attempt < self.max_retries - 1:
                print(f"Attempt {attempt + 1} failed. Retrying...")

        raise ValueError(
            f"Invalid response after {self.max_retries} attempts. Last response: {response}. Expected one of: {', '.join(self.valid_outputs)}"
        )
