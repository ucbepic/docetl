from motion.agent.base import BaseLLM
from openai import OpenAI
from typing import Optional


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model

    def generate(self, prompt: str) -> str:
        return (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for processing data.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            .choices[0]
            .message.content
        )
