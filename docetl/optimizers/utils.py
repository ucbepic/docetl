from typing import Any, Dict, List

from litellm import completion

from docetl.operations.utils import truncate_messages
from docetl.utils import completion_cost


class LLMClient:
    """
    A client for interacting with LLMs, mainly used for the agent.

    This class provides methods to generate responses using specified LLM models
    and keeps track of the total cost of API calls.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the LLMClient.

        Args:
            model (str, optional): The name of the LLM model to use. Defaults to "gpt-4o".
        """
        if model == "gpt-4o":
            model = "gpt-4o-2024-08-06"
        self.model = model
        self.total_cost = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        parameters: Dict[str, Any],
    ) -> Any:
        """
        Generate a response using the LLM.

        This method sends a request to the LLM with the given messages, system prompt,
        and parameters, and returns the response.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries to send to the LLM.
            system_prompt (str): The system prompt to use for the generation.
            parameters (Dict[str, Any]): Additional parameters for the LLM request.

        Returns:
            Any: The response from the LLM.
        """
        parameters["additionalProperties"] = False

        messages = truncate_messages(messages, self.model, from_agent=True)

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                *messages,
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": parameters,
                },
            },
        )
        cost = completion_cost(response)
        self.total_cost += cost
        return response
