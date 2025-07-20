import math
import time
from typing import Any

import pyrate_limiter
from litellm import RateLimitError, completion

from docetl.operations.utils import truncate_messages
from docetl.ratelimiter import create_bucket_factory
from docetl.utils import completion_cost


class LLMClient:
    """
    A client for interacting with LLMs, mainly used for the agent.

    This class provides methods to generate responses using specified LLM models
    and keeps track of the total cost of API calls.
    """

    def __init__(
        self,
        runner,
        rewrite_agent_model: str,
        judge_agent_model: str,
        rate_limits: dict[str, dict[str, Any]],
        **litellm_kwargs,
    ):
        """
        Initialize the LLMClient.

        Args:
            model (str, optional): The name of the LLM model to use. Defaults to "gpt-4o".
            **litellm_kwargs: Additional keyword arguments for the LLM model.
        """
        self.rewrite_agent_model = rewrite_agent_model
        self.judge_agent_model = judge_agent_model
        self.litellm_kwargs = litellm_kwargs
        if "temperature" not in self.litellm_kwargs:
            self.litellm_kwargs["temperature"] = 0.0

        self.total_cost = 0
        self.runner = runner

        # Initialize the rate limiter for judge model
        bucket_factory = create_bucket_factory(rate_limits)
        self.rate_limiter = pyrate_limiter.Limiter(bucket_factory, max_delay=math.inf)

    def _generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        parameters: dict[str, Any],
        model: str,
    ) -> Any:
        """
        Generate a response using the LLM.

        This method sends a request to the LLM with the given messages, system prompt,
        and parameters, and returns the response.

        Args:
            messages (list[dict[str, str]]): A list of message dictionaries to send to the LLM.
            system_prompt (str): The system prompt to use for the generation.
            parameters (dict[str, Any]): Additional parameters for the LLM request.

        Returns:
            Any: The response from the LLM.
        """
        parameters["additionalProperties"] = False

        messages = truncate_messages(messages, model, from_agent=True)

        if model == self.judge_agent_model:
            # Acquire
            self.rate_limiter.try_acquire("llm_call", weight=1)

        rate_limited_attempt = 0
        while rate_limited_attempt < 6:
            try:
                response = completion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        *messages,
                    ],
                    **self.litellm_kwargs,
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
            except RateLimitError:
                backoff_time = 4 * (2**rate_limited_attempt)  # Exponential backoff
                max_backoff = 120  # Maximum backoff time of 120 seconds
                sleep_time = min(backoff_time, max_backoff)
                self.runner.console.log(
                    f"[yellow]Rate limit hit. Retrying in {sleep_time:.2f} seconds...[/yellow]"
                )
                time.sleep(sleep_time)
                rate_limited_attempt += 1

        raise Exception("Rate limit hit too many times")

    def generate_rewrite(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        parameters: dict[str, Any],
    ) -> Any:
        return self._generate(
            messages, system_prompt, parameters, self.rewrite_agent_model
        )

    def generate_judge(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        parameters: dict[str, Any],
    ) -> Any:
        return self._generate(
            messages, system_prompt, parameters, self.judge_agent_model
        )
