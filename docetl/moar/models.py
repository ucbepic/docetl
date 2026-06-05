"""
Auto-detection of available LLM models based on environment API keys.
"""

import os
from typing import List, Optional


PROVIDER_MODELS = {
    "OPENAI_API_KEY": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "ANTHROPIC_API_KEY": [
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5-20251001",
    ],
    "GEMINI_API_KEY": [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
    ],
    "AZURE_API_KEY": [
        "azure/gpt-4.1",
        "azure/gpt-4.1-mini",
        "azure/gpt-4.1-nano",
        "azure/gpt-4o",
        "azure/gpt-4o-mini",
    ],
    "AZURE_OPENAI_API_KEY": [
        "azure/gpt-4.1",
        "azure/gpt-4.1-mini",
        "azure/gpt-4.1-nano",
        "azure/gpt-4o",
        "azure/gpt-4o-mini",
    ],
}

AGENT_MODEL_PREFERENCE = [
    ("OPENAI_API_KEY", "gpt-4.1"),
    ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-6"),
    ("GEMINI_API_KEY", "gemini/gemini-2.5-pro"),
    ("AZURE_API_KEY", "azure/gpt-4.1"),
    ("AZURE_OPENAI_API_KEY", "azure/gpt-4.1"),
]


def detect_available_models() -> List[str]:
    """
    Return models for all providers whose API key is set in the environment.

    Checks OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, AZURE_API_KEY,
    and AZURE_OPENAI_API_KEY. De-duplicates models (Azure keys can overlap).

    Raises:
        ValueError: If no API keys are found in the environment.
    """
    seen = set()
    models = []
    for env_var, model_list in PROVIDER_MODELS.items():
        if os.environ.get(env_var):
            for m in model_list:
                if m not in seen:
                    seen.add(m)
                    models.append(m)
    if not models:
        raise ValueError(
            "No LLM API keys found in environment. "
            "Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "GEMINI_API_KEY, AZURE_API_KEY, or AZURE_OPENAI_API_KEY."
        )
    return models


def default_agent_model(models: Optional[List[str]] = None) -> str:
    """
    Pick the best available model for the MOAR rewrite agent.

    If *models* is provided, returns the first model from the preference list
    that appears in *models*. Otherwise checks environment variables directly.

    Raises:
        ValueError: If no suitable model can be determined.
    """
    if models:
        for _, model in AGENT_MODEL_PREFERENCE:
            if model in models:
                return model
        return models[0]

    for env_var, model in AGENT_MODEL_PREFERENCE:
        if os.environ.get(env_var):
            return model

    raise ValueError(
        "Cannot determine a rewrite agent model. "
        "Set at least one LLM API key in the environment."
    )
