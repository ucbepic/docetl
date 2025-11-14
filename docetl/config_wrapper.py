import datetime
import os
import time
from typing import Any

import pyrate_limiter
from pyrate_limiter import BucketFullException, LimiterDelayException
from rich.console import Console

from docetl.console import get_console
from docetl.operations.utils import APIWrapper
from docetl.ratelimiter import create_bucket_factory
from docetl.utils import decrypt, load_config


class ConfigWrapper(object):

    @classmethod
    def from_yaml(cls, yaml_file: str, **kwargs):
        # check that file ends with .yaml or .yml
        if not yaml_file.endswith(".yaml") and not yaml_file.endswith(".yml"):
            raise ValueError(
                "Invalid file type. Please provide a YAML file ending with '.yaml' or '.yml'."
            )

        base_name = yaml_file.rsplit(".", 1)[0]
        suffix = yaml_file.split("/")[-1].split(".")[0]
        config = load_config(yaml_file)
        return cls(config, base_name=base_name, yaml_file_suffix=suffix, **kwargs)

    def __init__(
        self,
        config: dict,
        base_name: str | None = None,
        yaml_file_suffix: str | None = None,
        max_threads: int | None = None,
        console: Console | None = None,
        **kwargs,
    ):
        self.config = config
        self.base_name = base_name
        self.yaml_file_suffix = yaml_file_suffix or datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.default_model = self.config.get("default_model", "gpt-4o-mini")
        if console:
            self.console = console
        else:
            # Reset the DOCETL_CONSOLE
            global DOCETL_CONSOLE
            DOCETL_CONSOLE = get_console()

            self.console = DOCETL_CONSOLE
        self.max_threads = max_threads or (os.cpu_count() or 1) * 4
        self.status = None
        encrypted_llm_api_keys = self.config.get("llm_api_keys", {})
        if encrypted_llm_api_keys:
            self.llm_api_keys = {
                key: decrypt(value, os.environ.get("DOCETL_ENCRYPTION_KEY", ""))
                for key, value in encrypted_llm_api_keys.items()
            }
        else:
            self.llm_api_keys = {}

        # Temporarily set environment variables for API keys
        self._original_env = os.environ.copy()
        for key, value in self.llm_api_keys.items():
            os.environ[key] = value

        bucket_factory = create_bucket_factory(self.config.get("rate_limits", {}))
        self.rate_limiter = pyrate_limiter.Limiter(bucket_factory, max_delay=200)
        self.is_cancelled = False

        # Store fallback configs
        self.fallback_models_config = self.config.get("fallback_models", [])
        self.fallback_embedding_models_config = self.config.get("fallback_embedding_models", [])
        # Create base routers as instance variables (for fallback models only)
        self.router = self._create_router(self.fallback_models_config, "completion")
        self.embedding_router = self._create_router(self.fallback_embedding_models_config, "embedding")
        # Cache routers per operation model (operation model + fallbacks)
        self._router_cache: dict[str, Any] = {}

        self.api = APIWrapper(self)

    def _create_router(self, fallback_models: list, router_type: str) -> Any | None:
        """
        Create a LiteLLM Router with fallback models if configured.

        Args:
            fallback_models: List of fallback model configurations
            router_type: Type of router ("completion" or "embedding") for logging

        Returns:
            Router instance if fallback_models are configured, None otherwise.
        """
        if not fallback_models:
            return None

        try:
            from litellm import Router
        except ImportError:
            self.console.log(
                f"[yellow]Warning: LiteLLM Router not available. Fallback {router_type} models will be ignored.[/yellow]"
            )
            return None

        # Build model list and fallbacks for Router
        model_list = []
        fallback_model_names = []
        
        for fallback_config in fallback_models:
            if isinstance(fallback_config, dict):
                model_name = fallback_config.get("model_name")
                litellm_params = fallback_config.get("litellm_params", {})
            elif isinstance(fallback_config, str):
                model_name = fallback_config
                litellm_params = {}
            else:
                self.console.log(
                    f"[yellow]Warning: Invalid fallback_{router_type}_models entry: {fallback_config}. Skipping.[/yellow]"
                )
                continue

            if not model_name:
                self.console.log(
                    f"[yellow]Warning: fallback_{router_type}_models entry missing model_name: {fallback_config}. Skipping.[/yellow]"
                )
                continue

            # Ensure model is included in litellm_params (required by LiteLLM Router)
            litellm_params_with_model = litellm_params.copy()
            litellm_params_with_model["model"] = model_name

            model_list.append(
                {
                    "model_name": model_name,
                    "litellm_params": litellm_params_with_model,
                }
            )
            fallback_model_names.append(model_name)

        if not model_list:
            return None

        try:
            # Create Router with model_list and fallbacks parameter
            # fallbacks should be a list of dicts: [{"model1": ["fallback1", "fallback2"]}]
            router_kwargs = {"model_list": model_list}
            
            # Build fallbacks list: each model falls back to the remaining models in order
            if len(fallback_model_names) > 1:
                fallbacks = []
                for i, model_name in enumerate(fallback_model_names):
                    # Each model falls back to the models after it in the list
                    if i < len(fallback_model_names) - 1:
                        fallbacks.append({model_name: fallback_model_names[i + 1:]})
                router_kwargs["fallbacks"] = fallbacks
            
            router = Router(**router_kwargs)
            self.console.log(
                f"[green]Created LiteLLM {router_type} Router with {len(model_list)} fallback model(s) in order: {', '.join(fallback_model_names)}[/green]"
            )
            return router
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Failed to create LiteLLM {router_type} Router: {e}. Fallback models will be ignored.[/yellow]"
            )
            return None

    def reset_env(self):
        os.environ = self._original_env

    def blocking_acquire(self, key: str, weight: int, wait_time=0.5):
        while True:
            try:
                self.rate_limiter.try_acquire(key, weight=weight)
                return  # Acquired successfully
            except LimiterDelayException as e:
                print(e.meta_info)
                time_to_wait = e.meta_info["actual_delay"] / 1000
                self.console.log(
                    f"Rate limits met for {key}; sleeping for {max(time_to_wait, wait_time):.2f} seconds"
                )
                time.sleep(max(time_to_wait, wait_time))
            except BucketFullException as e:
                time_to_wait = e.meta_info["remaining_time"]
                self.console.log(
                    f"Rate limits met for {key}; sleeping for {max(time_to_wait, wait_time):.2f} seconds"
                )
                time.sleep(max(time_to_wait, wait_time))
