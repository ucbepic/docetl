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

        # Store fallback_models config for Router usage
        self.fallback_models_config = self.config.get("fallback_models", [])
        # Create LiteLLM Router if fallback_models are configured
        # Note: Router will be used with operation's model prepended at call time
        self.router = self._create_router()

        self.api = APIWrapper(self)

    def _create_router(self) -> Any | None:
        """
        Create a LiteLLM Router with fallback models if configured.

        The Router will automatically handle fallbacks when API errors or content errors occur.
        Note: The operation's model will be prepended to this list at call time to ensure it's tried first.

        Returns:
            Router instance if fallback_models are configured, None otherwise.
        """
        fallback_models = self.config.get("fallback_models", [])
        if not fallback_models:
            return None

        try:
            from litellm import Router
        except ImportError:
            self.console.log(
                "[yellow]Warning: LiteLLM Router not available. Fallback models will be ignored.[/yellow]"
            )
            return None

        # Build model list for Router
        model_list = []
        for fallback_config in fallback_models:
            if isinstance(fallback_config, dict):
                model_name = fallback_config.get("model_name")
                litellm_params = fallback_config.get("litellm_params", {})
            elif isinstance(fallback_config, str):
                # Simple string format: just model name
                model_name = fallback_config
                litellm_params = {}
            else:
                self.console.log(
                    f"[yellow]Warning: Invalid fallback_models entry: {fallback_config}. Skipping.[/yellow]"
                )
                continue

            if not model_name:
                self.console.log(
                    f"[yellow]Warning: fallback_models entry missing model_name: {fallback_config}. Skipping.[/yellow]"
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

        if not model_list:
            return None

        try:
            # Create Router with fallback models
            # The Router will automatically try models in order when errors occur
            router = Router(model_list=model_list)
            self.console.log(
                f"[green]Created LiteLLM Router with {len(model_list)} fallback model(s) in order: {', '.join([m['model_name'] for m in model_list])}[/green]"
            )
            return router
        except Exception as e:
            self.console.log(
                f"[yellow]Warning: Failed to create LiteLLM Router: {e}. Fallback models will be ignored.[/yellow]"
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
