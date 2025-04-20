import datetime
import os
import time
from typing import Dict, Optional

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
        config: Dict,
        base_name: Optional[str] = None,
        yaml_file_suffix: Optional[str] = None,
        max_threads: int = None,
        console: Optional[Console] = None,
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

        self.api = APIWrapper(self)

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
