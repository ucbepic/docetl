import datetime
import math
import os
from inspect import isawaitable
from typing import Dict, Optional

import pyrate_limiter
from rich.console import Console

from docetl.console import get_console
from docetl.operations.utils import APIWrapper
from docetl.utils import decrypt, load_config


class BucketCollection(pyrate_limiter.BucketFactory):
    def __init__(self, **buckets):
        self.clock = pyrate_limiter.TimeClock()
        self.buckets = buckets

    def wrap_item(self, name: str, weight: int = 1) -> pyrate_limiter.RateItem:
        now = self.clock.now()

        async def wrap_async():
            return pyrate_limiter.RateItem(name, await now, weight=weight)

        def wrap_sync():
            return pyrate_limiter.RateItem(name, now, weight=weight)

        return wrap_async() if isawaitable(now) else wrap_sync()

    def get(self, item: pyrate_limiter.RateItem) -> pyrate_limiter.AbstractBucket:
        if item.name not in self.buckets:
            return self.buckets["unknown"]
        return self.buckets[item.name]


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

        buckets = {
            param: pyrate_limiter.InMemoryBucket(
                [
                    pyrate_limiter.Rate(
                        param_limit["count"],
                        param_limit["per"]
                        * getattr(
                            pyrate_limiter.Duration,
                            param_limit.get("unit", "SECOND").upper(),
                        ),
                    )
                    for param_limit in param_limits
                ]
            )
            for param, param_limits in self.config.get("rate_limits", {}).items()
        }
        buckets["unknown"] = pyrate_limiter.InMemoryBucket(
            [pyrate_limiter.Rate(math.inf, 1)]
        )
        bucket_factory = BucketCollection(**buckets)
        self.rate_limiter = pyrate_limiter.Limiter(bucket_factory, max_delay=math.inf)

        self.api = APIWrapper(self)

    def reset_env(self):
        os.environ = self._original_env
