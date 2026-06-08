__version__ = "0.2.6"

import sys
import types
import warnings

import litellm

from docetl.runner import DSLRunner
from docetl.optimizer import Optimizer
from docetl.apis.pd_accessors import SemanticAccessor
from docetl.moar.optimizer import MOARResult, OptimizedPipeline
from docetl.utils_evaluation import register_eval

from docetl import _config
from docetl.frame import Frame, read_json, read_csv, read_parquet, from_list, yaml_to_python

# Drop unsupported params for models like gpt-5 that don't support temperature=0
litellm.drop_params = True

# TODO: Remove after https://github.com/BerriAI/litellm/issues/7560 is fixed
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")

__all__ = [
    "DSLRunner",
    "Optimizer",
    "SemanticAccessor",
    "MOARResult",
    "OptimizedPipeline",
    "register_eval",
    "Frame",
    "read_json",
    "read_csv",
    "read_parquet",
    "from_list",
    "yaml_to_python",
    # config attrs
    "default_model",
    "agent_model",
    "fallback_models",
    "fallback_embedding_models",
    "max_threads",
    "bypass_cache",
    "intermediate_dir",
    "rate_limits",
]

_CONFIG_ATTRS = {
    "default_model",
    "agent_model",
    "fallback_models",
    "fallback_embedding_models",
    "max_threads",
    "bypass_cache",
    "intermediate_dir",
    "rate_limits",
}


class _Module(types.ModuleType):
    def __getattr__(self, name):
        if name in _CONFIG_ATTRS:
            return getattr(_config, name)
        raise AttributeError(f"module 'docetl' has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _CONFIG_ATTRS:
            setattr(_config, name, value)
            return
        super().__setattr__(name, value)


_self = sys.modules[__name__]
_new = _Module(__name__)
_new.__dict__.update({k: v for k, v in _self.__dict__.items() if k != "__class__"})
_new.__spec__ = _self.__spec__
sys.modules[__name__] = _new
