__version__ = "0.2.6"

import warnings
import litellm

from docetl.runner import DSLRunner
from docetl.optimizer import Optimizer
from docetl.apis.pd_accessors import SemanticAccessor

# Drop unsupported params for models like gpt-5 that don't support temperature=0
litellm.drop_params = True

# TODO: Remove after https://github.com/BerriAI/litellm/issues/7560 is fixed
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")

__all__ = ["DSLRunner", "Optimizer", "SemanticAccessor"]
