__version__ = "0.2.3"

import warnings

# TODO: Remove after https://github.com/BerriAI/litellm/issues/7560 is fixed
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")

from docetl.runner import DSLRunner
from docetl.optimizer import Optimizer
from docetl.apis.pd_accessors import SemanticAccessor

__all__ = ["DSLRunner", "Optimizer", "SemanticAccessor"]
