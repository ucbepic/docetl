"""
The BaseOperation class is an abstract base class for all operations in the docetl framework. It provides a common structure and interface for various data processing operations.
"""

from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

from docetl.operations.utils import APIWrapper
from rich.console import Console
from rich.status import Status
import jsonschema
from pydantic import BaseModel
import jinja2

# FIXME: This should probably live in some utils module?
class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class BaseOperationMeta(ABCMeta):
    def __new__(cls, *arg, **kw):
        self = ABCMeta.__new__(cls, *arg, **kw)
        self.schema.__name__ = self.__name__
        return self


class BaseOperation(ABC, metaclass=BaseOperationMeta):
    def __init__(
        self,
        runner: "ConfigWrapper",
        config: Dict,
        default_model: str,
        max_threads: int,
        console: Optional[Console] = None,
        status: Optional[Status] = None,
        **kwargs,
    ):
        """
        Initialize the BaseOperation.

        Args:
            config (Dict): Configuration dictionary for the operation.
            default_model (str): Default language model to use.
            max_threads (int): Maximum number of threads for parallel processing.
            console (Optional[Console]): Rich console for outputting logs. Defaults to None.
            status (Optional[Status]): Rich status for displaying progress. Defaults to None.
        """
        assert "name" in config, "Operation must have a name"
        assert "type" in config, "Operation must have a type"
        self.runner = runner
        self.config = config
        self.default_model = default_model
        self.max_threads = max_threads
        self.console = console or Console()
        self.manually_fix_errors = self.config.get("manually_fix_errors", False)
        self.status = status
        self.num_retries_on_validate_failure = self.config.get(
            "num_retries_on_validate_failure", 0
        )
        self.syntax_check()
        self.compiled_configs = {}

    # This must be overridden in a subclass
    class schema(BaseModel, extra="allow"):
        name: str
        type: str

    @classproperty
    def json_schema(cls):
        assert hasattr(
            cls.schema, "model_json_schema"
        ), "Programming error: %s.schema must be a pydantic object but is a %s" % (
            cls,
            type(cls.schema),
        )
        return cls.schema.model_json_schema()

    @abstractmethod
    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Execute the operation on the input data.

        This method should be implemented by subclasses to perform the
        actual operation on the input data.

        Args:
            input_data (List[Dict]): List of input data items.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the processed data
            and the total cost of the operation.
        """
        pass

    @abstractmethod
    def syntax_check(self) -> None:
        """
        Perform syntax checks on the operation configuration.

        This method should be implemented by subclasses to validate the
        configuration specific to each operation type.

        Raises:
            ValueError: If the configuration is invalid.
        """
        jsonschema.validate(instance=self.config, schema=self.json_schema)

    def gleaning_check(self) -> None:
        """
        Perform checks on the gleaning configuration.

        This method validates the gleaning configuration if it's present
        in the operation config.

        Raises:
            ValueError: If the gleaning configuration is invalid.
            TypeError: If the gleaning configuration has incorrect types.
        """
        if "gleaning" not in self.config:
            return
        if "num_rounds" not in self.config["gleaning"]:
            raise ValueError("Missing 'num_rounds' in 'gleaning' configuration")
        if not isinstance(self.config["gleaning"]["num_rounds"], int):
            raise TypeError(
                "'num_rounds' in 'gleaning' configuration must be an integer"
            )
        if self.config["gleaning"]["num_rounds"] < 1:
            raise ValueError(
                "'num_rounds' in 'gleaning' configuration must be at least 1"
            )

        if "validation_prompt" not in self.config["gleaning"]:
            raise ValueError("Missing 'validation_prompt' in 'gleaning' configuration")
        if not isinstance(self.config["gleaning"]["validation_prompt"], str):
            raise TypeError(
                "'validation_prompt' in 'gleaning' configuration must be a string"
            )
        if not self.config["gleaning"]["validation_prompt"].strip():
            raise ValueError(
                "'validation_prompt' in 'gleaning' configuration cannot be empty"
            )

    def evaluate_expression(self, config_path, default_expression=None, default_value=None, **context):
        """Evaluates a jinja2 expression specified in the operation
        config (or a default expression if not found) against a given
        context.

        config_path is itself a jinja2 expression, evaluated within
        the context of the operation config (self.config) and one
        additional variable: config, which is bound to the entire
        config file.

        Evaluating config_path should yield a jinja2 expression, or
        alternatively a list of expressions (if so, a list of values
        is returned).

        If no expression is found and no default expression is
        provided, or the expression evaluates to Undefined,
        default_value is returned.

        Example:

        Assuming
        
          config_path = "input.title_keys"
          self.config = {"input": {"title_keys": ["title", "categories.0.title"]}}
          context = {"input": {"title": "Hello", "categories": [{"title": "world"}]}}

        this function will return ["Hello", "world"].

        """
        if config_path not in self.compiled_configs:
            env = jinja2.Environment()
            expression = env.compile_expression(config_path)(
                config=self.runner.config,
                **self.config)
            if expression is None:
                expression = default_expression
            if isinstance(expression, (list, tuple)):
                self.compiled_configs[config_path] = [env.compile_expression(e, undefined_to_none=False) for e in expression]
            elif isinstance(expression, str):
                self.compiled_configs[config_path] = env.compile_expression(expression, undefined_to_none=False)
            else:
                self.compiled_configs[config_path] = None
        expr = self.compiled_configs[config_path]
        if expr is None:
            return default_value
        def expr_or_default(expr):
            res = expr(**context)
            if res is jinja2.Undefined:
                return default_value
            return res
        if isinstance(expr, list):
            return [expr_or_default(e) for e in expr]
        return expr_or_default(expr)

    def evaluate_template(self, config_path, default_template=None, **context):
        """Renders a jinja2 template specified in the operation config
        (or a default template if not found) against a given context.

        config_path is a jinja2 expression, evaluated within
        the context of the operation config (self.config) and one
        additional variable: config, which is bound to the entire
        config file.
        """
        if config_path not in self.compiled_configs:
            env = jinja2.Environment()
            template = env.compile_expression(config_path)(
                config=self.runner.config,
                **self.config)
            if template is None:
                template = default_template
            self.compiled_configs[config_path] = jinja2.Template(template)
        return self.compiled_configs[config_path].render(**context)
