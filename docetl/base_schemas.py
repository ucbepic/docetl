from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# from ..operations import map
# MapOp = map.MapOperation.schema

class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(BaseModel):
    code: str
    function: ToolFunction


class ParsingTool(BaseModel):
    """
    Represents a parsing tool used for custom data parsing in the pipeline.

    Attributes:
        name (str): The name of the parsing tool. This should be unique within the pipeline configuration.
        function_code (str): The Python code defining the parsing function. This code will be executed
                             to parse the input data according to the specified logic. It should return a list of strings, where each string is its own document.

    Example:
        ```yaml
        parsing_tools:
          - name: ocr_parser
            function_code: |
              import pytesseract
              from pdf2image import convert_from_path
              def ocr_parser(filename: str) -> List[str]:
                  images = convert_from_path(filename)
                  text = ""
                  for image in images:
                      text += pytesseract.image_to_string(image)
                  return [text]
        ```
    """

    name: str
    function_code: str


class Dataset(BaseModel):
    """
    Represents a dataset configuration in the pipeline.

    Attributes:
        type (str): The type of the dataset. Must be either 'file' or 'memory'.
        path (str): The path to the dataset file or the in-memory data, depending on the type.
        source (str): The source of the dataset. Currently, only 'local' is supported. Defaults to 'local'.
        parsing (Optional[List[Dict[str, str]]]): A list of parsing tools to apply to the data. Each parsing tool
                                                  is represented by a dictionary with 'input_key', 'function', and
                                                  'output_key' keys. Defaults to None.

    Example:
        ```yaml
        datasets:
          my_dataset:
            type: file
            path: input.json
            parsing:
              - input_key: file_path
                function: txt_to_string
                output_key: content
        ```

    Note:
        The parsing tools are applied in the order they are listed. Each parsing tool takes the output
        of the previous tool as its input, allowing for chained processing of the data.
    """

    type: str
    path: str
    source: str = "local"
    parsing: Optional[List[Dict[str, str]]] = None

class PipelineStep(BaseModel):
    """
    Represents a step in the pipeline.

    Attributes:
        name (str): The name of the step.
        operations (List[Union[Dict[str, Any], str]]): A list of operations to be applied in this step.
            Each operation can be either a string (the name of the operation) or a dictionary
            (for more complex configurations).
        input (Optional[str]): The input for this step. It can be either the name of a dataset
            or the name of a previous step. If not provided, the step will use the output
            of the previous step as its input.

    Example:
        ```python
        # Simple step with a single operation
        process_step = PipelineStep(
            name="process_step",
            input="my_dataset",
            operations=["process"]
        )

        # Step with multiple operations
        summarize_step = PipelineStep(
            name="summarize_step",
            input="process_step",
            operations=["summarize"]
        )

        # Step with a more complex operation configuration
        custom_step = PipelineStep(
            name="custom_step",
            input="previous_step",
            operations=[
                {
                    "custom_operation": {
                        "model": "gpt-4",
                        "prompt": "Perform a custom analysis on the following text:"
                    }
                }
            ]
        )
        ```

    These examples show different ways to configure pipeline steps, from simple
    single-operation steps to more complex configurations with custom parameters.
    """

    name: str
    operations: List[Union[Dict[str, Any], str]]
    input: Optional[str] = None


class PipelineOutput(BaseModel):
    """
    Represents the output configuration for a pipeline.

    Attributes:
        type (str): The type of output. This could be 'file', 'database', etc.
        path (str): The path where the output will be stored. This could be a file path,
                    database connection string, etc., depending on the type.
        intermediate_dir (Optional[str]): The directory to store intermediate results,
                                          if applicable. Defaults to None.

    Example:
        ```python
        output = PipelineOutput(
            type="file",
            path="/path/to/output.json",
            intermediate_dir="/path/to/intermediate/results"
        )
        ```
    """

    type: str
    path: str
    intermediate_dir: Optional[str] = None
