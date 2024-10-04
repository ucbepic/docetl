import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from docetl.parsing_tools import get_parser, get_parsing_tools
from docetl.schemas import ParsingTool


def create_parsing_tool_map(
    parsing_tools: Optional[List[ParsingTool]],
) -> Dict[str, ParsingTool]:
    """
    Create a mapping of parsing tool names to their corresponding ParsingTool objects.

    Args:
        parsing_tools (Optional[List[ParsingTool]]): A list of ParsingTool objects.

    Returns:
        Dict[str, ParsingTool]: A dictionary mapping tool names to ParsingTool objects.
    """
    if parsing_tools is None:
        return {}

    return {tool.name: tool for tool in parsing_tools}


class Dataset:
    """
    A class representing a dataset with various loading and parsing capabilities.

    Attributes:
        type (str): The type of the dataset ('file' or 'memory').
        source (str): The source of the dataset (currently only 'local' is supported).
        path_or_data (Union[str, List[Dict]]): The file path or in-memory data.
        parsing (List[Dict[str, str]]): A list of parsing tools to apply to the data.
        user_defined_parsing_tool_map (Dict[str, ParsingTool]): A map of user-defined parsing tools.
    """

    def __init__(
        self,
        runner,
        type: str,
        path_or_data: Union[str, List[Dict]],
        source: str = "local",
        parsing: List[Dict[str, str]] = None,
        user_defined_parsing_tool_map: Dict[str, ParsingTool] = {},
    ):
        """
        Initialize a Dataset object.

        Args:
            type (str): The type of the dataset ('file' or 'memory').
            source (str): The source of the dataset (currently only 'local' is supported).
            path_or_data (Union[str, List[Dict]]): The file path or in-memory data.
            parsing (List[Dict[str, str]], optional): A list of parsing tools to apply to the data.
            user_defined_parsing_tool_map (Dict[str, ParsingTool], optional): A map of user-defined parsing tools.
        """
        self.runner = runner
        self.type = self._validate_type(type)
        self.source = self._validate_source(source)
        self.path_or_data = self._validate_path_or_data(path_or_data)
        self.parsing = self._validate_parsing(parsing)
        self.user_defined_parsing_tool_map = user_defined_parsing_tool_map

    def _validate_type(self, type: str) -> str:
        """
        Validate the dataset type.

        Args:
            type (str): The type to validate.

        Returns:
            str: The validated type.

        Raises:
            ValueError: If the type is not 'file' or 'memory'.
        """
        if type not in ["file", "memory"]:
            raise ValueError("Type must be 'file' or 'memory'")
        return type

    def _validate_source(self, source: str) -> str:
        """
        Validate the dataset source.

        Args:
            source (str): The source to validate.

        Returns:
            str: The validated source.

        Raises:
            ValueError: If the source is not 'local'.
        """
        if source != "local":
            raise ValueError("Source must be 'local'")
        return source

    def _validate_path_or_data(
        self, path_or_data: Union[str, List[Dict]]
    ) -> Union[str, List[Dict]]:
        """
        Validate the path or data of the dataset.

        Args:
            path_or_data (Union[str, List[Dict]]): The path or data to validate.

        Returns:
            Union[str, List[Dict]]: The validated path or data.

        Raises:
            ValueError: If the path or data is invalid for the given type.
        """
        if self.type == "file":
            if not isinstance(path_or_data, str):
                raise ValueError("For type 'file', path_or_data must be a string")
            valid_extensions = (".json", ".csv")
            if not path_or_data.lower().endswith(valid_extensions):
                raise ValueError(f"Path must end with one of {valid_extensions}")
        elif self.type == "memory":
            if not isinstance(path_or_data, list):
                raise ValueError(
                    "For type 'memory', path_or_data must be a list of dictionaries"
                )
        return path_or_data

    def _validate_parsing(
        self, parsing_tools: Union[List[Dict[str, str]], None]
    ) -> List[Dict[str, str]]:
        """
        Validate the parsing tools.

        Args:
            parsing_tools (Union[List[Dict[str, str]], None]): The parsing tools to validate.

        Returns:
            List[Dict[str, str]]: The validated parsing tools.

        Raises:
            ValueError: If any parsing tool is invalid.
        """
        if parsing_tools is None:
            return []

        for tool in parsing_tools:
            if (
                not isinstance(tool, dict)
                or "input_key" not in tool
                or "function" not in tool
                or "output_key" not in tool
            ):
                raise ValueError(
                    "Each parsing tool must be a dictionary with 'input_key', 'function', and 'output_key' keys"
                )
            if (
                not isinstance(tool["input_key"], str)
                or not isinstance(tool["function"], str)
                or not isinstance(tool["output_key"], str)
            ):
                raise ValueError(
                    "'input_key', 'function', and 'output_key' in parsing tools must be strings"
                )
            if "function_kwargs" in tool and not isinstance(
                tool["function_kwargs"], dict
            ):
                raise ValueError("'function_kwargs', if present, must be a dictionary")

        return parsing_tools

    def __repr__(self):
        """
        Return a string representation of the Dataset object.

        Returns:
            str: A string representation of the Dataset object.
        """
        return f"Dataset(type='{self.type}', source='{self.source}', path_or_data='{self.path_or_data}', parsing={self.parsing})"

    def load(self) -> List[Dict]:
        """
        Load the dataset from the specified path or return the in-memory data.

        Returns:
            List[Dict]: A list of dictionaries representing the dataset.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        if self.type == "memory":
            return self._apply_parsing_tools(self.path_or_data)

        _, ext = os.path.splitext(self.path_or_data.lower())

        if ext == ".json":
            import json

            with open(self.path_or_data, "r") as f:
                data = json.load(f)
        elif ext == ".csv":
            import csv

            with open(self.path_or_data, "r") as f:
                reader = csv.DictReader(f)
                data = list(reader)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return self._apply_parsing_tools(data)

    def _process_item(
        self,
        item: Dict[str, Any],
        input_key: str,
        output_key: str,
        func: Callable,
        **function_kwargs: Dict[str, Any],
    ):
        if input_key not in item:
            raise ValueError(f"Input key {input_key} not found in item: {item}")
        result = func(self.runner, item[input_key], **function_kwargs)
        if isinstance(result, list):
            return [item.copy() | {output_key: res} for res in result]
        else:
            return [item | {output_key: result}]
          
    def _apply_parsing_tools(self, data: List[Dict]) -> List[Dict]:
        """
        Apply parsing tools to the data.

        Args:
            data (List[Dict]): The data to apply parsing tools to.

        Returns:
            List[Dict]: The data with parsing tools applied.

        Raises:
            ValueError: If a parsing tool is not found or if an input key is missing from an item.
        """
        for tool in self.parsing:
            input_key = tool["input_key"]
            try:
                func = get_parser(tool["function"])
            except KeyError:
                if (
                    self.user_defined_parsing_tool_map
                    and tool["function"] in self.user_defined_parsing_tool_map
                ):
                    func = eval(
                        self.user_defined_parsing_tool_map[
                            tool["function"]
                        ].function_code
                    )
                else:
                    raise ValueError(
                        f"Parsing tool {tool['function']} not found. Please define it or use one of our existing parsing tools: {get_parsing_tools()}"
                    )

            output_key = tool["output_key"]
            function_kwargs = tool.get("function_kwargs", {})
            new_data = []

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._process_item,
                        item,
                        input_key,
                        output_key,
                        func,
                        **function_kwargs,
                    )
                    for item in data
                ]
                for future in as_completed(futures):
                    new_data.extend(future.result())

            data = new_data

        return data

    def sample(self, n: int, random: bool = True) -> List[Dict]:
        """
        Sample n items from the dataset.

        Args:
            n (int): Number of items to sample.
            random (bool): If True, sample randomly. If False, take the first n items.

        Returns:
            List[Dict]: A list of n sampled items.

        Raises:
            ValueError: If the sample size is larger than the dataset size or if the file extension is unsupported.
        """
        if self.type == "memory":
            import random as rd

            data = self.path_or_data
            if n > len(data):
                raise ValueError(
                    f"Sample size {n} is larger than dataset size {len(data)}"
                )
            sampled_data = rd.sample(data, n) if random else data[:n]
            return self._apply_parsing_tools(sampled_data)

        _, ext = os.path.splitext(self.path_or_data.lower())

        if ext == ".json":
            import json
            import random as rd

            with open(self.path_or_data, "r") as f:
                if random:
                    data = json.load(f)
                    if n > len(data):
                        raise ValueError(
                            f"Sample size {n} is larger than dataset size {len(data)}"
                        )
                    sampled_data = rd.sample(data, n)
                else:
                    sampled_data = []
                    for i, line in enumerate(f):
                        if i >= n:
                            break
                        sampled_data.append(json.loads(line))

        elif ext == ".csv":
            import csv
            import random as rd

            with open(self.path_or_data, "r") as f:
                reader = csv.DictReader(f)
                if random:
                    data = list(reader)
                    if n > len(data):
                        raise ValueError(
                            f"Sample size {n} is larger than dataset size {len(data)}"
                        )
                    sampled_data = rd.sample(data, n)
                else:
                    sampled_data = [next(reader) for _ in range(n)]

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return self._apply_parsing_tools(sampled_data)
