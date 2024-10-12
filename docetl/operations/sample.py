from typing import Any, Dict, List, Optional, Tuple
from docetl.operations.base import BaseOperation


class SampleOperation(BaseOperation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def syntax_check(self) -> None:
        """
        Checks the configuration of the SampleOperation for required keys and valid structure.

        Raises:
            ValueError: If required keys are missing or invalid in the configuration.
            TypeError: If configuration values have incorrect types.
        """
        pass

    def execute(
        self, input_data: List[Dict], is_build: bool = False
    ) -> Tuple[List[Dict], float]:
        """
        Executes the sample operation on the input data.

        Args:
            input_data (List[Dict]): A list of dictionaries to process.
            is_build (bool): Whether the operation is being executed
              in the build phase. Defaults to False.

        Returns:
            Tuple[List[Dict], float]: A tuple containing the filtered
              list of dictionaries and the total cost of the operation.
        """

        samples = self.config["samples"]
        if isinstance(samples, list):
            keys = list(samples[0].keys())
            key_to_doc = {tuple([doc[key] for key in keys]): doc for doc in input_data}

            output_data = [
                key_to_doc[tuple([sample[key] for key in keys])] for sample in samples
            ]
        else:
            stratify = None
            if "stratify" in self.config:
                stratify = [data[self.config["stratify"]] for data in input_data]

            import sklearn.model_selection

            output_data, dummy = sklearn.model_selection.train_test_split(
                input_data,
                train_size=samples,
                random_state=self.config.get("random_state", None),
                stratify=stratify,
            )
        return output_data, 0
