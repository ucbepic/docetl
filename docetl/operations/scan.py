from typing import Dict, List, Tuple

from docetl.operations.base import BaseOperation


class ScanOperation(BaseOperation):
    class schema(BaseOperation.schema):
        dataset_name: str

    def syntax_check(self) -> None:
        """Validate the scan operation configuration."""
        super().syntax_check()

    def execute(self, input_data: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Execute the scan operation to load data from the configured source.

        Args:
            input_data: Not used in scan operation

        Returns:
            Tuple[List[Dict], float]: Loaded data and cost (0 for scan)
        """

        # Look in the runner.datasets objects
        if self.config["dataset_name"] not in self.runner.datasets:
            raise ValueError(f"Dataset {self.config['dataset_name']} not found")

        return (
            self.runner.datasets[self.config["dataset_name"]].load(),
            0.0,
        )  # Scan has no LLM cost
