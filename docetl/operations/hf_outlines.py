from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, create_model
from docetl.operations.base import BaseOperation
from outlines import generate, models
import json

class HuggingFaceMapOperation(BaseOperation):
    class schema(BaseOperation.schema):
        name: str
        type: str = "hf_map"
        model_path: str
        output_schema: Dict[str, Any]
        prompt_template: str
        max_tokens: int = 4096

    def __init__(self, config: Dict[str, Any], runner=None, *args, **kwargs):
        super().__init__(
            config=config,
            default_model=config.get('default_model', config['model_path']),
            max_threads=config.get('max_threads', 1),
            runner=runner
        )
        
        self.model = models.transformers(
            self.config["model_path"]
        )
        
        # Create a dynamic Pydantic model from the output schema
        field_definitions = {
            k: (eval(v) if isinstance(v, str) else v, ...)
            for k, v in self.config["output_schema"].items()
        }
        output_model = create_model('OutputModel', **field_definitions)
        
        self.processor = generate.json(
            self.model,
            output_model
        )

    def syntax_check(self) -> None:
        """Validate the operation configuration."""
        self.schema(**self.config)

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item through the model."""
        try:
            result = self.processor(self.config["prompt_template"] + "\n" + str(item))
            result_dict = result.model_dump()
            final_dict = {**item, **result_dict}
            return final_dict
        except Exception as e:
            self.console.print(f"Error processing item: {e}")
            return item

    @classmethod
    def execute(cls, config: Dict[str, Any], input_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """Execute the operation on the input data."""
        instance = cls(config)
        results = [instance.process_item(item) for item in input_data]
        return results, 0.0