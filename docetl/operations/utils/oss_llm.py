'''
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, create_model
from outlines import generate, models
import json
import hashlib
from docetl.operations.base import BaseOperation 
from .llm import LLMResult, InvalidOutputError

class OSSLLMOperation(BaseOperation):
    class schema(BaseOperation.schema):
        name: str
        type: str = "oss_llm"
        model_path: str
        output_schema: Dict[str, Any]
        max_tokens: int = 4096

    def __init__(self, config: Dict[str, Any], runner=None, *args, **kwargs):
        super().__init__(
            config=config,
            default_model=config.get('default_model', config['model_path']),
            max_threads=config.get('max_threads', 1),
            runner=runner
        )
        self.model = models.transformers(self.config["model_path"])
        self._setup_output_model()

    def _setup_output_model(self):
        field_definitions = {
            k: (eval(v) if isinstance(v, str) else v, ...)
            for k, v in self.config["output_schema"].items()
        }
        self.output_model = create_model('OutputModel', **field_definitions)
        self.processor = generate.json(self.model, self.output_model)


    def syntax_check(self):
        self.schema(**self.config)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt_parts = []
        for msg in messages:
            role_prefix = {
                "system": "System: ",
                "user": "User: ",
                "assistant": "Assistant: "
            }.get(msg["role"], "")
            prompt_parts.append(f"{role_prefix}{msg['content']}")
        return "\n".join(prompt_parts)

    def _to_litellm_format(self, result: Any) -> Dict[str, Any]:
        result_dict = result.model_dump()
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "function": {
                            "name": "send_output",
                            "arguments": json.dumps(result_dict)
                        },
                        "id": "call_" + hashlib.md5(
                            json.dumps(result_dict).encode()
                        ).hexdigest(),
                        "type": "function"
                    }]
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "model": self.config["model_path"],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        return response

    def process_messages(self, messages: List[Dict[str, str]]) -> LLMResult:
        try:
            prompt = self._messages_to_prompt(messages)
            result = self.processor(prompt)
            response = self._to_litellm_format(result)
            return LLMResult(response=response, total_cost=0.0, validated=True)
        except Exception as e:
            raise InvalidOutputError(
                message=str(e),
                output=str(e),
                expected_schema=self.config["output_schema"],
                messages=messages
            )

    @classmethod
    def execute(cls, config: Dict[str, Any], messages: List[Dict[str, str]]) -> Tuple[LLMResult, float]:
        instance = cls(config)
        result = instance.process_messages(messages)
        return result, 0.0



class OSSLLMWrapper:
    def __init__(self):
        self._models = {}
        self._processors = {}

    def setup_model(self, model_path: str, output_schema: Dict[str, Any] = None):
        """Initialize OSS model and processor if needed"""
        if model_path not in self._models:
            self._models[model_path] = models.transformers(model_path)
            
            if output_schema:
                field_definitions = {
                    k: (eval(v) if isinstance(v, str) else v, ...)
                    for k, v in output_schema.items()
                }
                output_model = create_model('OutputModel', **field_definitions)
                self._processors[model_path] = generate.json(
                    self._models[model_path], 
                    output_model
                )

    def process_messages(
        self,
        model_path: str,
        messages: List[Dict[str, str]],
        output_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process messages through OSS model"""
        self.setup_model(model_path, output_schema)
        
        # Process with OSS model
        result = self._processors[model_path]("\n".join(
            msg["content"] for msg in messages
        ))
        
        # Convert to litellm format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "arguments": json.dumps(result.model_dump())
                        }
                    }]
                }
            }],
            "model": f"local/{model_path}",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
'''
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, create_model
from outlines import generate, models
import json
import hashlib
from .llm import LLMResult, InvalidOutputError

class OutlinesBackend:
    """Backend for handling Outlines (local) models in DocETL operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Outlines backend.
        
        Args:
            config: Optional configuration dictionary containing global settings
        """
        self._models = {}
        self._processors = {}
        self.config = config or {}

    def setup_model(self, model_path: str, output_schema: Dict[str, Any] = None):
        """Initialize Outlines model and processor if needed.
        
        Args:
            model_path: Path to the model, without the 'outlines/' prefix
            output_schema: Schema for the expected output
        """
        if model_path not in self._models:
            model_kwargs = {
                k: v for k, v in self.config.items() 
                if k in ['max_tokens']
            }
            self._models[model_path] = models.transformers(model_path, **model_kwargs)
            
            if output_schema:
                field_definitions = {
                    k: (eval(v) if isinstance(v, str) else v, ...)
                    for k, v in output_schema.items()
                }
                output_model = create_model('OutputModel', **field_definitions)
                self._processors[model_path] = generate.json(
                    self._models[model_path], 
                    output_model
                )

    def process_messages(
        self,
        model_path: str,
        messages: List[Dict[str, str]],
        output_schema: Dict[str, Any]
    ) -> LLMResult:
        """Process messages through Outlines model.
        
        Args:
            model_path: Path to the model, without the 'outlines/' prefix
            messages: List of message dictionaries with 'role' and 'content'
            output_schema: Schema for the expected output
        
        Returns:
            LLMResult containing the model's response in LiteLLM format
        """
        try:
            self.setup_model(model_path, output_schema)
            
            prompt = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in messages
            )
            
            result = self._processors[model_path](prompt)
            
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "function": {
                                "name": "send_output",
                                "arguments": json.dumps(result.model_dump())
                            },
                            "id": "call_" + hashlib.md5(
                                json.dumps(result.model_dump()).encode()
                            ).hexdigest(),
                            "type": "function"
                        }]
                    },
                    "finish_reason": "stop",
                    "index": 0
                }],
                "model": f"outlines/{model_path}",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            return LLMResult(response=response, total_cost=0.0, validated=True)
            
        except Exception as e:
            raise InvalidOutputError(
                message=str(e),
                output=str(e),
                expected_schema=output_schema,
                messages=messages
            )