"""Module-level configuration for docetl's Python API.

All attributes here are exposed as ``docetl.<name>`` via the module
replacement in ``__init__.py``.
"""

# Model selection
default_model: str | None = None
agent_model: str | None = None
fallback_models: list[str] | None = None
fallback_embedding_models: list[str] | None = None

# Execution
max_threads: int | None = None
bypass_cache: bool = False
intermediate_dir: str | None = None

# Rate limiting
rate_limits: dict | None = None
