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

# Logical-plan rewrites (docetl.plan): True (default rules), False, or a
# rule name / list of rule names to enable (opt-in rules included).
# Consulted by every in-process DSLRunner construction (Frame, dicts,
# from_yaml) as the default when the config has no plan_rewrites key;
# the CLI runs in its own process, so only the YAML key reaches it.
plan_rewrites: "bool | str | list" = True

# Optional system prompt applied to all operations:
# {"dataset_description": ..., "persona": ...}
system_prompt: dict | None = None


def runner_settings() -> dict:
    """Top-level runner config entries derived from the module-level settings.

    Single source of truth for propagating ``docetl.<attr>`` globals into a
    runner config — used by both the Frame API and the pandas accessors.
    """
    settings: dict = {}
    if default_model:
        settings["default_model"] = default_model
    if rate_limits:
        settings["rate_limits"] = rate_limits
    if bypass_cache:
        settings["bypass_cache"] = True
    if fallback_models:
        settings["fallback_models"] = fallback_models
    if fallback_embedding_models:
        settings["fallback_embedding_models"] = fallback_embedding_models
    if system_prompt:
        settings["system_prompt"] = system_prompt
    if plan_rewrites is not True:  # default-on; only emit overrides
        settings["plan_rewrites"] = plan_rewrites
    return settings
