"""Quail execution engine: run DocETL filter, map, and reduce operations on Quail Server.

Quail (https://github.com/fsdatalab/quail) is a SQL engine with AI functions
that runs local models on GPUs. This package compiles a run of consecutive
DocETL operations into one Quail SQL query and runs it on a Quail Server over
Arrow Flight. See ``docs/execution/quail.md``.
"""

from docetl.engines.quail.compile import (
    QuailUnsupported,
    check_eligible,
    compile_run,
    engine_for,
)

__all__ = ["QuailUnsupported", "check_eligible", "compile_run", "engine_for"]
