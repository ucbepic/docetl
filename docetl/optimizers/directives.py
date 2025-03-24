"""
Directives for the optimizer.
Each directive contains a "pattern", which is a list of operator types to match, followed by a "skeleton", which is a list of operator types to replace the matched pattern with.
"""

from typing import List, Optional

from pydantic import BaseModel


class OpSkeleton(BaseModel):
    op_type: str
    decomp_hint: Optional[str] = None  # LLM agent will see this hint


class Decomposition(BaseModel):
    pattern: List[str]
    skeleton: List[OpSkeleton]


class InstantiatedDecomposition(BaseModel):
    decomposition: Decomposition
    identifier: str


DECOMPOSITIONS = [
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(op_type="split"),
            OpSkeleton(op_type="gather"),
            OpSkeleton(op_type="map"),
            OpSkeleton(op_type="reduce"),
        ],
    ),
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(op_type="split"),
            OpSkeleton(
                op_type="map",
                decomp_hint="Summarize the chunk, so it can be used as a prefix for the next map operation. So, the next map operation does not need to operate on the chunk alone; it has context on the prefix of the document. The output schema of this map operation should just be a single key with string type.",
            ),
            OpSkeleton(op_type="gather"),
            OpSkeleton(op_type="map"),
            OpSkeleton(op_type="reduce"),
        ],
    ),
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(op_type="split"),
            OpSkeleton(op_type="gather"),
            OpSkeleton(
                op_type="sample",
                decomp_hint="Reduce the number of chunks to process, because the map operation does not actually need to read the entire document.",
            ),
            OpSkeleton(op_type="map"),
            OpSkeleton(op_type="reduce"),
        ],
    ),
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(
                op_type="map*",
                decomp_hint="Break this map operation into 2+ map operations; each building on the other. This is like thinking step by step, and doing each step one by one.",
            ),
            OpSkeleton(op_type="reduce"),
        ],
    ),
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(
                op_type="parallel_map",
                decomp_hint="Break this map operation into 2+ independent map operations; each performing a subset of the main task.",
            ),
            OpSkeleton(
                op_type="map",
                decomp_hint="Unify the outputs of the parallel map operations into a single output to match the original operation's output schema.",
            ),
        ],
    ),
    Decomposition(
        pattern=["reduce"],
        skeleton=[
            OpSkeleton(
                op_type="map",
                decomp_hint="Create a smaller, focused representation of the document to pass to the reduce operation.",
            ),
            OpSkeleton(op_type="reduce"),
        ],
    ),
]
