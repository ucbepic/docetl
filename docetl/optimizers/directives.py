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
    num_instantiations_needed: int = 1


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
        num_instantiations_needed=2,
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
        num_instantiations_needed=2,
    ),
    Decomposition(
        pattern=["reduce"],
        skeleton=[
            OpSkeleton(
                op_type="map",
                decomp_hint="Create a smaller, focused representation of the document to pass to the reduce operation. For example, the map should extract or synthesize only the relevant information to pass to the reduce operation.",
            ),
            OpSkeleton(op_type="reduce"),
        ],
    ),
    Decomposition(
        pattern=["reduce"],
        skeleton=[
            OpSkeleton(
                op_type="map",
                decomp_hint="Create a smaller, focused representation of the document to pass to the reduce operation. For example, the map should extract or synthesize only the relevant information to pass to the reduce operation.",
            ),
            OpSkeleton(
                op_type="sample",
                decomp_hint="Reduce the number of documents to pass to the reduce operation, because the reduce operation does not actually need to read all the documents.",
            ),
            OpSkeleton(op_type="reduce"),
        ],
    ),
    Decomposition(
        pattern=["reduce"],
        skeleton=[
            OpSkeleton(
                op_type="map",
                decomp_hint="Classify the document (or repeat one of the keys in the document) such that the reduce operation can be run on a sub-group of documents, before rolling up to the final result. For example, if the user-defined reduce operation is to summarize feedback by reduce key = department, you can generate this map operation to classify the document by city, and the next operation will then summarize feedback by city, before the final reduce operation rolls up to the department level.",
            ),
            OpSkeleton(
                op_type="reduce",
                decomp_hint="Run the operation on a sub-group of documents, before rolling up to the final result. For example, if the user-defined reduce operation is to summarize feedback by reduce key = department, this first reduce operation could summarize feedback by department and some other reduce key that exists in the document, and then the second reduce operation could roll up the results by department.",
            ),
            OpSkeleton(
                op_type="reduce",
                decomp_hint="This is a small rewrite of the original reduce operation, which will be run on the finer-grained sub-groups from the previous operation.",
            ),
        ],
    ),
]
