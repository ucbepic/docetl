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
    hint: Optional[str] = None  # LLM agent will see this hint


class InstantiatedDecomposition(BaseModel):
    decomposition: Decomposition
    identifier: str


DECOMPOSITIONS = [
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(op_type="split"),
            OpSkeleton(op_type="map"),
            OpSkeleton(op_type="reduce"),
        ],
        hint="This is a simple decomposition that splits the document into chunks, and then maps over the chunks to process them. This is a good starting point for long documents in map operations, and should be invoked when the task **does not** require knowledge of the document outside of the current chunk (since chunks are processed independently).",
    ),
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(op_type="split"),
            OpSkeleton(op_type="gather"),
            OpSkeleton(op_type="map"),
            OpSkeleton(op_type="reduce"),
        ],
        hint="This is a simple decomposition that splits the document into chunks, augments each chunk with a previous or next chunk (or both), and then maps over the chunks to process them. This is a good starting point for long documents in map operations, and should be invoked when the task requires knowledge of the document outside of the current chunk (but the knowledge can be found in the immediate vicinity of the current chunk).",
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
        hint="This is a decomposition that splits the document into chunks, augments each chunk with the summary of all previous chunks, and then processes the chunk. This is a good starting point for long documents in map operations, where the task requires knowledge of the document outside of the current chunk.",
    ),
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(
                op_type="sample",
                decomp_hint="Reduce the number of chunks to process, because the map operation does not actually need to read the entire document.",
            ),
            OpSkeleton(op_type="map"),
        ],
        hint="This decomposition samples a subset of the dataset and runs the map operation on the subset. This is useful for when the task does not require processing of all documents (e.g., we can take a random or stratified sample). Sampling can be applied to chunks (when the map operation is chunk-level) or documents (when the map operation is document-level).",
    ),
    Decomposition(
        pattern=["map"],
        skeleton=[
            OpSkeleton(
                op_type="map*",
                decomp_hint="Break this map operation into 2+ map operations; each building on the other. This is like thinking step by step, and doing each step one by one.",
            )
        ],
        num_instantiations_needed=2,
        hint="This decomposition breaks the map operation into 2+ map operations; each building on the other. This is like thinking step by step, and doing each step one by one.",
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
        hint="This decomposition breaks the map operation into 2+ independent map operations; each performing a subset of the main task. This is helpful when the task is a map operation that asks for many different output keys.",
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
        hint="This decomposition reduces the size of each document passed to the reduce operation. The map operation should extract or summarize or synthesize the information needed to do the reduce operation correctly.",
    ),
    Decomposition(
        pattern=["reduce"],
        skeleton=[
            OpSkeleton(
                op_type="sample",
                decomp_hint="Reduce the number of documents to pass to the reduce operation, because the reduce operation does not actually need to read all the documents.",
            ),
            OpSkeleton(op_type="reduce"),
        ],
        hint="This decomposition samples a subset of the dataset and runs the reduce operation on the subset. This is useful for when the task does not require processing of all documents (e.g., we can take a random or stratified sample). Sampling can be applied to chunks (when the reduce operation is chunk-level) or documents (when the reduce operation is document-level).",
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
        hint="This decomposition breaks the reduce operation into 2+ reduce operations; each building on the other. This is like thinking step by step, and doing each step one by one.",
    ),
]
