from . import dataset
from .base_schemas import *  # noqa: F403

# ruff: noqa: F403
from .operations import (
    cluster,
    code_operations,
    equijoin,
    extract,
    filter,
    gather,
    map,
    reduce,
    resolve,
    sample,
    split,
    unnest,
)

MapOp = map.MapOperation.schema
ResolveOp = resolve.ResolveOperation.schema
ReduceOp = reduce.ReduceOperation.schema
ParallelMapOp = map.ParallelMapOperation.schema
FilterOp = filter.FilterOperation.schema
EquijoinOp = equijoin.EquijoinOperation.schema
SplitOp = split.SplitOperation.schema
GatherOp = gather.GatherOperation.schema
UnnestOp = unnest.UnnestOperation.schema
ClusterOp = cluster.ClusterOperation.schema
SampleOp = sample.SampleOperation.schema
CodeMapOp = code_operations.CodeMapOperation.schema
CodeReduceOp = code_operations.CodeReduceOperation.schema
CodeFilterOp = code_operations.CodeFilterOperation.schema
ExtractOp = extract.ExtractOperation.schema

OpType = (
    MapOp
    | ResolveOp
    | ReduceOp
    | ParallelMapOp
    | FilterOp
    | EquijoinOp
    | SplitOp
    | GatherOp
    | UnnestOp
    | CodeMapOp
    | CodeReduceOp
    | CodeFilterOp
    | ExtractOp
)

Dataset = dataset.Dataset.schema
