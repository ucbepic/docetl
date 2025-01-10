from typing import Union

from . import dataset

# ruff: noqa: F403
from .base_schemas import *
from .operations import (
    cluster,
    equijoin,
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

OpType = Union[
    MapOp,
    ResolveOp,
    ReduceOp,
    ParallelMapOp,
    FilterOp,
    EquijoinOp,
    SplitOp,
    GatherOp,
    UnnestOp,
]

Dataset = dataset.Dataset.schema
