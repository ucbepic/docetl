from .base_schemas import *

from .operations import cluster
from .operations import equijoin
from .operations import filter
from .operations import gather
from .operations import map
from .operations import reduce
from .operations import resolve
from .operations import sample
from .operations import split
from .operations import unnest

from . import dataset

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
