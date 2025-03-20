"""
Directives for the optimizer.
Each directive contains a "pattern", which is a list of operator types to match, followed by a "skeleton", which is a list of operator types to replace the matched pattern with.
"""

DECOMPOSITIONS = [
    {"pattern": ["map"], "skeleton": ["split", "gather", "map", "reduce"]},
    {"pattern": ["map"], "skeleton": ["split", "gather", "sample", "map", "reduce"]},
    {"pattern": ["map"], "skeleton": ["map*"]},
    {"pattern": ["map"], "skeleton": ["parallel_map", "map"]},
    {"pattern": ["reduce"], "skeleton": ["map", "reduce"]},
]
