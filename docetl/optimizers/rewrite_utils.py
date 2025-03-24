import itertools
from typing import List, Optional

from docetl.optimizers.directives import InstantiatedDecomposition, OpSkeleton


class SkeletonNode:
    def __init__(
        self,
        op_type: str,
        original_op: "OpContainer",  # noqa: F821
        synthesized: bool,
        instantiated_rewrite: Optional[InstantiatedDecomposition] = None,
        op_skeleton_metadata: Optional[OpSkeleton] = None,
    ):
        self.op_type = op_type  # The operator type for this skeleton node.
        self.original_op = (
            original_op  # Pointer to the original OpContainer for context.
        )
        self.synthesized = (
            synthesized  # True if this node was generated via a rewrite directive.
        )
        self.children = []  # List of child SkeletonNode objects (supporting a tree).
        self.instantiated_rewrite = instantiated_rewrite
        self.op_skeleton_metadata = op_skeleton_metadata

    def __str__(self):
        tag = " (synth)" if self.synthesized else ""
        rewrite_info = ""
        if self.synthesized and self.instantiated_rewrite:
            rewrite_id = self.instantiated_rewrite.identifier
            pattern = self.instantiated_rewrite.decomposition.pattern
            skeleton = [
                op.op_type for op in self.instantiated_rewrite.decomposition.skeleton
            ]
            rewrite_info = f" [rewrite:{rewrite_id[:8]}:pattern={','.join(pattern)}:skeleton={','.join(skeleton)}]"
        s = f"{self.op_type}{tag}[orig:{self.original_op.config.get('type')}]{rewrite_info}"
        if self.children:
            child_strs = [str(child) for child in self.children]
            s += "\n" + "\n".join(
                [
                    f"  ├─ {child}" if i < len(self.children) - 1 else f"  └─ {child}"
                    for i, child in enumerate(child_strs)
                ]
            )
        return s


def generate_children_skeletons(
    children: List["OpContainer"],  # noqa: F821
) -> List[List[str]]:
    """
    Given a list of OpContainer children, recursively generate candidate skeleton trees for each.
    Returns a list of lists; each inner list represents one candidate combination for the children.
    If there are no children, return a list with one empty list.
    """
    if not children:
        return [[]]
    else:
        # For each child, get candidate skeleton trees.
        all_candidates = [child.generate_skeletons() for child in children]
        # Cartesian product of candidates from each child.
        combinations = list(itertools.product(*all_candidates))
        # Each combination is a tuple; convert to a list.
        return [list(combo) for combo in combinations]


def build_chain_from_directive(
    directive: InstantiatedDecomposition, original_op: "OpContainer"  # noqa: F821
) -> "SkeletonNode":
    """
    Build a linked chain (tree) of SkeletonNode objects from the given list of op types.
    Every node in the chain is tagged with the pointer original_op and marked as synthesized.
    """
    skeleton_list = directive.decomposition.skeleton
    reversed_skeleton_list = skeleton_list[::-1]
    head = SkeletonNode(
        reversed_skeleton_list[0].op_type,
        original_op,
        synthesized=True,
        instantiated_rewrite=directive,
        op_skeleton_metadata=reversed_skeleton_list[0],
    )
    current = head
    for op_skeleton in reversed_skeleton_list[1:]:
        new_node = SkeletonNode(
            op_skeleton.op_type,
            original_op,
            synthesized=True,
            instantiated_rewrite=directive,
            op_skeleton_metadata=op_skeleton,
        )
        current.children.append(new_node)  # For a chain, add as the only child.
        current = new_node

    return head


def get_last_node(skeleton_node):
    """
    Traverse the first-child branch of a SkeletonNode chain to obtain the last node.
    """
    current = skeleton_node
    while current.children:
        current = current.children[0]
    return current


def clone_node(node):
    """
    Clone a SkeletonNode without its children.
    """
    return SkeletonNode(
        node.op_type,
        node.original_op,
        node.synthesized,
        node.instantiated_rewrite,
        node.op_skeleton_metadata,
    )


def clone_chain(chain_head):
    """
    Clone an entire linked chain (tree) of SkeletonNode objects.
    """
    new_head = clone_node(chain_head)
    current_old = chain_head.children[0] if chain_head.children else None
    current_new = new_head
    while current_old:
        new_node = clone_node(current_old)
        current_new.children.append(new_node)
        current_new = new_node
        current_old = current_old.children[0] if current_old.children else None
    return new_head
