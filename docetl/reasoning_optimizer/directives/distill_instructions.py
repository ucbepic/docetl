"""Rewrite a small model's prompt so it reasons like a stronger plan.

``distill_instructions`` is ``clarify_instructions`` with a teacher: the most
accurate plan found so far that runs on another model. The agent sees the
teacher's prompt and, for a few documents, where the teacher's output and the
current plan's output differ, and writes the teacher's evident reasoning into
the current operation's prompt as explicit instructions.
"""

import json
from typing import Any, Dict, List

from pydantic import Field

from .clarify_instructions import ClarifyInstructionsDirective

MAX_DOCUMENTS = 6
MAX_VALUE_CHARS = 600


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + " ..."


def _differences(teacher: Any, student: Any) -> List[str]:
    """Lines describing where two values of one output field differ.

    Lists of objects are lined up by their first field (such as a clause
    type), so only the entries that differ are shown.
    """
    if (
        isinstance(teacher, list) and isinstance(student, list)
        and all(isinstance(x, dict) and x for x in teacher + student)
    ):
        key = next(iter((teacher or student)[0]))
        t = {str(x.get(key)): x for x in teacher}
        s = {str(x.get(key)): x for x in student}
        lines = []
        for name in list(dict.fromkeys([*t, *s])):
            a, b = t.get(name), s.get(name)
            if a != b:
                lines.append(f"  - {key} {name!r}: stronger plan {_short(a)} | current plan {_short(b)}")
        return lines
    return [] if teacher == student else [f"  stronger plan: {_short(teacher)}", f"  current plan: {_short(student)}"]


def teacher_context(
    teacher_model: str,
    teacher_prompt: str,
    teacher_rows: List[Dict],
    student_rows: List[Dict],
    output_keys: List[str],
    id_key: str,
) -> str:
    """The teacher section of the agent's first message.

    Args:
        teacher_model: The model the stronger plan runs on.
        teacher_prompt: The stronger plan's prompt for the same task.
        teacher_rows: The stronger plan's output rows.
        student_rows: The current plan's output rows for the same inputs.
        output_keys: The output fields to compare.
        id_key: The input field that names a document in both outputs.
    """
    by_id = {str(r.get(id_key)): r for r in student_rows}
    shown = []
    for row in teacher_rows:
        other = by_id.get(str(row.get(id_key)))
        if other is None:
            continue
        lines = []
        for key in output_keys:
            lines += _differences(row.get(key), other.get(key))
        if lines:
            shown.append(f"Document {row.get(id_key)!r}:\n" + "\n".join(lines))
        if len(shown) >= MAX_DOCUMENTS:
            break
    return (
        f"\n\nA stronger plan for the same task runs on {teacher_model} and scores higher. Its prompt:\n"
        f"---\n{teacher_prompt}\n---\n"
        f"Here is where its output differs from the current plan's output on the same documents:\n\n"
        + ("\n\n".join(shown) or "(no differences found)")
        + "\n\nThe current plan runs on a smaller model. Work out the reasoning behind the stronger plan's "
        "choices (what it includes, leaves out, and how much text it copies), read sample documents to "
        "check it, and write that reasoning into the improved prompt as explicit instructions the smaller "
        "model can follow. Do not copy answers for specific documents into the prompt."
    )


class DistillInstructionsDirective(ClarifyInstructionsDirective):
    name: str = Field(default="distill_instructions", description="The name of the directive")
    nl_description: str = Field(
        default="Rewrites a single operation's prompt for a smaller model by distilling the reasoning of a stronger plan: the agent compares the stronger plan's outputs with the current plan's on the same documents and writes the stronger plan's evident reasoning into the prompt as explicit instructions"
    )
    when_to_use: str = Field(
        default="When the operation runs on a small or cheap model and a more accurate plan on a stronger model exists, to close the accuracy gap without paying for the stronger model"
    )

    def __eq__(self, other):
        return isinstance(other, DistillInstructionsDirective)

    def __hash__(self):
        return hash("DistillInstructionsDirective")

    def instantiate(
        self,
        operators: List[Dict],
        target_ops: List[str],
        agent_llm: str,
        message_history: list = [],
        global_default_model: str = None,
        **kwargs,
    ):
        """Like clarify_instructions, with ``teacher_context`` (the text
        ``teacher_context()`` builds) added to the agent's first message."""
        extra = kwargs.get("teacher_context") or ""
        if not extra:
            raise ValueError("distill_instructions needs a stronger plan to learn from (teacher_context)")
        input_file_path = kwargs.get("input_file_path")
        if not input_file_path:
            raise ValueError("input_file_path is required for distill_instructions")
        target_ops_configs = [op for op in operators if op["name"] in target_ops]
        rewrite, message_history, call_cost = self.llm_instantiate(
            target_ops_configs, input_file_path, agent_llm, message_history,
            kwargs.get("pipeline_code"), extra_context=extra,
        )
        return (
            self.apply(global_default_model, operators, target_ops, rewrite),
            message_history,
            call_cost,
        )
