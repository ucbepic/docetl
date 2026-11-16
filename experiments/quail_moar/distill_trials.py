"""Write CUAD prompts with clarify_instructions and distill_instructions, for trials.

For each student pipeline (one map on a quail/* model), the script asks the
agent for ``--instances`` prompts with each directive and writes one trial
YAML per prompt, set to run on the test data. distill_instructions learns
from a teacher: the teacher's output rows on the training data, from a
stronger model, with the same task.

    python experiments/quail_moar/distill_trials.py --out trials/ --train cuad_train.json \\
        --student 4b=cuad_quail_qwen3-4b-fp8.yaml:cuad_quail_qwen3-4b-fp8.json \\
        --teacher gemini/gemini-4-hydrogen=cuad_gemini_gemini-4-hydrogen.json
"""

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docetl.reasoning_optimizer.directives.clarify_instructions import ClarifyInstructionsDirective  # noqa: E402
from docetl.reasoning_optimizer.directives.distill_instructions import (  # noqa: E402
    DistillInstructionsDirective,
    teacher_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--student", action="append", required=True, help="tag=pipeline.yaml:train_output.json")
    parser.add_argument("--teacher", required=True, help="model=train_output.json")
    parser.add_argument("--agent", default="gemini/gemini-4-carbon")
    parser.add_argument("--agent-reasoning-effort", default="medium")
    parser.add_argument("--instances", type=int, default=3)
    args = parser.parse_args()

    import run_quail_moar

    run_quail_moar.use_reasoning_effort(None, args.agent, args.agent_reasoning_effort)
    teacher_model, teacher_file = args.teacher.split("=", 1)
    teacher_rows = json.loads(Path(teacher_file).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in args.student:
        tag, files = spec.split("=", 1)
        yaml_file, rows_file = files.split(":", 1)
        config = yaml.safe_load(Path(yaml_file).read_text())
        student_rows = json.loads(Path(rows_file).read_text())
        (op,) = [o for o in config["operations"] if o["type"] == "map"]
        context = teacher_context(teacher_model, op["prompt"], teacher_rows, student_rows,
                                  list(op["output"]["schema"]), "name")
        for directive in (ClarifyInstructionsDirective(), DistillInstructionsDirective()):
            for i in range(args.instances):
                jobs.append((tag, config, op["name"], directive, i, context))

    def instantiate(job):
        tag, config, target, directive, i, context = job
        name = f"{tag}_{directive.name}_{i}"
        try:
            ops, _, cost = directive.instantiate(
                operators=copy.deepcopy(config["operations"]), target_ops=[target], agent_llm=args.agent,
                message_history=[], global_default_model=config.get("default_model"),
                input_file_path=args.train, pipeline_code=config, teacher_context=context,
            )
        except Exception as error:  # noqa: BLE001 - record and keep going
            print(f"{name}: failed: {error}", flush=True)
            return
        trial = {**copy.deepcopy(config), "operations": ops, "trial_data": "test"}
        trial.pop("bypass_cache", None)
        (out / f"{name}.yaml").write_text(yaml.safe_dump(trial, sort_keys=False, width=10000))
        print(f"{name}: written (agent ${cost:.3f})", flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(instantiate, jobs))


if __name__ == "__main__":
    main()
