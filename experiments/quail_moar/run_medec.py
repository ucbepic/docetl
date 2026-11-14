"""MOAR on MEDEC with hosted Gemini models and open models on Quail Server.

MOAR searches pipeline rewrites for the best cost and accuracy trade-off. Here
its candidate models are Gemini models (through LiteLLM) and open models that
Quail Server runs (``quail/<model>``, priced by GPU time). After the search,
each pipeline on the Pareto frontier runs on a held-out MEDEC test set.

Local run, with Quail Servers already running:

    python experiments/quail_moar/run_medec.py --work-dir out \\
        --quail qwen3-4b-fp8=grpc://gpu-host:50051 \\
        --hosted gemini/gemini-2.5-flash-lite gemini/gemini-2.5-flash

On Modal, ``modal_medec.py`` starts one quail-server per GPU and calls ``run``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

def use_reasoning_effort(effort: str | None, prefix: str = "gemini/gemini-4") -> None:
    """Send ``reasoning_effort`` on every LiteLLM call to models under ``prefix``.

    Installed before DocETL is imported, so MOAR's agent, its directives, and
    every pipeline operation (including ones MOAR adds) get the same effort.
    """
    if not effort:
        return
    import litellm

    def wrap(fn):
        def call(*args, **kwargs):
            model = kwargs.get("model", args[0] if args else "")
            if isinstance(model, str) and model.startswith(prefix):
                kwargs.setdefault("reasoning_effort", effort)
            return fn(*args, **kwargs)

        return call

    litellm.completion = wrap(litellm.completion)
    litellm.acompletion = wrap(litellm.acompletion)


from experiments.quail_moar.medec_data import build as build_data  # noqa: E402
from experiments.reasoning.evaluation.medec import evaluate_results  # noqa: E402

METRIC = "combined_score"
# Modal's H100 price per hour (https://modal.com/pricing), used to price GPU time.
H100_PER_HOUR = 3.95

PROMPT = (
    "You are a physician reviewing a clinical note. The note contains either no "
    "medical error or exactly one error in diagnosis, management, treatment, "
    "pharmacotherapy, or causal organism.\n\n"
    "Clinical note:\n{{ input.Text }}\n\n"
    "Decide whether the note contains a medical error. If it does, copy the "
    "sentence with the error exactly as written, and write the corrected "
    "sentence. If it does not, leave both sentences empty."
)


def build_pipeline(path: Path, data: str, output: str, default_model: str, quail_models: dict) -> Path:
    config = {
        "default_model": default_model,
        "quail_models": quail_models,
        "datasets": {"medec": {"type": "file", "path": data}},
        "operations": [
            {
                "name": "detect_error",
                "type": "map",
                "prompt": PROMPT,
                "output": {
                    "schema": {
                        "is_error": "bool",
                        "error_sentence": "string",
                        "corrected_sentence": "string",
                    }
                },
                "litellm_completion_kwargs": {"max_tokens": 256},
            }
        ],
        "pipeline": {
            "steps": [{"name": "medec", "input": "medec", "operations": ["detect_error"]}],
            "output": {"type": "file", "path": output},
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def evaluate(results_file: str) -> dict:
    return evaluate_results("docetl_moar", results_file)


def run_pipeline(yaml_path: str, data: str, output: str) -> tuple[float, dict]:
    """Run one pipeline YAML on ``data``; return (cost, evaluation)."""
    from docetl.runner import DSLRunner

    config = yaml.safe_load(Path(yaml_path).read_text())
    for dataset in config["datasets"].values():
        dataset["path"] = data
    config["pipeline"]["output"]["path"] = output
    config["pipeline"]["output"].pop("intermediate_dir", None)
    cost = DSLRunner(config, max_threads=16).load_run_save()
    return cost, evaluate(output)


def run(
    work_dir: str,
    quail_models: dict,
    hosted_models: list[str],
    agent_model: str = "gemini/gemini-4-argon",
    iterations: int = 10,
    n_train: int = 100,
    n_test: int = 200,
    pipeline: str | None = None,
    train: str | None = None,
    test: str | None = None,
) -> dict:
    """Run the MOAR search and the held-out test; write and return a summary."""
    from docetl.moar.MOARSearch import MOARSearch
    from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not (train and test):
        train, test = build_data(str(work / "data"), n_train=n_train, n_test=n_test)
    available = [*hosted_models, *(f"quail/{name}" for name in quail_models)]
    if pipeline:
        # A given pipeline (such as the MOAR paper's medec.yaml), pointed at
        # this run's data, models, and output.
        config = yaml.safe_load(Path(pipeline).read_text())
        for dataset in config["datasets"].values():
            dataset["path"] = train
        config["default_model"] = available[0]
        config["quail_models"] = quail_models
        config["pipeline"]["output"]["path"] = str(work / "medec_root.json")
        root = work / "medec.yaml"
        root.write_text(yaml.safe_dump(config, sort_keys=False))
    else:
        root = build_pipeline(
            work / "medec.yaml", train, str(work / "medec_root.json"), available[0], quail_models
        )

    (work / "moar").mkdir(exist_ok=True)
    started = time.time()
    moar = MOARSearch(
        root_yaml_path=str(root),
        available_actions=set(ALL_DIRECTIVES),
        sample_input=json.loads(Path(train).read_text())[:5],
        dataset_name="medec",
        available_models=available,
        evaluate_func=evaluate,
        max_iterations=iterations,
        model=agent_model,
        output_dir=str(work / "moar"),
        custom_metric_key=METRIC,
        sample_dataset_path=train,
    )
    moar.search()
    search_seconds = time.time() - started

    frontier = []
    for plan in moar.pareto_frontier.frontier_plans:
        entry = {
            "yaml": str(plan.yaml_file_path),
            "models": sorted(
                {op.get("model") or yaml.safe_load(Path(plan.yaml_file_path).read_text())["default_model"]
                 for op in yaml.safe_load(Path(plan.yaml_file_path).read_text())["operations"]}
            ),
            "train_cost": plan.cost,
            "train_accuracy": moar.pareto_frontier.plans_accuracy.get(plan),
        }
        try:
            out = str(work / "test" / (Path(plan.yaml_file_path).stem + ".json"))
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            cost, scores = run_pipeline(plan.yaml_file_path, test, out)
            entry.update(test_cost=cost, test_accuracy=scores[METRIC], test_scores=scores)
        except Exception as error:  # noqa: BLE001 - record and keep going
            entry["test_error"] = f"{type(error).__name__}: {error}"
        frontier.append(entry)

    summary = {
        "agent_model": agent_model,
        "available_models": available,
        "quail_models": quail_models,
        "iterations": iterations,
        "pipeline": pipeline,
        "train": train,
        "test": test,
        "n_train": len(json.loads(Path(train).read_text())),
        "n_test": len(json.loads(Path(test).read_text())),
        "search_seconds": round(search_seconds, 1),
        "search_cost": moar.total_search_cost,
        "first_layer": moar.model_stats,
        "frontier": frontier,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--quail", nargs="*", default=[], help="model=endpoint, e.g. qwen3-4b-fp8=grpc://host:50051")
    parser.add_argument("--hosted", nargs="*", default=["gemini/gemini-4-hydrogen", "gemini/gemini-4-argon"])
    parser.add_argument("--agent", default="gemini/gemini-4-argon")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--gpu-cost-per-hour", type=float, default=H100_PER_HOUR)
    parser.add_argument("--pipeline", help="a pipeline YAML to start from (default: a built-in one)")
    parser.add_argument("--train", help="training data JSON (default: sampled from MEDEC)")
    parser.add_argument("--test", help="held-out test data JSON (default: sampled from MEDEC)")
    parser.add_argument("--reasoning-effort", default=None, help="reasoning_effort for gemini-4 models")
    args = parser.parse_args()
    use_reasoning_effort(args.reasoning_effort)
    quail_models = {}
    for item in args.quail:
        name, endpoint = item.split("=", 1)
        quail_models[name] = {"endpoint": endpoint, "gpu_cost_per_hour": args.gpu_cost_per_hour}
    summary = run(args.work_dir, quail_models, args.hosted, args.agent, args.iterations,
                  args.n_train, args.n_test, args.pipeline, args.train, args.test)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
