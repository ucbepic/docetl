"""Try CUAD pipelines by hand on a warm Quail Server, on Modal.

``worker`` starts one quail-server for ``--model`` and watches the
``docetl-quail-moar`` volume's ``trials/<model>/inbox`` folder. For each
pipeline YAML dropped there, it points the pipeline at the CUAD train (or
test) data, runs it, scores it, and writes ``trials/<model>/out/<name>.json``
(F1, cost, per-clause scores) and the pipeline's output rows. It stops after
``--idle-minutes`` with an empty inbox.

    cd /path/to/quail
    DOCETL_REPO=/path/to/docetl modal run --detach experiments/quail_moar/modal_trials.py --model qwen3-4b-fp8

Then upload pipelines with ``modal volume put docetl-quail-moar x.yaml /trials/qwen3-4b-fp8/inbox/x.yaml``.
A pipeline may name ``quail/<model>`` models; the worker fills in ``quail_models``.
A top-level ``trial_data: test`` runs it on the test data.
"""

import json
import os
import subprocess
import sys
import time
import traceback

import modal

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/opt/docetl/experiments/quail_moar")  # inside the container
from modal_quail_moar import H100_PER_HOUR, _serve, hf_cache, image, kernel_cache, results  # noqa: E402

app = modal.App("docetl-quail-trials")

RUNNER = r'''
import json, sys, time, yaml
sys.path.insert(0, "/opt/docetl")
import run_quail_moar as r
from docetl.runner import DSLRunner
from experiments.reasoning.evaluation.cuad import evaluate_results
spec = json.loads(sys.argv[1])
config = yaml.safe_load(open(spec["yaml"]))
if spec.get("reduce_model"):
    r.route_reduces(spec["reduce_model"])
r.use_reasoning_effort("medium")
data = spec["test"] if config.pop("trial_data", "train") == "test" else spec["train"]
for dataset in config["datasets"].values():
    dataset["path"] = data
config["pipeline"]["output"]["path"] = spec["output"]
config["pipeline"]["output"].pop("intermediate_dir", None)
config["quail_models"] = {spec["model"]: {"endpoint": spec["endpoint"], "gpu_cost_per_hour": spec["price"]}}
config.setdefault("default_model", "quail/" + spec["model"])
started = time.time()
cost = DSLRunner(config, max_threads=16).load_run_save()
scores = evaluate_results("trial", spec["output"], spec["gt"], data)
json.dump({"cost": cost, "seconds": time.time() - started, "data": data, **scores},
          open(spec["scores"], "w"), indent=1, default=str)
'''


@app.function(
    image=image, gpu="H100", timeout=8 * 3600, memory=65_536,
    secrets=[modal.Secret.from_name("docetl-gemini")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/kernels": kernel_cache,
             "/results": results},
)
def worker(model: str, idle_minutes: int = 45, reduce_model: str = "") -> str:
    servers, endpoints = _serve([model])
    base = f"/results/trials/{model}"
    inbox, out = f"{base}/inbox", f"{base}/out"
    data = "/results/trials/data"
    os.makedirs(inbox, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    with open("/tmp/trial_runner.py", "w") as f:
        f.write(RUNNER)
    last = time.time()
    try:
        while time.time() - last < idle_minutes * 60:
            results.reload()
            names = sorted(n for n in os.listdir(inbox) if n.endswith(".yaml"))
            if not names:
                time.sleep(15)
                continue
            for name in names:
                stem = name[:-5]
                src = f"{inbox}/{name}"
                os.makedirs(f"{base}/done", exist_ok=True)
                os.replace(src, f"{base}/done/{name}")
                results.commit()
                spec = {
                    "yaml": f"{base}/done/{name}", "model": model, "endpoint": endpoints[model],
                    "price": H100_PER_HOUR, "train": f"{data}/cuad_train.json",
                    "test": f"{data}/cuad_test.json", "gt": f"{data}/cuad_gt.csv",
                    "output": f"{out}/{stem}.rows.json", "scores": f"{out}/{stem}.json",
                    "reduce_model": reduce_model,
                }
                print(f"=== trial {stem}", flush=True)
                done = subprocess.run(
                    ["/opt/dvenv/bin/python", "/tmp/trial_runner.py", json.dumps(spec)],
                    env={**os.environ, "PYTHONPATH": "/opt/docetl:/opt/docetl/experiments/quail_moar"},
                    capture_output=True, text=True,
                )
                tail = (done.stdout + done.stderr)[-6000:]
                with open(f"{out}/{stem}.log", "w") as f:
                    f.write(done.stdout + done.stderr)
                if done.returncode != 0:
                    with open(spec["scores"], "w") as f:
                        json.dump({"error": tail[-3000:]}, f)
                    print(f"=== trial {stem} FAILED\n{tail[-2000:]}", flush=True)
                else:
                    scores = json.load(open(spec["scores"]))
                    print(f"=== trial {stem}: f1 {scores['avg_f1']:.4f} cost ${scores['cost']:.3f} "
                          f"({scores['seconds']:.0f} s)", flush=True)
                results.commit()
                last = time.time()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        for server in servers:
            server.terminate()
    return "idle"


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", idle_minutes: int = 45, reduce_model: str = ""):
    print(worker.remote(model, idle_minutes, reduce_model))
