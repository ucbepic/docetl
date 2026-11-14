"""Run the Quail MOAR experiment (MEDEC or CUAD) on Modal.

One container gets one H100 per open model. It starts a quail-server per GPU,
then runs ``run_quail_moar.py`` with Gemini (the ``docetl-gemini`` Modal secret,
holding GEMINI_API_KEY) as MOAR's agent and as the hosted models. Results go
to the ``docetl-quail-moar`` Modal volume.

Run it from a Quail checkout (https://github.com/fsdatalab/quail), whose
``quail.deploy.images`` builds the GPU image. ``--pipeline``, ``--train``, and
``--test`` (and CUAD's ``--ground-truth``) are local files, such as the MOAR
paper's pipelines and data, uploaded to the volume first:

    cd /path/to/quail
    DOCETL_REPO=/path/to/docetl modal run /path/to/docetl/experiments/quail_moar/modal_quail_moar.py \\
        --workload cuad --pipeline cuad.yaml --train cuad_train.json --test cuad_test.json \\
        --ground-truth CUAD-master_clauses.csv --models qwen3-4b-fp8 --iterations 10
"""

import json
import os
import socket
import subprocess
import time

import modal

from quail.deploy.images import gpu_image

DOCETL = os.environ.get(
    "DOCETL_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
BRANCH = os.environ.get("DOCETL_BRANCH", "claude/nifty-meitner-afmb36")
IGNORE = [".venv", ".git", "website", "node_modules", "docs", "**/__pycache__",
          "experiments/reasoning/data", "experiments/reasoning/outputs", "site"]
H100_PER_HOUR = 3.95

app = modal.App("docetl-quail-moar")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
results = modal.Volume.from_name("docetl-quail-moar", create_if_missing=True)
# DocETL's dependencies come from the branch on GitHub; the mounted source
# then runs the latest code without rebuilding the image.
image = (gpu_image(commands=(
            "pip install uv",
            "uv venv /opt/dvenv --python 3.12",
            f"VIRTUAL_ENV=/opt/dvenv uv pip install 'docetl[quail] @ git+https://github.com/ucbepic/docetl@{BRANCH}'",
         ))
         .add_local_dir(DOCETL, "/opt/docetl", ignore=IGNORE))


def _serve(models: list[str]) -> tuple[list[subprocess.Popen], dict]:
    servers, endpoints = [], {}
    for gpu, model in enumerate(models):
        port = 50051 + gpu
        servers.append(subprocess.Popen(
            ["python", "-m", "quail.remote.server", "--model", model, "--device", "h100-sxm",
             "--listen", f"127.0.0.1:{port}", "--data", f"/tmp/quail-{model}"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
        endpoints[model] = f"grpc://127.0.0.1:{port}"
    for endpoint in endpoints.values():
        port = int(endpoint.rsplit(":", 1)[1])
        for _ in range(600):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
    return servers, endpoints


@app.function(
    image=image, gpu="H100", timeout=6 * 3600, memory=65_536,
    secrets=[modal.Secret.from_name("docetl-gemini")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/kernels": kernel_cache,
             "/results": results},
)
def experiment(models: list[str], hosted: list[str], agent: str, iterations: int,
               name: str, inputs: dict, reasoning_effort: str, workload: str) -> str:
    servers, endpoints = _serve(models)
    work = f"/results/{name}"
    try:
        args = ["/opt/dvenv/bin/python", "/opt/docetl/experiments/quail_moar/run_quail_moar.py",
                "--workload", workload, "--work-dir", work, "--agent", agent, "--iterations", str(iterations),
                "--hosted", *hosted, "--gpu-cost-per-hour", str(H100_PER_HOUR),
                "--quail", *(f"{m}={e}" for m, e in endpoints.items())]
        for flag, path in inputs.items():
            args += [f"--{flag}", f"/results/{path}"]
        if reasoning_effort:
            args += ["--reasoning-effort", reasoning_effort]
        done = subprocess.run(args, env={**os.environ, "PYTHONPATH": "/opt/docetl"})
        results.commit()
        summary = os.path.join(work, "summary.json")
        if done.returncode != 0 or not os.path.exists(summary):
            return f"run_quail_moar exited with {done.returncode}; see the logs and {work}"
        return open(summary).read()
    finally:
        for server in servers:
            server.terminate()


@app.local_entrypoint()
def main(models: str = "qwen3-4b-fp8",
         hosted: str = "gemini/gemini-4-hydrogen,gemini/gemini-4-argon",
         agent: str = "gemini/gemini-4-carbon", iterations: int = 10,
         pipeline: str = "", train: str = "", test: str = "", ground_truth: str = "",
         reasoning_effort: str = "high", name: str = "", workload: str = "medec"):
    chosen = [m for m in models.split(",") if m]
    name = name or f"{workload}-{time.strftime('%Y%m%d-%H%M%S')}"
    inputs = {}
    with results.batch_upload(force=True) as upload:
        for flag, path in (("pipeline", pipeline), ("train", train), ("test", test),
                           ("ground-truth", ground_truth)):
            if path:
                remote = f"{name}/inputs/{os.path.basename(path)}"
                upload.put_file(path, f"/{remote}")
                inputs[flag] = remote
    print(f"experiment {name}: {len(chosen)} H100s for {chosen}; inputs {inputs}")
    run = experiment.with_options(gpu=f"H100:{len(chosen)}") if len(chosen) > 1 else experiment
    out = run.remote(chosen, [h for h in hosted.split(",") if h], agent, iterations,
                     name, inputs, reasoning_effort, workload)
    try:
        print(json.dumps(json.loads(out), indent=2))
    except json.JSONDecodeError:
        print(out)
