"""Run the Quail MOAR experiment on MEDEC on Modal.

One container gets one H100 per open model. It starts a quail-server per GPU,
then runs ``run_medec.run`` with Gemini (the ``docetl-gemini`` Modal secret,
holding GEMINI_API_KEY) as MOAR's agent and as the hosted models. Results go
to the ``docetl-quail-moar`` Modal volume.

Run it from a Quail checkout (https://github.com/fsdatalab/quail), whose
``quail.deploy.images`` builds the GPU image:

    cd /path/to/quail
    DOCETL_REPO=/path/to/docetl modal run /path/to/docetl/experiments/quail_moar/modal_medec.py \\
        --models qwen3-4b-fp8,qwen3.5-4b-fp8,lfm2-2.6b-bf16 --iterations 10
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
MAX_GPUS = 8

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
        endpoints[model] = {"endpoint": f"grpc://127.0.0.1:{port}", "gpu_cost_per_hour": H100_PER_HOUR}
    for model, served in endpoints.items():
        port = int(served["endpoint"].rsplit(":", 1)[1])
        for _ in range(600):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
    return servers, endpoints


def _experiment(models, hosted, agent, iterations, n_train, n_test, name) -> str:
    servers, endpoints = _serve(models)
    work = f"/results/{name}"
    try:
        args = ["/opt/dvenv/bin/python", "/opt/docetl/experiments/quail_moar/run_medec.py",
                "--work-dir", work, "--agent", agent, "--iterations", str(iterations),
                "--n-train", str(n_train), "--n-test", str(n_test), "--hosted", *hosted,
                "--gpu-cost-per-hour", str(H100_PER_HOUR),
                "--quail", *(f"{m}={e['endpoint']}" for m, e in endpoints.items())]
        done = subprocess.run(args, env={**os.environ, "PYTHONPATH": "/opt/docetl"})
        results.commit()
        summary = os.path.join(work, "summary.json")
        if done.returncode != 0 or not os.path.exists(summary):
            return f"run_medec exited with {done.returncode}; see the logs and {work}"
        return open(summary).read()
    finally:
        for server in servers:
            server.terminate()


def _function(gpus: int):
    return app.function(
        image=image, gpu=f"H100:{gpus}", timeout=6 * 3600, memory=65_536 * gpus,
        secrets=[modal.Secret.from_name("docetl-gemini")],
        volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/kernels": kernel_cache,
                 "/results": results},
        name=f"experiment_{gpus}gpu", serialized=True,
    )(_experiment)


EXPERIMENTS = {gpus: _function(gpus) for gpus in range(1, MAX_GPUS + 1)}


@app.local_entrypoint()
def main(models: str = "qwen3-4b-fp8,qwen3.5-4b-fp8,lfm2-2.6b-bf16",
         hosted: str = "gemini/gemini-2.5-flash-lite,gemini/gemini-2.5-flash",
         agent: str = "gemini/gemini-2.5-pro", iterations: int = 10,
         n_train: int = 100, n_test: int = 200, name: str = ""):
    chosen = [m for m in models.split(",") if m]
    name = name or f"medec-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"experiment {name}: {len(chosen)} H100s for {chosen}")
    out = EXPERIMENTS[len(chosen)].remote(
        chosen, [h for h in hosted.split(",") if h], agent, iterations, n_train, n_test, name)
    try:
        print(json.dumps(json.loads(out), indent=2))
    except json.JSONDecodeError:
        print(out)
