"""Run the Quail MOAR experiment (MEDEC or CUAD) on Modal.

One container gets one H200 per open model. It starts a quail-server per GPU,
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

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time

import modal

from quail.deploy.images import gpu_image

DOCETL = os.environ.get(
    "DOCETL_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
BRANCH = os.environ.get("DOCETL_BRANCH", "claude/nifty-meitner-afmb36")
IGNORE = [".venv", ".git", "website", "node_modules", "docs", "**/__pycache__",
          "experiments/reasoning/data", "experiments/reasoning/outputs", "site"]
# Modal's H200 price per hour (https://modal.com/pricing), used to price GPU
# time. An H200 (141 GB) holds a 27B model with Quail's KV cache; an 80 GB
# H100 does not. (Modal may place a plain "H100" request on an H200.)
GPU = "H200"
H100_PER_HOUR = 4.54

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
         .add_local_dir(DOCETL, "/opt/docetl", ignore=IGNORE)
         # MOAR's agents read operator docs (read_operator_doc) from here.
         .add_local_dir(os.path.join(DOCETL, "docs", "operators"), "/opt/docetl/docs/operators"))


class Servers:
    """One quail-server per GPU, restarted when one exits.

    Each server's output goes to this container's log and, when ``log_dir``
    is given, to ``<log_dir>/<model>-gpu<n>.log``. A thread restarts a server
    that exits, so DocETL's retry finds it listening again.

    Args:
        models: The model of each GPU, in GPU order.
        log_dir: Where to write each server's log, or None.
    """

    def __init__(self, models: list[str], log_dir: str | None = None):
        self.models = models
        self.log_dir = log_dir
        self.procs: list[subprocess.Popen] = [None] * len(models)
        self.restarts = [0] * len(models)
        self._stopping = threading.Event()
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        for gpu in range(len(models)):
            self._start(gpu)
        threading.Thread(target=self._watch, daemon=True).start()

    def _start(self, gpu: int) -> None:
        model = self.models[gpu]
        proc = subprocess.Popen(
            ["python", "-m", "quail.remote.server", "--model", model, "--device", "h100-sxm",
             "--listen", f"127.0.0.1:{50051 + gpu}", "--data", f"/tmp/quail-{model}-{gpu}"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.procs[gpu] = proc
        path = os.path.join(self.log_dir, f"{model}-gpu{gpu}.log") if self.log_dir else None

        def copy() -> None:
            with open(path, "a") if path else contextlib.nullcontext() as log:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    if log:
                        log.write(line)
                        log.flush()

        threading.Thread(target=copy, daemon=True).start()

    def _watch(self) -> None:
        while not self._stopping.wait(5):
            for gpu, proc in enumerate(self.procs):
                if proc.poll() is not None and not self._stopping.is_set():
                    self.restarts[gpu] += 1
                    print(f"quail-server on GPU {gpu} ({self.models[gpu]}) exited with code "
                          f"{proc.returncode}; restart {self.restarts[gpu]}", flush=True)
                    self._start(gpu)

    def endpoints(self) -> dict:
        """Each model's endpoint; a model on several GPUs lists them all, comma-separated."""
        out = {}
        for gpu, model in enumerate(self.models):
            address = f"grpc://127.0.0.1:{50051 + gpu}"
            out[model] = f"{out[model]},{address}" if model in out else address
        return out

    def wait_until_listening(self) -> None:
        for gpu in range(len(self.models)):
            for _ in range(600):
                try:
                    socket.create_connection(("127.0.0.1", 50051 + gpu), timeout=1).close()
                    break
                except OSError:
                    time.sleep(1)

    def stop(self) -> None:
        self._stopping.set()
        for proc in self.procs:
            proc.terminate()


def _serve(models: list[str], log_dir: str | None = None) -> tuple[Servers, dict]:
    """Start one quail-server per GPU. A model listed twice gets two GPUs:
    its endpoint is then both servers, comma-separated, and DocETL sends each
    query to one of them."""
    servers = Servers(models, log_dir)
    servers.wait_until_listening()
    return servers, servers.endpoints()


@app.function(
    image=image, gpu=GPU, timeout=24 * 3600, memory=65_536,
    secrets=[modal.Secret.from_name("docetl-gemini")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/kernels": kernel_cache,
             "/results": results},
)
def experiment(models: list[str], hosted: list[str], agent: str, iterations: int,
               name: str, inputs: dict, reasoning_effort: str, workload: str,
               agent_reasoning_effort: str = "", concurrent_agents: int = 3,
               focus_quail: bool = False, reduce_model: str = "") -> str:
    work = f"/results/{name}"
    servers, endpoints = _serve(models, f"{work}/servers")
    try:
        args = ["/opt/dvenv/bin/python", "/opt/docetl/experiments/quail_moar/run_quail_moar.py",
                "--workload", workload, "--work-dir", work, "--agent", agent, "--iterations", str(iterations),
                "--hosted", *hosted, "--gpu-cost-per-hour", str(H100_PER_HOUR),
                "--quail", *(f"{m}={e}" for m, e in endpoints.items())]
        for flag, path in inputs.items():
            args += [f"--{flag}", f"/results/{path}"]
        if reasoning_effort:
            args += ["--reasoning-effort", reasoning_effort]
        if agent_reasoning_effort:
            args += ["--agent-reasoning-effort", agent_reasoning_effort]
        args += ["--concurrent-agents", str(concurrent_agents)]
        if focus_quail:
            args += ["--focus-quail"]
        if reduce_model:
            args += ["--reduce-model", reduce_model]
        done = subprocess.run(args, env={**os.environ, "PYTHONPATH": "/opt/docetl"})
        results.commit()
        summary = os.path.join(work, "summary.json")
        if done.returncode != 0 or not os.path.exists(summary):
            return f"run_quail_moar exited with {done.returncode}; see the logs and {work}"
        return open(summary).read()
    finally:
        servers.stop()
        results.commit()


@app.local_entrypoint()
def main(models: str = "qwen3-4b-fp8",
         hosted: str = "gemini/gemini-4-hydrogen,gemini/gemini-4-argon",
         agent: str = "gemini/gemini-4-carbon", iterations: int = 10,
         pipeline: str = "", train: str = "", test: str = "", ground_truth: str = "",
         reasoning_effort: str = "medium", agent_reasoning_effort: str = "medium",
         concurrent_agents: int = 6, name: str = "", workload: str = "medec",
         focus_quail: bool = False, reduce_model: str = ""):
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
    print(f"experiment {name}: {len(chosen)} {GPU}s for {chosen}; inputs {inputs}")
    run = experiment.with_options(gpu=f"{GPU}:{len(chosen)}") if len(chosen) > 1 else experiment
    out = run.remote(chosen, [h for h in hosted.split(",") if h], agent, iterations,
                     name, inputs, reasoning_effort, workload,
                     agent_reasoning_effort, concurrent_agents, focus_quail,
                     reduce_model)
    try:
        print(json.dumps(json.loads(out), indent=2))
    except json.JSONDecodeError:
        print(out)
