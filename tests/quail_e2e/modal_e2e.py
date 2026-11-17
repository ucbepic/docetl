"""End-to-end: DocETL's Quail engine against quail-server on a Modal H100.

Runs three pipelines inside one Modal GPU container, next to quail-server
(gRPC to localhost, since some proxies cannot carry gRPC to Modal):

1. a filter and a four-item map on 12 movie reviews,
2. the same pipeline again, which reads Quail Server's saved rows,
3. a map and a reduce per movie, fused into one query.

Run it from a Quail checkout (https://github.com/fsdatalab/quail), whose
``quail.deploy.images`` builds the GPU image, with Modal credentials:

    cd /path/to/quail
    DOCETL_REPO=/path/to/docetl modal run /path/to/docetl/tests/quail_e2e/modal_e2e.py
"""
import json
import os
import socket
import subprocess
import time

import modal

from quail.deploy.images import gpu_image

DOCETL = os.environ.get("DOCETL_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
IGNORE = [".venv", ".git", "website", "node_modules", "experiments", "docs", "**/__pycache__", "tests"]

app = modal.App("docetl-quail-e2e")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
image = (gpu_image(commands=("pip install uv",
                             "uv venv /opt/dvenv --python 3.12",
                             "VIRTUAL_ENV=/opt/dvenv uv pip install docetl pyarrow"))
         .add_local_dir(DOCETL, "/opt/docetl", ignore=IGNORE))

REVIEWS = [
    "The ending was perfect and the cast was great.",
    "Too long, and the ending made no sense at all.",
    "Great music, but I want my money back.",
    "The final twist in the last scene floored me. Five stars.",
    "Boring first hour. The soundtrack was the only highlight.",
    "I hated how it ended; the villain just walked away.",
    "Beautiful cinematography and a satisfying conclusion.",
    "The popcorn was stale and the seats were uncomfortable.",
    "Loved the ending, cried like a baby.",
    "Mediocre acting, forgettable plot, no idea how it ended because I left.",
    "The ending felt rushed but the rest was fantastic.",
    "Fun for kids, nothing special for adults.",
]

PIPELINE = r'''
import json, sys, tempfile, os
from docetl.runner import DSLRunner
d = tempfile.mkdtemp()
inp, out = os.path.join(d, "in.json"), os.path.join(d, "out.json")
json.dump([{"id": i, "movie": ["Ocean Drift", "Night Train", "Paper Moon"][i % 3], "body": b} for i, b in enumerate(json.loads(sys.argv[2]))], open(inp, "w"))
config = {
  "default_model": "gpt-4o-mini",
  "engine": {"type": "quail", "endpoint": sys.argv[1], "fallback": "strict"},
  "datasets": {"reviews": {"type": "file", "path": inp}},
  "operations": [
    {"name": "ending", "type": "filter", "prompt": "Does this movie review talk about the ending of the movie? {{ input.body }}", "output": {"schema": {"discusses_ending": "bool"}}},
    {"name": "label", "type": "map", "prompt": "Read this movie review: {{ input.body }}", "output": {"schema": {"sentiment": "enum[positive, negative, mixed]", "liked_ending": "bool", "summary": "string", "stars": "int"}}, "litellm_completion_kwargs": {"max_tokens": 48}},
  ],
  "operations_reduce": [
    {"name": "per_movie", "type": "reduce", "reduce_key": "movie", "prompt": "These are reviews of one movie. Say in one sentence what reviewers think of it.{% for r in inputs %}\n- {{ r.body }} ({{ r.sentiment }}){% endfor %}", "output": {"schema": {"consensus": "string", "overall": "enum[positive, negative, mixed]"}}, "litellm_completion_kwargs": {"max_tokens": 64}},
  ],
  "pipeline": {"steps": [{"name": "s", "input": "reviews", "operations": ["ending", "label"]}], "output": {"type": "file", "path": out}},
}
if len(sys.argv) > 3 and sys.argv[3] == "reduce":
    config["operations"] = [config["operations"][1]] + config.pop("operations_reduce")
    config["pipeline"]["steps"][0]["operations"] = ["label", "per_movie"]
else:
    config.pop("operations_reduce")
t = __import__("time").time()
DSLRunner(config, max_threads=4).load_run_save()
print("ELAPSED", round(__import__("time").time() - t, 1))
print("RESULT", json.dumps(json.load(open(out))))
'''


@app.function(image=image, gpu="H100!", timeout=3_600, memory=65_536,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/kernels": kernel_cache})
def e2e() -> str:
    server = subprocess.Popen(["python", "-m", "quail.remote.server", "--model", "qwen3-4b-fp8",
                               "--device", "h100-sxm", "--listen", "127.0.0.1:50051",
                               "--data", "/tmp/qdata"])
    for _ in range(600):
        try:
            socket.create_connection(("127.0.0.1", 50051), timeout=1).close()
            break
        except OSError:
            time.sleep(1)
    open("/tmp/pipeline.py", "w").write(PIPELINE)
    check = subprocess.run(["/opt/dvenv/bin/python", "-c", "import docetl.engines.quail, docetl; print(docetl.__file__)"],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "/opt/docetl"})
    print("DOCETL", check.stdout, check.stderr[-2000:], flush=True)
    runs = []
    for mode in ("filter_map", "filter_map", "reduce"):  # the 2nd run should read saved rows
        done = subprocess.run(["/opt/dvenv/bin/python", "/tmp/pipeline.py", "grpc://127.0.0.1:50051",
                               json.dumps(REVIEWS), mode], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": "/opt/docetl"})
        runs.append(done.stdout[-6000:] + done.stderr[-3000:])
    server.terminate()
    return "\n=====NEXT RUN=====\n".join(runs)


@app.local_entrypoint()
def main():
    print(e2e.remote())
