"""
Simplified Python API for MOAR optimization.

Usage with a YAML file::

    from docetl.moar import MOAROptimizer

    optimizer = MOAROptimizer(
        pipeline="pipeline.yaml",
        eval_fn=lambda results_path: {"score": compute_score(results_path)},
        metric_key="score",
    )
    results = optimizer.optimize()
    print(results.best())

Usage with the Pipeline Python API::

    from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput

    pipeline = Pipeline(
        name="my_pipeline",
        datasets={"input": Dataset(type="file", path="data.json")},
        operations=[MapOp(name="extract", type="map", prompt="...")],
        steps=[PipelineStep(name="step1", input="input", operations=["extract"])],
        output=PipelineOutput(type="file", path="output.json"),
    )

    optimizer = MOAROptimizer(
        pipeline=pipeline,
        eval_fn=lambda results_path: {"score": compute_score(results_path)},
        metric_key="score",
    )
    results = optimizer.optimize()
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import yaml

from docetl.console import DOCETL_CONSOLE
from docetl.moar.models import default_agent_model, detect_available_models
from docetl.reasoning_optimizer.directives import ALL_DIRECTIVES
from docetl.utils_dataset import get_dataset_stats

if TYPE_CHECKING:
    from docetl.api import Pipeline


@dataclass
class FrontierPoint:
    """A single point on the Pareto frontier."""

    yaml_path: str
    cost: float
    accuracy: float
    node_id: int
    on_frontier: bool = True

    def __repr__(self) -> str:
        return (
            f"FrontierPoint(cost={self.cost:.4f}, accuracy={self.accuracy:.4f}, "
            f"yaml={self.yaml_path!r})"
        )


@dataclass
class MOARResult:
    """Results returned by MOAROptimizer.optimize()."""

    frontier: List[FrontierPoint] = field(default_factory=list)
    all_plans: List[FrontierPoint] = field(default_factory=list)
    total_search_cost: float = 0.0
    iterations: int = 0
    duration_seconds: float = 0.0
    save_dir: Optional[str] = None

    def best(self) -> Optional[FrontierPoint]:
        """Return the highest-accuracy point on the frontier."""
        frontier = [p for p in self.frontier if p.on_frontier]
        if not frontier:
            return None
        return max(frontier, key=lambda p: p.accuracy)

    def cheapest(self) -> Optional[FrontierPoint]:
        """Return the lowest-cost point on the frontier."""
        frontier = [p for p in self.frontier if p.on_frontier]
        if not frontier:
            return None
        return min(frontier, key=lambda p: p.cost)

    def summary(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            f"MOAR Optimization Results ({self.iterations} iterations, "
            f"{self.duration_seconds:.1f}s, ${self.total_search_cost:.2f} search cost)",
            f"{'─' * 70}",
            f"  {'#':<4} {'Cost':>10} {'Accuracy':>10} {'Frontier':>10}   YAML Path",
            f"  {'─'*4} {'─'*10} {'─'*10} {'─'*10}   {'─'*30}",
        ]
        for i, p in enumerate(
            sorted(self.all_plans, key=lambda x: x.accuracy, reverse=True), 1
        ):
            marker = "  *" if p.on_frontier else ""
            lines.append(
                f"  {i:<4} ${p.cost:>9.4f} {p.accuracy:>10.4f} {marker:>10}   {p.yaml_path}"
            )
        lines.append(f"\n  Frontier points marked with *")
        if self.save_dir:
            lines.append(f"  Full results saved to: {self.save_dir}")
        return "\n".join(lines)


class MOAROptimizer:
    """
    Simplified interface for MOAR (Multi-Objective Agentic Rewrites) optimization.

    Wraps ``MOARSearch`` with sensible defaults and auto-detection of available
    models from environment API keys.

    Args:
        pipeline: A ``Pipeline`` object, a path to a YAML pipeline file (str or
            Path), or a raw config dict. When a ``Pipeline`` is given, it is
            serialized to a temporary YAML file automatically.
        eval_fn: Evaluation function with signature
            ``(results_file_path: str) -> dict[str, Any]``.
            Alternatively, a path to a Python file containing a function
            decorated with ``@docetl.register_eval``.
        metric_key: Key to extract from the eval function's return dict.
        models: List of model names to search over. If ``None``, auto-detects
            from environment API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY,
            GEMINI_API_KEY, AZURE_API_KEY).
        agent_model: Model for the MOAR rewrite agent LLM calls. If ``None``,
            picks the best available model from *models*.
        max_iterations: Maximum MCTS iterations (default 20).
        save_dir: Directory for output files. If ``None``, uses a temp directory.
        exploration_weight: UCB exploration constant (default sqrt(2)).
        dataset_path: Explicit path to the sample dataset JSON. If ``None``,
            inferred from the pipeline's ``datasets`` section.
        dataset_name: Name of the dataset. If ``None``, inferred from the
            pipeline's ``datasets`` section.

    Examples::

        # From a YAML file
        optimizer = MOAROptimizer(
            pipeline="pipeline.yaml",
            eval_fn=lambda p: {"score": my_score(p)},
            metric_key="score",
        )
        results = optimizer.optimize()

        # From a Pipeline object
        from docetl.api import Pipeline, Dataset, MapOp, PipelineStep, PipelineOutput

        pipeline = Pipeline(
            name="my_pipeline",
            datasets={"input": Dataset(type="file", path="data.json")},
            operations=[MapOp(name="extract", type="map", prompt="Extract entities")],
            steps=[PipelineStep(name="step1", input="input", operations=["extract"])],
            output=PipelineOutput(type="file", path="output.json"),
        )
        optimizer = MOAROptimizer(
            pipeline=pipeline,
            eval_fn=lambda p: {"score": my_score(p)},
            metric_key="score",
        )
        results = optimizer.optimize()
        print(results.best())
    """

    def __init__(
        self,
        pipeline: Union["Pipeline", str, Path, Dict[str, Any]],
        eval_fn: Union[Callable[[str], Dict[str, Any]], str, Path],
        metric_key: str,
        models: Optional[List[str]] = None,
        agent_model: Optional[str] = None,
        max_iterations: int = 20,
        save_dir: Optional[Union[str, Path]] = None,
        exploration_weight: float = 1.414,
        dataset_path: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
    ):
        self.pipeline_path = self._resolve_pipeline(pipeline, save_dir)
        self.metric_key = metric_key
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight

        # Auto-detect models if not provided
        if models is not None:
            self.models = list(models)
        else:
            self.models = detect_available_models()

        # Auto-detect agent model if not provided
        self.agent_model = agent_model or default_agent_model(self.models)

        # Resolve save directory
        if save_dir is not None:
            self._save_dir = Path(save_dir).resolve()
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="moar_")
            self._save_dir = Path(self._temp_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # Resolve eval function
        self._eval_fn = self._resolve_eval_fn(eval_fn, dataset_path)

        # Resolve dataset info
        self._dataset_path, self._dataset_name = self._resolve_dataset(
            dataset_path, dataset_name
        )

    def _resolve_pipeline(
        self,
        pipeline: Union["Pipeline", str, Path, Dict[str, Any]],
        save_dir: Optional[Union[str, Path]],
    ) -> str:
        """Convert pipeline input to a resolved YAML file path."""
        from docetl.api import Pipeline as PipelineClass

        if isinstance(pipeline, PipelineClass):
            target_dir = Path(save_dir).resolve() if save_dir else Path(tempfile.mkdtemp(prefix="moar_pipeline_"))
            target_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = target_dir / f"{pipeline.name}.yaml"
            pipeline.to_yaml(str(yaml_path))
            return str(yaml_path)

        if isinstance(pipeline, dict):
            target_dir = Path(save_dir).resolve() if save_dir else Path(tempfile.mkdtemp(prefix="moar_pipeline_"))
            target_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = target_dir / "pipeline.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
            return str(yaml_path)

        return str(Path(pipeline).resolve())

    def _resolve_eval_fn(
        self,
        eval_fn: Union[Callable, str, Path],
        dataset_path: Optional[Union[str, Path]],
    ) -> Callable[[str], Dict[str, Any]]:
        """Accept a callable directly or load from a file path."""
        if callable(eval_fn):
            return eval_fn

        from docetl.utils_evaluation import load_custom_evaluate_func

        eval_file = str(Path(eval_fn).resolve())
        ds_path = str(Path(dataset_path).resolve()) if dataset_path else ""
        if not ds_path:
            ds_path, _ = self._infer_dataset_info()
        return load_custom_evaluate_func(eval_file, ds_path)

    def _infer_dataset_info(self) -> tuple[str, str]:
        """Infer dataset path and name from the pipeline YAML."""
        with open(self.pipeline_path, "r") as f:
            config = yaml.safe_load(f)

        datasets = config.get("datasets", {})
        if not datasets:
            raise ValueError("Pipeline YAML must contain a 'datasets' section")

        ds_name, ds_config = next(iter(datasets.items()))
        ds_path = ds_config.get("path", "")
        if not ds_path:
            raise ValueError(f"Dataset '{ds_name}' must have a 'path' field")

        p = Path(ds_path)
        if not p.is_absolute():
            yaml_dir = Path(self.pipeline_path).parent
            candidate = yaml_dir / ds_path
            if candidate.exists():
                p = candidate
            elif Path(ds_path).exists():
                p = Path(ds_path)
            else:
                p = candidate
        return str(p.resolve()), ds_name

    def _resolve_dataset(
        self,
        dataset_path: Optional[Union[str, Path]],
        dataset_name: Optional[str],
    ) -> tuple[str, str]:
        """Resolve dataset path and name, using explicit values or inference."""
        inferred_path, inferred_name = self._infer_dataset_info()

        if dataset_path is not None:
            resolved_path = str(Path(dataset_path).resolve())
        else:
            resolved_path = inferred_path

        resolved_name = dataset_name if dataset_name is not None else inferred_name
        return resolved_path, resolved_name

    def optimize(self) -> MOARResult:
        """
        Run full MOAR MCTS optimization.

        Returns:
            MOARResult with the Pareto frontier and all explored plans.
        """
        from docetl.moar import MOARSearch

        DOCETL_CONSOLE.log("[bold blue]MOAROptimizer: loading dataset...[/bold blue]")
        with open(self._dataset_path, "r") as f:
            dataset_data = json.load(f)
        sample_input = dataset_data[:5] if isinstance(dataset_data, list) else dataset_data

        dataset_stats = get_dataset_stats(self.pipeline_path, self._dataset_name)
        available_actions = set(ALL_DIRECTIVES)

        DOCETL_CONSOLE.log(
            f"[bold blue]MOAROptimizer: starting search "
            f"({self.max_iterations} iterations, {len(self.models)} models)[/bold blue]"
        )
        DOCETL_CONSOLE.log(f"[dim]Models: {', '.join(self.models)}[/dim]")
        DOCETL_CONSOLE.log(f"[dim]Agent model: {self.agent_model}[/dim]")
        DOCETL_CONSOLE.log(f"[dim]Save dir: {self._save_dir}[/dim]")

        start_time = datetime.now()

        moar = MOARSearch(
            root_yaml_path=self.pipeline_path,
            available_actions=available_actions,
            sample_input=sample_input,
            dataset_stats=dataset_stats,
            dataset_name=self._dataset_name,
            available_models=self.models,
            evaluate_func=self._eval_fn,
            exploration_constant=self.exploration_weight,
            max_iterations=self.max_iterations,
            model=self.agent_model,
            output_dir=str(self._save_dir),
            build_first_layer=False,
            custom_metric_key=self.metric_key,
            sample_dataset_path=self._dataset_path,
        )

        moar.search()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = self._build_result(moar, duration)

        self._save_results(result, moar, start_time, end_time, duration)

        DOCETL_CONSOLE.log(
            f"[green]MOAROptimizer: completed in {duration:.1f}s, "
            f"{len(result.frontier)} frontier points[/green]"
        )
        return result

    def _build_result(self, moar: Any, duration: float) -> MOARResult:
        """Extract results from MOARSearch into a MOARResult."""
        frontier_set = set(moar.pareto_frontier.frontier_plans)

        all_plans = []
        frontier = []
        for node in moar.pareto_frontier.plans:
            accuracy = moar.pareto_frontier.plans_accuracy.get(node, 0.0)
            on_frontier = node in frontier_set
            point = FrontierPoint(
                yaml_path=node.yaml_file_path,
                cost=node.cost,
                accuracy=accuracy,
                node_id=node.get_id(),
                on_frontier=on_frontier,
            )
            all_plans.append(point)
            if on_frontier:
                frontier.append(point)

        return MOARResult(
            frontier=frontier,
            all_plans=all_plans,
            total_search_cost=getattr(moar, "total_search_cost", 0.0),
            iterations=moar.iteration_count,
            duration_seconds=duration,
            save_dir=str(self._save_dir),
        )

    def _save_results(
        self,
        result: MOARResult,
        moar: Any,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> None:
        """Save experiment summary and Pareto frontier to disk."""
        summary = {
            "optimizer": "moar",
            "input_pipeline": self.pipeline_path,
            "agent_model": self.agent_model,
            "models": self.models,
            "max_iterations": self.max_iterations,
            "exploration_weight": self.exploration_weight,
            "metric_key": self.metric_key,
            "save_dir": str(self._save_dir),
            "dataset": self._dataset_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_search_cost": result.total_search_cost,
            "num_frontier_points": len(result.frontier),
            "num_plans_explored": len(result.all_plans),
        }

        summary_file = self._save_dir / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        if result.frontier:
            pareto_file = self._save_dir / "pareto_frontier.json"
            pareto_data = [
                {
                    "node_id": p.node_id,
                    "yaml_path": p.yaml_path,
                    "cost": p.cost,
                    "accuracy": p.accuracy,
                }
                for p in result.frontier
            ]
            with open(pareto_file, "w") as f:
                json.dump(pareto_data, f, indent=2)
