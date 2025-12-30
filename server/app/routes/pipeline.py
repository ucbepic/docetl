from typing import Any
import uuid
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from docetl.runner import DSLRunner
from docetl.optimizers.fast_should_optimize import FastShouldOptimizeAnalyzer
from docetl.optimizers.fast_decomposer import FastDecomposer
import asyncio
from asyncio import Task
from rich.logging import RichHandler
import logging
import yaml
from datetime import datetime, timedelta
from enum import Enum
from server.app.models import (
    OptimizeResult,
    TaskStatus,
    OptimizeRequest,
    PipelineRequest,
    DecomposeRequest,
    DecomposeResult,
)

# Setup logging
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

router = APIRouter()

# Task storage for optimize tasks
tasks: dict[str, OptimizeResult] = {}
asyncio_tasks: dict[str, Task] = {}

# Task storage for decompose tasks
decompose_tasks: dict[str, DecomposeResult] = {}
decompose_asyncio_tasks: dict[str, Task] = {}

# Configuration
COMPLETED_TASK_TTL = timedelta(hours=1)

async def cleanup_old_tasks():
    """Background task to clean up completed tasks"""
    while True:
        try:
            current_time = datetime.now()

            # Clean up optimize tasks
            task_ids_to_remove = []
            for task_id, task in tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and
                    current_time - task.completed_at > COMPLETED_TASK_TTL):
                    task_ids_to_remove.append(task_id)
            for task_id in task_ids_to_remove:
                del tasks[task_id]

            # Clean up decompose tasks
            decompose_ids_to_remove = []
            for task_id, task in decompose_tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and
                    current_time - task.completed_at > COMPLETED_TASK_TTL):
                    decompose_ids_to_remove.append(task_id)
            for task_id in decompose_ids_to_remove:
                del decompose_tasks[task_id]

            await asyncio.sleep(60)

        except Exception as e:
            logging.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

async def run_optimization(task_id: str, yaml_config: str, step_name: str, op_name: str):
    """Execute the optimization task using fast single-LLM-call analysis."""
    try:
        tasks[task_id].status = TaskStatus.PROCESSING

        # yaml_config is a file path, not YAML content - read and parse the file
        if yaml_config.endswith(".yaml") or yaml_config.endswith(".yml"):
            with open(yaml_config, "r") as f:
                config = yaml.safe_load(f)
        else:
            # Fallback: try parsing as YAML string
            config = yaml.safe_load(yaml_config)

        # Validate that we got a dict
        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid yaml_config: expected dict after parsing, got {type(config).__name__}. "
                f"Input: {str(yaml_config)[:200]}"
            )

        # Get intermediate directory from config
        intermediate_dir = (
            config.get("pipeline", {})
            .get("output", {})
            .get("intermediate_dir")
        )

        if not intermediate_dir:
            raise ValueError("No intermediate_dir configured in pipeline output")

        # Find the operation config
        op_config = None
        for op in config.get("operations", []):
            if op.get("name") == op_name:
                op_config = op
                break

        if not op_config:
            raise ValueError(f"Operation '{op_name}' not found in config")

        # Get optimizer settings from config
        optimizer_model = (
            config.get("optimizer_config", {})
            .get("rewrite_agent_model", "gpt-5.1")
        )
        litellm_kwargs = (
            config.get("optimizer_config", {})
            .get("litellm_kwargs", {})
        )

        # Create analyzer and run in thread pool
        analyzer = FastShouldOptimizeAnalyzer(
            intermediate_dir=intermediate_dir,
            optimizer_model=optimizer_model,
            litellm_kwargs=litellm_kwargs,
        )

        should_optimize, output_data, num_docs_analyzed, cost = await asyncio.to_thread(
            analyzer.analyze,
            op_config,
            step_name,
            op_name,
        )

        # Update task result
        tasks[task_id].status = TaskStatus.COMPLETED
        tasks[task_id].should_optimize = should_optimize
        tasks[task_id].input_data = None  # We don't have input_data in fast mode
        tasks[task_id].output_data = output_data
        tasks[task_id].num_docs_analyzed = num_docs_analyzed
        tasks[task_id].cost = cost
        tasks[task_id].completed_at = datetime.now()

    except asyncio.CancelledError:
        tasks[task_id].status = TaskStatus.CANCELLED
        tasks[task_id].completed_at = datetime.now()
        raise

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        tasks[task_id].status = TaskStatus.FAILED
        tasks[task_id].error = f"{str(e)}\n{error_traceback}"
        tasks[task_id].completed_at = datetime.now()
        raise

    finally:
        if task_id in asyncio_tasks:
            del asyncio_tasks[task_id]

@router.on_event("startup")
async def startup_event():
    """Start the cleanup task when the application starts"""
    asyncio.create_task(cleanup_old_tasks())

@router.post("/should_optimize", status_code=202)
async def submit_optimize_task(request: OptimizeRequest):
    """Submit a new optimization task"""
    task_id = str(uuid.uuid4())
    
    # Create task record
    tasks[task_id] = OptimizeResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now()
    )
    
    # Create and store the asyncio task
    task = asyncio.create_task(
        run_optimization(
            task_id,
            request.yaml_config,
            request.step_name,
            request.op_name
        )
    )
    asyncio_tasks[task_id] = task
    
    return {"task_id": task_id}

@router.get("/should_optimize/{task_id}")
async def get_optimize_status(task_id: str) -> OptimizeResult:
    """Get the current status of an optimization task"""
    if task_id not in tasks:
        raise HTTPException(
            status_code=404, 
            detail="Task not found or has been cleaned up"
        )
    
    return tasks[task_id]

@router.post("/should_optimize/{task_id}/cancel")
async def cancel_optimize_task(task_id: str):
    """Cancel a running optimization task"""
    if task_id not in tasks:
        raise HTTPException(
            status_code=404, 
            detail="Task not found or has been cleaned up"
        )
    
    if task_id not in asyncio_tasks:
        raise HTTPException(
            status_code=400, 
            detail="Task already finished or cannot be cancelled"
        )
    
    asyncio_task = asyncio_tasks[task_id]
    asyncio_task.cancel()
    
    try:
        await asyncio_task
    except asyncio.CancelledError:
        pass
    
    return {"message": "Task cancelled successfully"}


# ============================================================================
# Decompose endpoints
# ============================================================================

async def run_decomposition(task_id: str, yaml_config: str, step_name: str, op_name: str):
    """Execute the decomposition task using fast directive-based approach."""
    try:
        decompose_tasks[task_id].status = TaskStatus.PROCESSING

        # yaml_config is a file path
        if not (yaml_config.endswith(".yaml") or yaml_config.endswith(".yml")):
            raise ValueError("yaml_config must be a path to a YAML file")

        # Get optimizer settings from config
        with open(yaml_config, "r") as f:
            config = yaml.safe_load(f)

        optimizer_model = (
            config.get("optimizer_config", {})
            .get("rewrite_agent_model", "gpt-5.1")
        )
        litellm_kwargs = (
            config.get("optimizer_config", {})
            .get("litellm_kwargs", {})
        )

        # Create decomposer and run in thread pool
        decomposer = FastDecomposer(
            yaml_config_path=yaml_config,
            optimizer_model=optimizer_model,
            sample_size=5,
            litellm_kwargs=litellm_kwargs,
        )

        result = await asyncio.to_thread(
            decomposer.decompose,
            step_name,
            op_name,
        )

        # Update task result
        decompose_tasks[task_id].status = TaskStatus.COMPLETED
        decompose_tasks[task_id].decomposed_operations = result["decomposed_ops"]
        decompose_tasks[task_id].winning_directive = result["winning_directive"]
        decompose_tasks[task_id].candidates_evaluated = result["candidates_evaluated"]
        decompose_tasks[task_id].original_outputs = result["original_outputs"]
        decompose_tasks[task_id].decomposed_outputs = result["decomposed_outputs"]
        decompose_tasks[task_id].comparison_rationale = result["comparison_rationale"]
        decompose_tasks[task_id].cost = result["cost"]
        decompose_tasks[task_id].completed_at = datetime.now()

    except asyncio.CancelledError:
        decompose_tasks[task_id].status = TaskStatus.CANCELLED
        decompose_tasks[task_id].completed_at = datetime.now()
        raise

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        decompose_tasks[task_id].status = TaskStatus.FAILED
        decompose_tasks[task_id].error = f"{str(e)}\n{error_traceback}"
        decompose_tasks[task_id].completed_at = datetime.now()
        raise

    finally:
        if task_id in decompose_asyncio_tasks:
            del decompose_asyncio_tasks[task_id]


@router.post("/decompose", status_code=202)
async def submit_decompose_task(request: DecomposeRequest):
    """Submit a new decomposition task"""
    task_id = str(uuid.uuid4())

    # Create task record
    decompose_tasks[task_id] = DecomposeResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now()
    )

    # Create and store the asyncio task
    task = asyncio.create_task(
        run_decomposition(
            task_id,
            request.yaml_config,
            request.step_name,
            request.op_name
        )
    )
    decompose_asyncio_tasks[task_id] = task

    return {"task_id": task_id}


@router.get("/decompose/{task_id}")
async def get_decompose_status(task_id: str) -> DecomposeResult:
    """Get the current status of a decomposition task"""
    if task_id not in decompose_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found or has been cleaned up"
        )

    return decompose_tasks[task_id]


@router.post("/decompose/{task_id}/cancel")
async def cancel_decompose_task(task_id: str):
    """Cancel a running decomposition task"""
    if task_id not in decompose_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found or has been cleaned up"
        )

    if task_id not in decompose_asyncio_tasks:
        raise HTTPException(
            status_code=400,
            detail="Task already finished or cannot be cancelled"
        )

    asyncio_task = decompose_asyncio_tasks[task_id]
    asyncio_task.cancel()

    try:
        await asyncio_task
    except asyncio.CancelledError:
        pass

    return {"message": "Decomposition task cancelled successfully"}


# Keep the original run_pipeline endpoint
@router.post("/run_pipeline")
def run_pipeline(request: PipelineRequest) -> dict[str, Any]:
    try:
        runner = DSLRunner.from_yaml(request.yaml_config)
        cost = runner.load_run_save()
        runner.reset_env()
        return {"cost": cost, "message": "Pipeline executed successfully"}
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error occurred:\n{e}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=str(e) + "\n" + error_traceback)

@router.websocket("/ws/run_pipeline/{client_id}")
async def websocket_run_pipeline(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for running pipelines with real-time output streaming.

    Note: The old 'optimize' flag for full pipeline optimization has been removed.
    Use the /decompose endpoint for fast operation decomposition instead.
    """
    await websocket.accept()
    runner = None
    try:
        config = await websocket.receive_json()
        runner = DSLRunner.from_yaml(config["yaml_config"])

        if config.get("clear_intermediate", False):
            runner.clear_intermediate()

        async def run_pipeline():
            return await asyncio.to_thread(runner.load_run_save)

        pipeline_task = asyncio.create_task(run_pipeline())

        while not pipeline_task.done():
            console_output = runner.console.file.getvalue()
            await websocket.send_json({"type": "output", "data": console_output})

            # Check for incoming messages from the user
            try:
                user_message = await asyncio.wait_for(
                    websocket.receive_json(), timeout=0.1
                )

                if user_message == "kill":
                    runner.console.log("Stopping process...")
                    runner.is_cancelled = True

                    await websocket.send_json({
                        "type": "error",
                        "message": "Process stopped by user request"
                    })
                    raise Exception("Process stopped by user request")

                # Process the user message and send it to the runner
                runner.console.post_input(user_message)
            except asyncio.TimeoutError:
                pass  # No message received, continue with the loop
            except asyncio.CancelledError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Process stopped by user request"
                })
                raise

            await asyncio.sleep(0.5)

        # Final check to send any remaining output
        result = await pipeline_task

        console_output = runner.console.file.getvalue()
        if console_output:
            await websocket.send_json({"type": "output", "data": console_output})

        # Sleep for a short duration to ensure all output is captured
        await asyncio.sleep(3)

        await websocket.send_json(
            {
                "type": "result",
                "data": {
                    "message": "Pipeline executed successfully",
                    "cost": result,
                    "yaml_config": config["yaml_config"],
                },
            }
        )
    except WebSocketDisconnect:
        if runner is not None:
            runner.reset_env()
        print("Client disconnected")
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        print(f"Error occurred:\n{error_traceback}")
        await websocket.send_json({"type": "error", "data": str(e), "traceback": error_traceback})
    finally:
        if runner is not None:
            runner.reset_env()
        await websocket.close()


@router.websocket("/ws/decompose/{client_id}")
async def websocket_decompose(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for fast operation decomposition with real-time output streaming.

    Expects JSON message with:
    - yaml_config: Path to the pipeline YAML config file
    - step_name: Name of the pipeline step
    - op_name: Name of the operation to decompose
    """
    await websocket.accept()
    decomposer = None

    try:
        config = await websocket.receive_json()
        yaml_config = config["yaml_config"]
        step_name = config["step_name"]
        op_name = config["op_name"]

        # Validate yaml_config is a path
        if not (yaml_config.endswith(".yaml") or yaml_config.endswith(".yml")):
            raise ValueError("yaml_config must be a path to a YAML file")

        # Get optimizer settings from config
        with open(yaml_config, "r") as f:
            pipeline_config = yaml.safe_load(f)

        optimizer_model = (
            pipeline_config.get("optimizer_config", {})
            .get("rewrite_agent_model", "gpt-5.1")
        )
        litellm_kwargs = (
            pipeline_config.get("optimizer_config", {})
            .get("litellm_kwargs", {})
        )

        # Create a ThreadSafeConsole for streaming output
        from docetl.console import ThreadSafeConsole
        console = ThreadSafeConsole(
            force_terminal=True,
            soft_wrap=True,
            highlight=False,
            log_path=False,
            color_system="truecolor",
            width=120,
            style="bright_white on black",
            record=True,
        )

        # Create decomposer with the console
        decomposer = FastDecomposer(
            yaml_config_path=yaml_config,
            optimizer_model=optimizer_model,
            sample_size=5,
            litellm_kwargs=litellm_kwargs,
            console=console,
        )

        # Run decomposition in a thread
        async def run_decomposition():
            return await asyncio.to_thread(
                decomposer.decompose,
                step_name,
                op_name,
            )

        decompose_task = asyncio.create_task(run_decomposition())

        # Stream console output while decomposition runs
        accumulated_output = ""
        while not decompose_task.done():
            # get_output() processes carriage returns and clears the buffer
            new_output = console.get_output()
            if new_output:
                # Handle spinner updates: if new output doesn't start with newline
                # and accumulated doesn't end with newline, replace the last line
                if accumulated_output and not accumulated_output.endswith('\n') and not new_output.startswith('\n'):
                    # Find the last newline in accumulated output
                    last_newline = accumulated_output.rfind('\n')
                    if last_newline >= 0:
                        # Replace everything after the last newline
                        accumulated_output = accumulated_output[:last_newline + 1] + new_output
                    else:
                        # No newline in accumulated, replace everything
                        accumulated_output = new_output
                else:
                    accumulated_output += new_output
                await websocket.send_json({"type": "output", "data": accumulated_output})

            # Check for kill message
            try:
                user_message = await asyncio.wait_for(
                    websocket.receive_json(), timeout=0.1
                )
                if user_message == "kill":
                    console.log("[red]Stopping decomposition...[/red]")
                    decompose_task.cancel()
                    await websocket.send_json({
                        "type": "error",
                        "message": "Decomposition stopped by user request"
                    })
                    raise Exception("Process stopped by user request")
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Decomposition stopped by user request"
                })
                raise

            await asyncio.sleep(0.5)

        # Get final result
        result = await decompose_task

        # Send any remaining console output
        final_output = console.get_output()
        if final_output:
            accumulated_output += final_output
            await websocket.send_json({"type": "output", "data": accumulated_output})

        await asyncio.sleep(1)

        # Send the result
        await websocket.send_json({
            "type": "result",
            "data": {
                "decomposed_operations": result["decomposed_ops"],
                "winning_directive": result["winning_directive"],
                "candidates_evaluated": result["candidates_evaluated"],
                "original_outputs": result["original_outputs"],
                "decomposed_outputs": result["decomposed_outputs"],
                "comparison_rationale": result["comparison_rationale"],
                "cost": result["cost"],
            },
        })

    except WebSocketDisconnect:
        print(f"Decompose client {client_id} disconnected")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Decompose error:\n{error_traceback}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": str(e),
                "traceback": error_traceback
            })
        except Exception:
            pass
    finally:
        await websocket.close()
