from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from docetl.runner import DSLRunner
import asyncio
from asyncio import Task
from rich.logging import RichHandler
import logging
from datetime import datetime, timedelta
from enum import Enum
from server.app.models import OptimizeResult, TaskStatus, OptimizeRequest, PipelineRequest

# Setup logging
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

router = APIRouter()

# Task storage
tasks: Dict[str, OptimizeResult] = {}
asyncio_tasks: Dict[str, Task] = {}

# Configuration
COMPLETED_TASK_TTL = timedelta(hours=1)

async def cleanup_old_tasks():
    """Background task to clean up completed tasks"""
    while True:
        try:
            current_time = datetime.now()
            task_ids_to_remove = []

            for task_id, task in tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and 
                    current_time - task.completed_at > COMPLETED_TASK_TTL):
                    task_ids_to_remove.append(task_id)

            for task_id in task_ids_to_remove:
                del tasks[task_id]
                
            await asyncio.sleep(60)
            
        except Exception as e:
            logging.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

async def run_optimization(task_id: str, yaml_config: str, step_name: str, op_name: str):
    """Execute the optimization task"""
    try:
        tasks[task_id].status = TaskStatus.PROCESSING
        
        # Run the actual optimization in a separate thread to not block
        runner = DSLRunner.from_yaml(yaml_config)
        should_optimize, input_data, output_data, cost = await asyncio.to_thread(
            runner.should_optimize,
            step_name,
            op_name
        )
        
        # Update task result
        tasks[task_id].status = TaskStatus.COMPLETED
        tasks[task_id].should_optimize = should_optimize
        tasks[task_id].input_data = input_data
        tasks[task_id].output_data = output_data
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
        runner.reset_env()

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



# @router.websocket("/ws/run_pipeline")
# async def websocket_run_pipeline(websocket: WebSocket):
#     await websocket.accept()
#     runner = None
#     try:
#         config = await websocket.receive_json()
#         runner = DSLRunner.from_yaml(config["yaml_config"])

#         if config.get("clear_intermediate", False):
#             runner.clear_intermediate()

#         if config.get("optimize", False):
#             logging.info(f"Optimizing pipeline with model {config.get('optimizer_model', 'gpt-4o')}")
            
#             async def run_pipeline():
#                 return await asyncio.to_thread(runner.optimize, return_pipeline=False, model=config.get("optimizer_model", "gpt-4o"))

#         else:
#             async def run_pipeline():
#                 return await asyncio.to_thread(runner.load_run_save)

#         pipeline_task = asyncio.create_task(run_pipeline())

#         while not pipeline_task.done():
#             console_output = runner.console.file.getvalue()
#             await websocket.send_json({"type": "output", "data": console_output})

#             if config.get("optimize", False):
#                 optimizer_progress = runner.console.get_optimizer_progress()
#                 rationale = runner.console.optimizer_rationale
#                 await websocket.send_json({
#                     "type": "optimizer_progress", 
#                     "status": optimizer_progress[0], 
#                     "progress": optimizer_progress[1],
#                     "rationale": rationale[1] if rationale is not None else "",
#                     "should_optimize": rationale[0] if rationale is not None else False,
#                     "validator_prompt": rationale[2] if rationale is not None else ""
#                 })

#             # Check for incoming messages from the user
#             try:
#                 user_message = await asyncio.wait_for(
#                     websocket.receive_json(), timeout=0.1
#                 )

#                 if user_message == "kill":
#                     runner.console.print("Stopping process...")
#                     await websocket.send_json({
#                         "type": "error",
#                         "message": "Process stopped by user request"
#                     })
#                     raise Exception("Process stopped by user request")

#                 # Process the user message and send it to the runner
#                 runner.console.post_input(user_message)
#             except asyncio.TimeoutError:
#                 pass  # No message received, continue with the loop

#             await asyncio.sleep(0.5)

#         # Final check to send any remaining output
#         result = await pipeline_task

#         console_output = runner.console.file.getvalue()
#         if console_output:
#             await websocket.send_json({"type": "output", "data": console_output})

#         # Sleep for a short duration to ensure all output is captured
#         await asyncio.sleep(3)

#         # If optimize is true, send back the optimized operations
#         if config.get("optimize", False):
#             optimized_config, cost = result

#             # Send the operations back in order
#             new_pipeline_steps = optimized_config["pipeline"]["steps"]
#             new_pipeline_op_name_to_op_map = {op["name"]: op for op in optimized_config["operations"]}
#             new_ops_in_order = []
#             for new_step in new_pipeline_steps:
#                 for op in new_step.get("operations", []):
#                     if op not in new_ops_in_order:
#                         new_ops_in_order.append(new_pipeline_op_name_to_op_map[op])

#             await websocket.send_json(
#                 {
#                     "type": "result",
#                     "data": {
#                         "message": "Pipeline executed successfully",
#                         "cost": cost,
#                         "optimized_ops": new_ops_in_order,
#                         "yaml_config": config["yaml_config"],
#                     },
#                 }
#             )
#         else:
#             await websocket.send_json(
#                 {
#                     "type": "result",
#                     "data": {
#                         "message": "Pipeline executed successfully",
#                         "cost": result,
#                         "yaml_config": config["yaml_config"],
#                     },
#                 }
#             )
#     except WebSocketDisconnect:
#         if runner is not None:
#             runner.reset_env()
#         print("Client disconnected")
#     except Exception as e:
#         import traceback

#         error_traceback = traceback.format_exc()
#         print(f"Error occurred:\n{error_traceback}")
#         await websocket.send_json({"type": "error", "data": str(e), "traceback": error_traceback})
#     finally:
#         if runner is not None:
#             runner.reset_env()
#         await websocket.close()

import modal

# Modal queue setup
pipeline_queue = modal.Queue.from_name("docetl-message-queue", create_if_missing=True)

async def process_pipeline(config: Dict[str, Any], client_id: str, task_id: str):
    """Process a pipeline task"""
    runner = None
    last_console_output = ""
    last_output_time = 0
    
    try:
        runner = DSLRunner.from_yaml(config["yaml_config"])

        if config.get("clear_intermediate", False):
            runner.clear_intermediate()

        # Start console output monitoring task
        async def monitor_console_output():
            nonlocal last_console_output
            while True:
                current_output = runner.console.file.getvalue()
                if current_output != last_console_output:
                    pipeline_queue.put({
                        "type": "output",
                        "task_id": task_id,
                        "data": current_output
                    }, partition=client_id)
                    last_console_output = current_output
                await asyncio.sleep(1)  # Check every second

        # Start the monitoring task
        monitor_task = asyncio.create_task(monitor_console_output())

        if config.get("optimize", False):
            logging.info(f"Optimizing pipeline with model {config.get('optimizer_model', 'gpt-4o')}")
            result = await asyncio.to_thread(
                runner.optimize,
                return_pipeline=False,
                model=config.get("optimizer_model", "gpt-4o")
            )
            optimized_config, cost = result
            
            # Process optimized operations
            new_pipeline_steps = optimized_config["pipeline"]["steps"]
            new_pipeline_op_name_to_op_map = {op["name"]: op for op in optimized_config["operations"]}
            new_ops_in_order = []
            for new_step in new_pipeline_steps:
                for op in new_step.get("operations", []):
                    if op not in new_ops_in_order:
                        new_ops_in_order.append(new_pipeline_op_name_to_op_map[op])
            
            final_result = {
                "cost": cost,
                "optimized_ops": new_ops_in_order,
                "yaml_config": config["yaml_config"]
            }
        else:
            cost = await asyncio.to_thread(runner.load_run_save)
            final_result = {
                "cost": cost,
                "yaml_config": config["yaml_config"]
            }

        # Send final result
        pipeline_queue.put({
            "type": "result",
            "task_id": task_id,
            "data": {
                "message": "Pipeline executed successfully",
                **final_result
            }
        }, partition=client_id)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        pipeline_queue.put({
            "type": "error",
            "task_id": task_id,
            "data": str(e),
            "traceback": error_traceback
        }, partition=client_id)
    finally:
        # Cancel the console monitoring task
        if 'monitor_task' in locals():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
        if runner is not None:
            runner.reset_env()

async def message_send_loop(websocket: WebSocket, client_id: str, task_id: str, runner: DSLRunner, config: Dict[str, Any]):
    """Loop to continuously send messages from queue and process user input"""
    while True:
        try:
            # All output now comes through the queue
            # Check for messages from queue
            if config.get("optimize", False) and hasattr(runner.console, 'get_optimizer_progress'):
                optimizer_progress = runner.console.get_optimizer_progress()
                rationale = runner.console.optimizer_rationale
                if optimizer_progress:
                    await websocket.send_json({
                        "type": "optimizer_progress",
                        "status": optimizer_progress[0],
                        "progress": optimizer_progress[1],
                        "rationale": rationale[1] if rationale is not None else "",
                        "should_optimize": rationale[0] if rationale is not None else False,
                        "validator_prompt": rationale[2] if rationale is not None else ""
                    })

            # Check for messages from queue
            try:
                message = await asyncio.wait_for(
                    pipeline_queue.get.aio(partition=client_id),
                    timeout=0.1
                )
                
                if message["task_id"] == task_id:
                    await websocket.send_json(message)
                    if message["type"] in ["result", "error"]:
                        break
            except asyncio.TimeoutError:
                pass

            # Check for user messages
            try:
                user_message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1
                )

                if user_message == "kill":
                    runner.console.print("Stopping process...")
                    await websocket.send_json({
                        "type": "error",
                        "task_id": task_id,
                        "message": "Process stopped by user request"
                    })
                    raise Exception("Process stopped by user request")

                # Process user message
                runner.console.post_input(user_message)
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Error in message loop: {e}")
            break

@router.websocket("/ws/run_pipeline/{client_id}")
async def websocket_run_pipeline(websocket: WebSocket, client_id: str):
    task_id = str(uuid.uuid4())
    runner = None
    send_task = None

    try:
        await websocket.accept()
        
        # Get configuration
        config = await websocket.receive_json()
        
        # Initialize runner
        runner = DSLRunner.from_yaml(config["yaml_config"])
        
        # Start message send loop
        send_task = asyncio.create_task(
            message_send_loop(websocket, client_id, task_id, runner, config)
        )
        
        # Start pipeline processing
        process_task = asyncio.create_task(
            process_pipeline(config, client_id, task_id)
        )
        
        # Wait for both tasks to complete
        await asyncio.gather(send_task, process_task)
        
    except WebSocketDisconnect:
        logging.info(f"Client {client_id} disconnected")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logging.error(f"Error occurred:\n{error_traceback}")
        if websocket.client_state.connected:
            await websocket.send_json({
                "type": "error",
                "task_id": task_id,
                "data": str(e),
                "traceback": error_traceback
            })
    finally:
        if send_task and not send_task.done():
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
        if runner is not None:
            runner.reset_env()
        await websocket.close()