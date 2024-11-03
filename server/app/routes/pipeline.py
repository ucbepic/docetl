import os
import signal
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from server.app.models import PipelineRequest
from docetl.runner import DSLRunner
import asyncio
from rich.logging import RichHandler
import logging

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
router = APIRouter()


@router.post("/run_pipeline")
def run_pipeline(request: PipelineRequest):
    try:
        runner = DSLRunner.from_yaml(request.yaml_config)
        cost = runner.load_run_save()
        return {"cost": cost, "message": "Pipeline executed successfully"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/run_pipeline")
async def websocket_run_pipeline(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await websocket.receive_json()
        runner = DSLRunner.from_yaml(config["yaml_config"])

        if config.get("clear_intermediate", False):
            runner.clear_intermediate()

        if config.get("optimize", False):
            logging.info(f"Optimizing pipeline with model {config.get('optimizer_model', 'gpt-4o')}")
            
            async def run_pipeline():
                return await asyncio.to_thread(runner.optimize, return_pipeline=False, model=config.get("optimizer_model", "gpt-4o"))

        else:

            async def run_pipeline():
                return await asyncio.to_thread(runner.load_run_save)

        pipeline_task = asyncio.create_task(run_pipeline())

        while not pipeline_task.done():
            console_output = runner.console.file.getvalue()
            await websocket.send_json({"type": "output", "data": console_output})

            if config.get("optimize", False):
                optimizer_progress = runner.console.get_optimizer_progress()
                rationale = runner.console.optimizer_rationale
                await websocket.send_json({
                    "type": "optimizer_progress", 
                    "status": optimizer_progress[0], 
                    "progress": optimizer_progress[1],
                    "rationale": rationale[1] if rationale is not None else "",
                    "should_optimize": rationale[0] if rationale is not None else False,
                    "validator_prompt": rationale[2] if rationale is not None else ""
                })

            # Check for incoming messages from the user
            try:
                user_message = await asyncio.wait_for(
                    websocket.receive_json(), timeout=0.1
                )

                if user_message == "kill":
                    runner.console.print("Stopping process...")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Process stopped by user request"
                    })
                    raise Exception("Process stopped by user request")

                # Process the user message and send it to the runner
                runner.console.post_input(user_message)
            except asyncio.TimeoutError:
                pass  # No message received, continue with the loop

            await asyncio.sleep(0.5)

        # Final check to send any remaining output
        result = await pipeline_task

        console_output = runner.console.file.getvalue()
        if console_output:
            await websocket.send_json({"type": "output", "data": console_output})

        # Sleep for a short duration to ensure all output is captured
        await asyncio.sleep(3)

        # If optimize is true, send back the optimized operations
        if config.get("optimize", False):
            optimized_config, cost = result

            # Send the operations back in order
            new_pipeline_steps = optimized_config["pipeline"]["steps"]
            new_pipeline_op_name_to_op_map = {op["name"]: op for op in optimized_config["operations"]}
            new_ops_in_order = []
            for new_step in new_pipeline_steps:
                for op in new_step.get("operations", []):
                    if op not in new_ops_in_order:
                        new_ops_in_order.append(new_pipeline_op_name_to_op_map[op])

            await websocket.send_json(
                {
                    "type": "result",
                    "data": {
                        "message": "Pipeline executed successfully",
                        "cost": cost,
                        "optimized_ops": new_ops_in_order,
                    },
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "result",
                    "data": {
                        "message": "Pipeline executed successfully",
                        "cost": result,
                    },
                }
            )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        print(f"Error occurred:\n{error_traceback}")
        await websocket.send_json({"type": "error", "data": str(e) + "\n" + error_traceback})
    finally:
        await websocket.close()
