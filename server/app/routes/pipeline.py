from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from server.app.models import PipelineRequest
from docetl.runner import DSLRunner
import asyncio
import queue

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

            async def run_pipeline():
                return await asyncio.to_thread(runner.optimize, return_pipeline=False)

        else:

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
            # find the operation that has optimize = true
            optimized_op = None
            for op in optimized_config["operations"]:
                if op.get("optimize", False):
                    optimized_op = op
                    break

            if not optimized_op:
                raise HTTPException(
                    status_code=500, detail="No optimized operation found"
                )

            await websocket.send_json(
                {
                    "type": "result",
                    "data": {
                        "message": "Pipeline executed successfully",
                        "cost": cost,
                        "optimized_op": optimized_op,
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
        await websocket.send_json({"type": "error", "data": str(e)})
