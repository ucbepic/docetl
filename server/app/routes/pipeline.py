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
        cost = runner.run()
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

        async def run_pipeline():
            return await asyncio.to_thread(runner.run)

        pipeline_task = asyncio.create_task(run_pipeline())

        while not pipeline_task.done():
            console_output = runner.console.file.getvalue()
            await websocket.send_json({"type": "output", "data": console_output})
            await asyncio.sleep(0.5)

        # Final check to send any remaining output
        # Sleep for a short duration to ensure all output is captured

        cost = await pipeline_task

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
                    "cost": cost,
                },
            }
        )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "data": str(e)})
