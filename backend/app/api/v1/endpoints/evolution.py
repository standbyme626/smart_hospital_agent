import asyncio
import json
from typing import List, Dict, Any, AsyncGenerator
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.evolution_runner import EvolutionRunner
from pathlib import Path

router = APIRouter()

class EvolutionStartRequest(BaseModel):
    department: str
    speed_multiplier: float = 1.0

class HumanJudgeRequest(BaseModel):
    feedback: Dict[str, Any]

class EvolutionManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.queues = []
            cls._instance.is_running = False # [Lock] Global lock
            cls._instance.should_stop = False # [Signal] Stop signal
        return cls._instance

    def __init__(self):
        # Ensure init is idempotent
        if not hasattr(self, "queues"):
            self.queues = []
            self.is_running = False
            self.should_stop = False

    async def subscribe(self):
        """
        Subscribe to the event stream.
        Yields SSE events.
        """
        queue = asyncio.Queue()
        self.queues.append(queue)
        try:
            while True:
                data = await queue.get()
                # SSE format: data: {json_content}\n\n
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            if queue in self.queues:
                self.queues.remove(queue)

    async def broadcast(self, event: dict):
        """
        Broadcast event to all connected clients.
        """
        for queue in self.queues:
            await queue.put(event)

    def stop_evolution(self):
        """Signal the evolution to stop."""
        self.should_stop = True

manager = EvolutionManager()

async def run_evolution_background(department: str, speed_multiplier: float):
    """
    Background task wrapper for EvolutionRunner.
    """
    # NOTE: is_running is now set in the API endpoint to prevent race conditions
    try:
        runner = EvolutionRunner(
            department=department, 
            rounds=5, # Default 5 rounds
            speed_multiplier=speed_multiplier,
            event_callback=manager.broadcast,
            stop_signal_check=lambda: manager.should_stop
        )
        await runner.run()
    except Exception as e:
        await manager.broadcast({"sender": "SYSTEM", "content": f"Critical Error: {str(e)}", "type": "error"})
    finally:
        manager.is_running = False
        manager.should_stop = False # Reset stop signal
        await manager.broadcast({"sender": "SYSTEM", "content": "Evolution Cycle Finished.", "type": "status"})

@router.post("/start")
async def start_evolution(req: EvolutionStartRequest, background_tasks: BackgroundTasks):
    """
    Start the evolution loop in the background.
    """
    if manager.is_running:
        raise HTTPException(status_code=409, detail="Evolution cycle is already running.")
    
    # [Fix] Acquire lock IMMEDIATELY in the request handler, not in the background task.
    # This prevents race conditions where multiple requests could slip through
    # before the background task starts.
    manager.is_running = True
    manager.should_stop = False
        
    background_tasks.add_task(run_evolution_background, req.department, req.speed_multiplier)
    return {"status": "started", "department": req.department, "mode": "evolution"}

@router.post("/stop")
async def stop_evolution():
    """
    Force stop the running evolution cycle.
    """
    if not manager.is_running:
        return {"status": "ignored", "message": "No evolution running."}
    
    manager.stop_evolution()
    return {"status": "stopping", "message": "Stop signal sent to background task."}

@router.get("/stream")
async def stream_evolution():
    """
    SSE Stream for evolution events.
    """
    return StreamingResponse(manager.subscribe(), media_type="text/event-stream")

@router.post("/human_judge")
async def human_judge(req: HumanJudgeRequest):
    """
    Record human feedback for Audit Agent fine-tuning.
    """
    # Save to data/knowledge_base/audit_training_set.json
    file_path = Path("data/knowledge_base/audit_training_set.json")
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    if file_path.exists():
        try:
            content = file_path.read_text(encoding='utf-8')
            if content.strip():
                data = json.loads(content)
        except Exception:
            data = []
            
    data.append(req.feedback)
    
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    return {"status": "recorded", "count": len(data)}
