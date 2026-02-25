import asyncio
import time
import os
from typing import Optional, Callable, Dict, Any, List
from app.qa_engine.patient_agent import PatientAdversary
from app.qa_engine.audit_agent import AuditSystem
from app.core.graph.workflow import app as medical_core_graph
from app.core.config import settings

class EvolutionRunner:
    def __init__(
        self, 
        department: str, 
        rounds: int = 1, 
        speed_multiplier: float = 1.0, 
        event_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        stop_signal_check: Optional[Callable[[], bool]] = None
    ):
        self.department = department
        self.rounds = rounds
        self.speed_multiplier = speed_multiplier
        self.event_callback = event_callback
        self.stop_signal_check = stop_signal_check
        self.audit_system = AuditSystem()
        self.scores = []
        self.run_id = f"run_{int(time.time())}" # [New] Unique Run ID
        
        # Ensure Evolution Mode is ON
        os.environ["EVOLUTION_MODE"] = "true"
        settings.EVOLUTION_MODE = True

    async def emit(self, sender: str, node_id: str, content: str, telemetry: Optional[Dict[str, Any]] = None):
        event = {
            "run_id": self.run_id, # [New]
            "sender": sender,
            "node_id": node_id,
            "content": content,
            "telemetry": telemetry or {},
            "is_test": True
        }
        
        # Call callback
        if self.event_callback:
            if asyncio.iscoroutinefunction(self.event_callback):
                await self.event_callback(event)
            else:
                self.event_callback(event)
        
        # Also print to stdout for debug if needed, or if no callback
        if not self.event_callback:
            print(f"[{sender}] {content}")

        # Speed control (simulating thinking time or slow motion)
        if self.speed_multiplier > 0:
            # Base delay 0.5s / multiplier. 
            # Multiplier 0.1 -> 5s delay (slow motion). 
            # Multiplier 5 -> 0.1s delay (fast).
            delay = 0.5 / self.speed_multiplier
            await asyncio.sleep(delay)

    async def run(self):
        await self.emit("SYSTEM", "init", f"🚀 Starting Evolution Cycle for [{self.department}] (RunID: {self.run_id})")
        
        for i in range(self.rounds):
            # [Check] Stop Signal
            if self.stop_signal_check and self.stop_signal_check():
                await self.emit("SYSTEM", "stopped", "🛑 Evolution manually stopped by user.")
                break

            round_id = f"{self.run_id}_round_{i+1}"
            await self.emit("SYSTEM", "round_start", f"Round {i+1}/{self.rounds} Started", {"round": i+1})
            
            # 1. Initialize Red Team (Patient)
            try:
                patient = PatientAdversary(department=self.department)
                await self.emit("PATIENT", "persona_load", f"Persona: {patient.persona['type']} | Case: {patient.case_data['id']}")
            except Exception as e:
                await self.emit("SYSTEM", "error", f"Failed to init patient: {str(e)}")
                continue

            chat_history = []
            config = {"configurable": {"thread_id": round_id}}
            
            # Turn 1: Patient speaks
            try:
                user_msg = await patient.speak()
                await self.emit("PATIENT", "speak", user_msg)
            except Exception as e:
                await self.emit("SYSTEM", "error", f"Patient failed to speak: {str(e)}")
                continue

            current_input = user_msg
            start_time = time.time()
            turn_count = 0
            emergency_stop = False
            
            # Conversation Loop (3 turns)
            conversation_failed = False
            for turn in range(3):
                # [Check] Stop Signal
                if self.stop_signal_check and self.stop_signal_check():
                    await self.emit("SYSTEM", "stopped", "🛑 Evolution manually stopped by user.")
                    emergency_stop = True
                    break

                turn_count += 1
                await self.emit("SYSTEM", "thinking", f"Turn {turn+1} processing...")
                
                # Check for empty input (loop prevention)
                if not current_input or not current_input.strip():
                    await self.emit("SYSTEM", "warning", "Empty input detected, skipping turn.")
                    break

                input_state = {"user_input": current_input, "messages": [("user", current_input)]}
                
                # Run Graph
                try:
                    # [Critical] Use explicit timeout to prevent hanging
                    # Increase timeout for local models (e.g. 120s)
                    final_state = await asyncio.wait_for(
                        medical_core_graph.ainvoke(input_state, config=config),
                        timeout=120.0 
                    )
                    
                    messages = final_state.get("messages", [])
                    if messages and hasattr(messages[-1], "content"):
                        system_reply = messages[-1].content
                    else:
                        system_reply = final_state.get("final_output", "系统未响应")
                except asyncio.TimeoutError:
                    system_reply = "System Error: Timeout"
                    await self.emit("SYSTEM", "error", "Graph Execution Timeout")
                    conversation_failed = True
                    break
                except Exception as e:
                    system_reply = f"System Error: {str(e)}"
                    await self.emit("SYSTEM", "error", f"Graph Execution Failed: {str(e)}")
                    conversation_failed = True
                    break
                
                # Check for repetitive or invalid reply
                if not system_reply or "系统未响应" in system_reply:
                     await self.emit("SYSTEM", "warning", "Empty or Invalid System Reply, retrying...")
                     # Optional: Retry logic here
                
                await self.emit("DOCTOR", "reply", system_reply)
                
                # Update Chat History correctly
                # Note: PatientAdversary maintains its own history, but we keep a local copy for Audit
                chat_history.append({"role": "patient", "content": current_input})
                chat_history.append({"role": "doctor", "content": system_reply})
                
                # [NEW] Wait for state sync to ensure atomic processing and avoid race conditions
                await asyncio.sleep(2)
                
                try:
                    # Pass the doctor's reply to the patient so they can respond to IT
                    current_input = await patient.speak(system_reply)
                    await self.emit("PATIENT", "speak", current_input)
                except Exception as e:
                     await self.emit("SYSTEM", "error", f"Patient failed to reply: {str(e)}")
                     conversation_failed = True
                     break
            
            if conversation_failed:
                await self.emit("SYSTEM", "warning", "Round aborted due to errors.")
                continue

            end_time = time.time()
            
            end_time = time.time()
            
            # 3. Patient Evaluation
            ux_score = await patient.evaluate_satisfaction()
            await self.emit("PATIENT", "eval", f"Satisfaction Score: {ux_score}", {"score": ux_score})
            
            # 4. Audit Evaluation
            metadata = {
                "latency_ms": (end_time - start_time) * 1000,
                "total_tokens": 500 * turn_count, 
                "ux_score": ux_score
            }
            
            await self.emit("AUDITOR", "analyzing", "Analyzing conversation against standards...")
            
            audit_result = await self.audit_system.audit_session(
                patient.case_data, 
                chat_history, 
                ux_score,
                metadata=metadata
            )
            
            global_score = audit_result['global_score']
            self.scores.append(global_score)
            
            await self.emit(
                "AUDITOR", 
                "score", 
                f"Global Score: {global_score}", 
                {"score": global_score, "breakdown": audit_result['breakdown']}
            )
            
            # 5. Feedback Loop (DSPy Stub)
            if global_score < 70:
                await self.emit("SYSTEM", "optimization", "⚠️ Score < 70. Triggering DSPy optimization...")
                # self.optimize_prompt(self.department) # Placeholder

            if emergency_stop:
                await self.emit("SYSTEM", "complete", "🛑 Evolution halted due to Emergency Case.")
                break
                
        avg_score = sum(self.scores) / len(self.scores) if self.scores else 0
        await self.emit("SYSTEM", "complete", f"Evolution Cycle Complete. Average: {avg_score:.2f}", {"avg_score": avg_score})
        return avg_score
