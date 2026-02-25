
import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.llm.llm_factory import get_smart_llm
from app.core.config import settings
from app.domain.states.sub_states import PrescriptionState

AUDIT_SYSTEM_PROMPT = """You are a Senior Medical Auditor (Safety Officer).
Your goal is to review the draft prescription for safety, accuracy, and guideline compliance.

### Context
- Diagnosis: {diagnosis}
- Patient Info: {patient_info}

### Draft Prescription
{draft}

### Audit Rules
1. **Contraindications**: Check for pregnancy, allergies (if known), or major conditions.
2. **Interactions**: Ensure drug interactions have been checked (look at context).
3. **Dosage**: Verify dosage is within standard range.
4. **Appropriateness**: Is the medication suitable for the diagnosis?

### Output Format
Return a JSON object with:
- `approved`: boolean
- `feedback`: string (reason for rejection or comments)
- `risk_level`: "LOW", "MEDIUM", "HIGH"
"""

class SafetyAuditNode:
    def __init__(self):
        self.llm = get_smart_llm(temperature=0.0)

    async def __call__(self, state: PrescriptionState) -> Dict[str, Any]:
        print("[DEBUG] Node SafetyAuditNode Start")
        
        draft = state.get("draft_prescription")
        
        # If no draft in state, try to find it in recent tool calls
        if not draft:
            messages = state.get("messages", [])
            # Find last AIMessage with tool_calls
            last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls), None)
            
            if last_ai_msg:
                for tool_call in last_ai_msg.tool_calls:
                    if tool_call["name"] == "draft_prescription":
                        draft = tool_call["args"]
                        print("[DEBUG] Found draft in tool_calls")
                        break

        if not draft:
            print("[WARN] No draft prescription found.")
            return {
                "audit_feedback": "ERROR: No draft prescription found. Please use `draft_prescription` tool.",
                "step": "pharmacist" 
            }
            
        diagnosis = state.get("confirmed_diagnosis", "Unknown")
        # In a real scenario, we'd fetch patient profile
        patient_info = f"ID: {state.get('patient_id')}" 
        
        prompt = AUDIT_SYSTEM_PROMPT.format(
            diagnosis=diagnosis,
            patient_info=patient_info,
            draft=json.dumps(draft, ensure_ascii=False, indent=2)
        )
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Please audit this prescription.")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            result = json.loads(content)
            
            approved = result.get("approved", False)
            feedback = result.get("feedback", "No feedback provided.")
            risk_level = result.get("risk_level", "UNKNOWN")
            
            print(f"🔍 Audit Result: Approved={approved}, Risk={risk_level}")
            if not approved:
                print(f"❌ Audit Feedback: {feedback}")
            
            if approved:
                return {
                    "audit_feedback": f"AUDIT APPROVED: {feedback}",
                    "risk_level": risk_level,
                    "draft_prescription": draft, # Ensure state is updated
                    "final_prescription": draft, # Promote draft to final
                    "step": "finalizing" # Change step to finalizing
                }
            else:
                return {
                    "audit_feedback": f"AUDIT REJECTED: {feedback}",
                    "risk_level": risk_level,
                    "draft_prescription": draft, # Persist draft
                    "audit_retry_count": state.get("audit_retry_count", 0) + 1,
                    "step": "pharmacist" # Send back to pharmacist
                }
                
        except Exception as e:
            print(f"❌ Audit failed: {e}")
            return {
                "audit_feedback": f"Audit system error: {str(e)}",
                "step": "pharmacist" # Fallback
            }
