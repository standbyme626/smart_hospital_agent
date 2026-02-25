
from typing import Dict, Any, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class PrescriptionItem(BaseModel):
    medication_name: str = Field(..., description="Drug name")
    dosage: str = Field(..., description="Dosage (e.g., '500mg')")
    frequency: str = Field(..., description="Frequency (e.g., '3 times a day')")
    duration: str = Field(..., description="Duration (e.g., '7 days')")

class DraftPrescriptionInput(BaseModel):
    patient_id: str = Field(..., description="Patient ID")
    diagnosis: str = Field(..., description="Confirmed diagnosis")
    medications: List[PrescriptionItem] = Field(..., description="List of medications")
    notes: str = Field(None, description="Additional notes")

@tool("draft_prescription")
def draft_prescription(patient_id: str, diagnosis: str, medications: List[Dict[str, str]], notes: str = "") -> str:
    """
    Draft a prescription for review.
    This tool does NOT submit the prescription. It only saves it for audit.
    """
    # In a real implementation, this would update the state.
    # Since tools usually return strings, we return a success message.
    # The graph node calling this tool should be responsible for parsing the tool call 
    # and updating the state's `draft_prescription` field.
    return "Prescription drafted successfully. Waiting for audit."
