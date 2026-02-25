from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import Dict, Any, List, Optional

from app.db.models.medical import Consultation, Prescription

class MedicalRecordService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_consultation(self, 
                                patient_id: str, 
                                session_id: str, 
                                symptoms: str, 
                                medical_history: Optional[str] = None) -> Consultation:
        """创建新的诊疗记录"""
        consultation = Consultation(
            patient_id=patient_id,
            session_id=session_id,
            symptoms=symptoms,
            medical_history=medical_history,
            status="pending"
        )
        self.db.add(consultation)
        await self.db.commit()
        await self.db.refresh(consultation)
        return consultation

    async def update_diagnosis(self, session_id: str, diagnosis_result: str, dialogue_history: List[Dict]) -> Optional[Consultation]:
        """更新诊断结果和对话历史"""
        stmt = select(Consultation).where(Consultation.session_id == session_id).order_by(Consultation.created_at.desc())
        result = await self.db.execute(stmt)
        # [Fix] Use scalars().first() instead of scalar_one_or_none() to handle potential duplicate session_ids gracefully
        consultation = result.scalars().first()
        
        if consultation:
            consultation.diagnosis_result = diagnosis_result
            consultation.dialogue_history = dialogue_history
            consultation.status = "diagnosed"
            await self.db.commit()
            await self.db.refresh(consultation)
        return consultation

    async def add_prescription(self, consultation_id: int, medication: Dict[str, str]) -> Prescription:
        """添加处方"""
        prescription = Prescription(
            consultation_id=consultation_id,
            medication_name=medication.get("name", "Unknown"),
            dosage=medication.get("dosage", ""),
            frequency=medication.get("frequency", ""),
            duration=medication.get("duration", ""),
            status="draft"
        )
        self.db.add(prescription)
        await self.db.commit()
        await self.db.refresh(prescription)
        return prescription
        
    async def get_consultation_history(self, patient_id: str) -> List[Consultation]:
        """获取患者历史诊疗记录 (包含处方)"""
        stmt = select(Consultation).where(Consultation.patient_id == patient_id).options(selectinload(Consultation.prescriptions))
        result = await self.db.execute(stmt)
        return result.scalars().all()
