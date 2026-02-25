import dspy
from pydantic import BaseModel, Field
from typing import List, Optional
from typing import List, Optional

# --- Pydantic Models for Structured Output ---

class MedicalDiagnosisOutput(BaseModel):
    reasoning: str = Field(..., description="Step-by-step clinical reasoning process based on symptoms and history.")
    suggested_diagnosis: List[str] = Field(..., description="List of potential diagnoses, ranked by likelihood.")
    confidence_score: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    follow_up_questions: List[str] = Field(..., description="List of questions to ask the patient to clarify the diagnosis.")

# --- DSPy Signatures ---

class MedicalDiagnosisSignature(dspy.Signature):
    """
    基于患者画像、病史、对话历史、当前症状及检索到的医学知识进行诊断推理。
    """
    patient_profile = dspy.InputField(desc="患者的基础信息（年龄、性别等）")
    medical_history = dspy.InputField(desc="患者的既往病史和过敏史")
    conversation_history = dspy.InputField(desc="医生与患者的完整对话历史")
    current_symptoms = dspy.InputField(desc="患者在本次会话中报告的症状")
    retrieved_knowledge = dspy.InputField(desc="从向量数据库和知识图谱检索到的相关医学知识")
    
    # DSPy will automatically enforce the Pydantic model structure if we use TypedPredictor
    # output: MedicalDiagnosisOutput = dspy.OutputField()
    # However, standard dspy.Predict uses string fields. 
    # For now, we define fields explicitly for ChainOfThought, 
    # or use Functional/Typed predictors if dspy version supports it well.
    # Let's stick to standard signature for flexibility and parse later, 
    # OR use the newer typed signature style.
    
    reasoning = dspy.OutputField(desc="详细的临床推理过程（中文）")
    suggested_diagnosis = dspy.OutputField(desc="可能的诊断列表（按可能性排序）")
    confidence_score = dspy.OutputField(desc="置信度评分（0.0 到 1.0 之间的浮点数）")
    follow_up_questions = dspy.OutputField(desc="3个相关的追问问题（用于明确诊断）")

# --- DSPy Modules ---

class MedicalConsultant(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning capabilities
        self.prog = dspy.ChainOfThought(MedicalDiagnosisSignature)

    def forward(self, patient_profile: str, medical_history: str, conversation_history: str, current_symptoms: str, retrieved_knowledge: str):
        return self.prog(
            patient_profile=patient_profile,
            medical_history=medical_history,
            conversation_history=conversation_history,
            current_symptoms=current_symptoms,
            retrieved_knowledge=retrieved_knowledge
        )
