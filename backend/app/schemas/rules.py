from typing import List
from pydantic import BaseModel, Field

class InteractionCreate(BaseModel):
    drug_a: str = Field(..., description="药物 A 的英文名", min_length=1)
    drug_b: str = Field(..., description="药物 B 的英文名", min_length=1)
    severity: str = Field(..., description="严重程度 (High, Medium, Low)")
    description: str = Field(..., description="相互作用描述")

class InteractionResponse(BaseModel):
    id: int
    drug_a: str
    drug_b: str
    severity: str
    description: str

    class Config:
        from_attributes = True
