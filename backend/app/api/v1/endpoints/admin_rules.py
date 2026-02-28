from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.db.models.medical_rules import DrugInteraction
from app.schemas.rules import InteractionCreate, InteractionResponse
from app.services.medical_rule_service import medical_rule_service
from app.core.security.rbac import require_roles
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()

@router.post("/interactions", response_model=InteractionResponse, status_code=201)
async def create_interaction(
    interaction: InteractionCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    actor: dict = Depends(require_roles("admin")),
):
    """
    新增药物相互作用规则 (Admin Only)
    
    创建新规则后，会触发 MedicalRuleService 的缓存刷新。
    """
    try:
        # 1. 创建 DB 对象
        new_rule = DrugInteraction(
            drug_a=interaction.drug_a,
            drug_b=interaction.drug_b,
            severity=interaction.severity,
            description=interaction.description
        )
        
        # 2. 写入数据库
        db.add(new_rule)
        await db.commit()
        await db.refresh(new_rule)
        
        logger.info(
            "interaction_rule_created",
            drug_a=interaction.drug_a,
            drug_b=interaction.drug_b,
            actor_id=actor.get("user_id", ""),
            actor_role=actor.get("role", ""),
        )

        # 3. 触发缓存刷新 (使用后台任务避免阻塞 API 响应)
        background_tasks.add_task(medical_rule_service.refresh_rules)
        
        return new_rule

    except Exception as e:
        logger.error(f"Failed to create interaction rule: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
