import json
import structlog
import asyncio
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.core.config import settings

logger = structlog.get_logger(__name__)

class SQLAdapter:
    """
    HIS 数据库适配器 (Phase 4)
    负责将 Agent 的状态数据无损同步到真实的 HIS 业务表中。
    支持事务写回与错误回滚。
    """
    
    def __init__(self):
        # 使用 aiosqlite 进行本地测试，生产环境应切换为 PostgreSQL/MySQL
        # 为了兼容性，我们根据 settings.DATABASE_URL 判断
        self.db_url = settings.DATABASE_URL
        if "sqlite" in self.db_url:
             self.engine = create_async_engine(self.db_url, echo=False)
        else:
             # Assume standard SQL (Postgres/MySQL)
             self.engine = create_async_engine(self.db_url, pool_size=10, max_overflow=20)
             
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        """
        初始化 HIS 表结构 (仅用于 Demo/Test 环境)
        """
        async with self.engine.begin() as conn:
            # Check dialect
            dialect = self.engine.dialect.name
            
            # Syntax adaptation
            if dialect == "sqlite":
                pk_syntax = "INTEGER PRIMARY KEY AUTOINCREMENT"
            else: # Postgres
                pk_syntax = "SERIAL PRIMARY KEY"

            # 创建模拟表
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS his_patient_records (
                    id {pk_syntax},
                    patient_id VARCHAR(50) UNIQUE,
                    name VARCHAR(100),
                    age INTEGER,
                    gender VARCHAR(10),
                    medical_history TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS his_appointments (
                    id {pk_syntax},
                    order_id VARCHAR(50) UNIQUE,
                    patient_id VARCHAR(50),
                    department VARCHAR(100),
                    slot_id VARCHAR(50),
                    status VARCHAR(20),
                    payment_status VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS his_audit_logs (
                    id {pk_syntax},
                    action VARCHAR(50),
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
    async def sync_to_his(self, state: Dict[str, Any]) -> bool:
        """
        核心写回逻辑 (Side-loading)
        将 AgentState 中的 user_profile 和 order_context 同步到 HIS。
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    # 1. Sync Patient Record
                    user_profile = state.get("user_profile")
                    if user_profile:
                        # Normalize Pydantic to dict
                        p_data = user_profile.dict() if hasattr(user_profile, "dict") else user_profile
                        
                        # Upsert Patient
                        # 注意: SQLite 不支持 ON CONFLICT UPDATE 标准语法 (部分版本支持)，这里用简单的 Check-Insert
                        # 生产环境建议使用 UPSERT
                        
                        patient_id = p_data.get("patient_id")
                        if patient_id and patient_id != "guest":
                            # Check existence
                            res = await session.execute(
                                text("SELECT 1 FROM his_patient_records WHERE patient_id = :pid"), 
                                {"pid": patient_id}
                            )
                            if not res.scalar():
                                await session.execute(
                                    text("""
                                        INSERT INTO his_patient_records (patient_id, name, age, gender, medical_history)
                                        VALUES (:pid, :name, :age, :gender, :hist)
                                    """),
                                    {
                                        "pid": patient_id,
                                        "name": p_data.get("name"),
                                        "age": p_data.get("age"),
                                        "gender": p_data.get("gender"),
                                        "hist": json.dumps(p_data.get("medical_history", []), ensure_ascii=False)
                                    }
                                )
                    
                    # 2. Sync Appointment / Order
                    order_ctx = state.get("order_context")
                    if order_ctx:
                        o_data = order_ctx.dict() if hasattr(order_ctx, "dict") else order_ctx
                        order_id = o_data.get("order_id")
                        
                        if order_id:
                            # Check existence
                            res = await session.execute(
                                text("SELECT 1 FROM his_appointments WHERE order_id = :oid"),
                                {"oid": order_id}
                            )
                            if not res.scalar():
                                await session.execute(
                                    text("""
                                        INSERT INTO his_appointments (order_id, patient_id, department, slot_id, status, payment_status)
                                        VALUES (:oid, :pid, :dept, :slot, :status, :pay_status)
                                    """),
                                    {
                                        "oid": order_id,
                                        "pid": p_data.get("patient_id", "guest") if user_profile else "guest",
                                        "dept": o_data.get("department"),
                                        "slot": o_data.get("slot_id"), # Assuming slot_id is mapped inside order_id or separate
                                        "status": "booked", # Default
                                        "pay_status": o_data.get("payment_status")
                                    }
                                )
                            else:
                                # Update status
                                await session.execute(
                                    text("""
                                        UPDATE his_appointments 
                                        SET payment_status = :pay_status
                                        WHERE order_id = :oid
                                    """),
                                    {
                                        "pay_status": o_data.get("payment_status"),
                                        "oid": order_id
                                    }
                                )

                    logger.info("his_sync_success", patient_id=p_data.get("patient_id") if user_profile else "N/A")
                    return True

        except Exception as e:
            logger.error("his_sync_failed", error=str(e))
            return False

    async def log_audit(self, action: str, details: Dict):
        """记录安全审计日志"""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    await session.execute(
                        text("INSERT INTO his_audit_logs (action, details) VALUES (:act, :det)"),
                        {"act": action, "det": json.dumps(details, ensure_ascii=False)}
                    )
        except Exception as e:
            logger.error("audit_log_failed", error=str(e))

# Singleton
sql_adapter = SQLAdapter()
