from fastapi import APIRouter
from app.api.v1.endpoints import auth, chat, doctor

"""
API 路由汇总 (API Router Aggregator)
将所有 V1 版本的子路由（如 auth, users, triage 等）统一通过 include_router 注册到根路由下。
"""
api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(doctor.router, prefix="/doctor", tags=["doctor"])
from app.api.v1.endpoints import admin_rules
api_router.include_router(admin_rules.router, prefix="/admin/rules", tags=["admin-rules"])

from app.api.v1.endpoints import evolution
api_router.include_router(evolution.router, prefix="/evolution", tags=["evolution"])
