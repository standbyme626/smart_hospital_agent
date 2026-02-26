from fastapi import APIRouter

from app.api.v1.endpoints import admin_rules, auth, chat, doctor, evolution

"""
API 路由汇总 (API Router Aggregator)
将所有 V1 版本的子路由统一注册到根路由下。
"""

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(doctor.router, prefix="/doctor", tags=["doctor"])
api_router.include_router(admin_rules.router, prefix="/admin/rules", tags=["admin-rules"])
api_router.include_router(evolution.router, prefix="/evolution", tags=["evolution"])
