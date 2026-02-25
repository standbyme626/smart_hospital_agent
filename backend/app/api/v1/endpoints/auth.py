from fastapi import APIRouter

router = APIRouter()

@router.post("/login")
async def login():
    """
    用户登录接口 (Login Endpoint)
    Day 1 阶段的 Mock 实现。
    在真实场景中，这里应该接收用户名/密码，验证后返回 JWT Token。
    """
    return {"message": "Login successful", "token": "fake-jwt-token-for-day-1"}
