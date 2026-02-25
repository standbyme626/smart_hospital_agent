from typing import Dict, Optional, List
from datetime import datetime

# 模拟数据库 - 后期替换为 SQL 查询
MOCK_USERS_DB = {
    "1001": {
        "user_id": "1001",
        "name": "张伟",
        "age": 35,
        "gender": "男",
        "phone": "13800138000",
        "insurance_type": "public",  # 医保
        "past_medical_history": ["高血压", "青霉素过敏"]
    },
    "1002": {
        "user_id": "1002",
        "name": "李娜",
        "age": 28,
        "gender": "女",
        "phone": "13900139000",
        "insurance_type": "commercial", # 商保
        "past_medical_history": ["无"]
    },
    "1003": {
        "user_id": "1003",
        "name": "王强",
        "age": 65,
        "gender": "男",
        "phone": "13700137000",
        "insurance_type": "public",
        "past_medical_history": ["糖尿病", "冠心病史"]
    }
}

class AuthService:
    """
    [Phase 1] 身份认证服务 (Mock Implementation)
    负责用户登录、信息获取。未来将对接真实 Auth/HIS 接口。
    """
    
    def login(self, user_id: str) -> Optional[Dict]:
        """
        模拟登录，返回用户概要信息。
        """
        if user_id in MOCK_USERS_DB:
            return MOCK_USERS_DB[user_id]
        return None

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """
        获取完整的用户画像。
        """
        # 在这个简单的 mock 中，login 返回的就是 profile
        # 在真实系统中，login 可能只返回 token，这里再查详情
        return MOCK_USERS_DB.get(user_id)

    def create_guest_user(self) -> Dict:
        """
        创建临时访客用户
        """
        return {
            "user_id": f"guest_{int(datetime.now().timestamp())}",
            "name": "访客用户",
            "age": 0, # 未知
            "gender": "未知",
            "phone": "",
            "insurance_type": "none",
            "past_medical_history": []
        }

# 单例模式
auth_service = AuthService()
