from app.domain.states.agent_state import UserProfile

class AuthService:
    async def mock_get_user_profile(self, user_id: str) -> UserProfile:
        """
        Mock implementation of user profile retrieval from HIS.
        """
        if user_id == "test_001":
            return UserProfile(
                patient_id="test_001",
                name="张三",
                age=45,
                gender="男",
                medical_history=["高血压", "糖尿病"],
                allergies=["青霉素"],
                identity_verified=True
            )
        else:
            # Anonymous or new user
            return UserProfile(
                patient_id=user_id,
                name="匿名用户",
                age=0,
                gender="未知",
                medical_history=[],
                allergies=[],
                identity_verified=False
            )
