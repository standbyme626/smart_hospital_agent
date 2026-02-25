from typing import Any, Dict, Optional

class AppException(Exception):
    """
    Base class for all application exceptions.
    Ensures that all raised errors have a consistent structure.
    """
    def __init__(
        self,
        code: int = 400,
        slug: str = "bad_request",
        msg: str = "Bad Request",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.slug = slug
        self.msg = msg
        self.details = details or {}
        super().__init__(self.msg)

class ResourceNotFoundException(AppException):
    def __init__(self, msg: str = "Resource not found", details: dict = None):
        super().__init__(
            code=404,
            slug="resource_not_found",
            msg=msg,
            details=details
        )

class ValidationException(AppException):
    def __init__(self, msg: str = "Validation failed", details: dict = None):
        super().__init__(
            code=422,
            slug="validation_error",
            msg=msg,
            details=details
        )

class AuthenticationException(AppException):
    def __init__(self, msg: str = "Authentication failed", details: dict = None):
        super().__init__(
            code=401,
            slug="authentication_failed",
            msg=msg,
            details=details
        )

class SystemException(AppException):
    """Critical system failures (DB down, Model missing)"""
    def __init__(self, msg: str = "Internal System Error", details: dict = None):
        super().__init__(
            code=500,
            slug="system_error",
            msg=msg,
            details=details
        )
