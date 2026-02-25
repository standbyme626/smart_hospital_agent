"""
Legacy compatibility module.

The canonical streaming endpoint is defined in `app.api.v1.endpoints.chat`.
This module intentionally exposes an empty router to avoid accidental
duplicate route registration.
"""

from fastapi import APIRouter

router = APIRouter()

