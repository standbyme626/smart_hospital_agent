from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

REPORT_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def resolve_report_image(settings: Any, image_path: str | None) -> Path | None:
    """Resolve a stored report image path, rejecting anything outside UPLOADS_DIR."""
    if not image_path:
        return None
    candidate = Path(image_path)
    if candidate.suffix.lower() not in REPORT_IMAGE_SUFFIXES:
        return None
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(Path(settings.UPLOADS_DIR).resolve()):
        return None
    return resolved


def has_report_image(settings: Any, image_path: str | None) -> bool:
    path = resolve_report_image(settings, image_path)
    return bool(path and path.is_file())


def report_image_media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def remove_report_image(settings: Any, image_path: str | None) -> None:
    path = resolve_report_image(settings, image_path)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
