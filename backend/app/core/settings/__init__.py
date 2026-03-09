from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import base, feature_flags, model, observability, rag, storage


@dataclass(slots=True)
class SettingsFacade:
    settings_obj: Any

    def base(self) -> dict[str, Any]:
        return base.snapshot(self.settings_obj)

    def storage(self) -> dict[str, Any]:
        return storage.snapshot(self.settings_obj)

    def model(self) -> dict[str, Any]:
        return model.snapshot(self.settings_obj)

    def rag(self) -> dict[str, Any]:
        return rag.snapshot(self.settings_obj)

    def observability(self) -> dict[str, Any]:
        return observability.snapshot(self.settings_obj)

    def feature_flags(self) -> dict[str, Any]:
        return feature_flags.snapshot(self.settings_obj)

    def effective_snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            "base": self.base(),
            "storage": self.storage(),
            "model": self.model(),
            "rag": self.rag(),
            "observability": self.observability(),
            "feature_flags": self.feature_flags(),
        }


def get_settings_facade(settings_obj: Any | None = None) -> SettingsFacade:
    if settings_obj is None:
        from app.core.config import settings as current_settings

        settings_obj = current_settings
    return SettingsFacade(settings_obj=settings_obj)


__all__ = [
    "SettingsFacade",
    "get_settings_facade",
]
