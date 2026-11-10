from __future__ import annotations

import os
from dataclasses import dataclass, replace

VALID_MODES = ("mock", "openhis")
VALID_PAYMENT_MODES = ("cashier", "direct")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    mode: str = "mock"
    openhis_base_url: str = "http://127.0.0.1:18080/openhis"
    openhis_username: str = ""
    openhis_password: str = ""
    openhis_tenant_id: str = ""
    readonly_dsn: str = ""
    database_url: str = ""
    timeout_seconds: float = 10.0
    poll_interval: int = 30
    payment_mode: str = "cashier"
    quote_ttl_seconds: int = 900
    task_sla_minutes: int = 30
    id_card_key: str = ""
    host: str = "127.0.0.1"
    port: int = 9101

    @classmethod
    def from_env(cls) -> AdapterConfig:
        mode = (_env("HIS_ADAPTER_MODE", "mock") or "mock").lower()
        if mode not in VALID_MODES:
            raise ValueError(f"HIS_ADAPTER_MODE must be one of {VALID_MODES}, got {mode!r}")
        payment_mode = (_env("HIS_PAYMENT_MODE", "cashier") or "cashier").lower()
        if payment_mode not in VALID_PAYMENT_MODES:
            raise ValueError(f"HIS_PAYMENT_MODE must be one of {VALID_PAYMENT_MODES}")
        try:
            timeout = float(_env("OPENHIS_TIMEOUT_SECONDS", "10") or "10")
        except ValueError:
            timeout = 10.0
        return cls(
            mode=mode,
            openhis_base_url=_env("OPENHIS_BASE_URL", "http://127.0.0.1:18080/openhis"),
            openhis_username=_env("OPENHIS_USERNAME"),
            openhis_password=_env("OPENHIS_PASSWORD"),
            openhis_tenant_id=_env("OPENHIS_TENANT_ID"),
            readonly_dsn=_env("OPENHIS_RO_DSN"),
            database_url=_env("ADAPTER_DATABASE_URL"),
            timeout_seconds=timeout,
            poll_interval=int(_env("POLL_INTERVAL", "30") or "30"),
            payment_mode=payment_mode,
            quote_ttl_seconds=int(_env("HIS_QUOTE_TTL_SECONDS", "900") or "900"),
            task_sla_minutes=int(_env("HIS_TASK_SLA_MINUTES", "30") or "30"),
            id_card_key=_env("ID_CARD_KEY") or _env("PSEUDONYM_KEY"),
            host=_env("ADAPTER_HOST", "127.0.0.1") or "127.0.0.1",
            port=int(_env("ADAPTER_PORT", "9101") or "9101"),
        )

    def with_(self, **changes: object) -> AdapterConfig:
        return replace(self, **changes)
