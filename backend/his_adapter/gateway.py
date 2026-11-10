from __future__ import annotations

from typing import Any, Protocol

import httpx

from .config import AdapterConfig
from .his_rest import OpenHISGateway
from .mock_backend import MockHIS
from .readonly import ReadOnlyPool


class Gateway(Protocol):
    name: str

    async def close(self) -> None: ...

    async def search_patients(
        self,
        *,
        id_card: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        birth_date: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def create_patient(self, patient: dict[str, Any]) -> dict[str, Any]: ...

    async def catalog(
        self,
        *,
        department: str | None = None,
        org_id: str | None = None,
        practitioner_id: str | None = None,
        service_id: str | None = None,
        day: str | None = None,
    ) -> dict[str, Any]: ...

    async def create_quote(self, request: dict[str, Any]) -> dict[str, Any]: ...

    async def pay_registration(
        self, quote: dict[str, Any], payment_details: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]: ...

    async def get_encounter(self, encounter_id: str) -> dict[str, Any]: ...

    async def find_encounter(
        self,
        *,
        patient_id: str,
        org_id: str | None = None,
        service_id: str | None = None,
        day: str | None = None,
        created_after: str | None = None,
    ) -> dict[str, Any] | None: ...

    async def payment_status(self, encounter_id: str) -> dict[str, Any]: ...

    async def list_prescriptions(
        self, encounter_id: str, include_details: bool = False
    ) -> list[dict[str, Any]]: ...

    async def cancel_registration(
        self, encounter_id: str, reason: str = "user_cancelled"
    ) -> dict[str, Any]: ...

    async def return_registration(
        self,
        encounter_id: str,
        reason: str = "user_cancelled",
        refund: bool = True,
        patient_id: str | None = None,
        amount: float | None = None,
        payment_id: str | None = None,
    ) -> dict[str, Any]: ...


def build_gateway(
    config: AdapterConfig,
    client: httpx.AsyncClient | None = None,
    readonly: ReadOnlyPool | None = None,
) -> Gateway:
    if config.mode == "openhis":
        return OpenHISGateway(config, client=client, readonly=readonly)
    return MockHIS()
