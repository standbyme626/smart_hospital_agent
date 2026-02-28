from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import structlog
from langchain_core.tools import tool
from pydantic import BaseModel

from app.core.config import settings

logger = structlog.get_logger(__name__)


class SlotInfo(BaseModel):
    slot_id: str
    doctor: str
    time: str
    fee: float
    date: str
    status: str


class LegacyHISMCPBridge:
    """
    Legacy HIS protocol simulator behind an MCP-like bridge.

    This class simulates a fragile old HIS endpoint style while keeping a stable
    modern Python call interface for tools.
    """

    SUCCESS = "00"
    NOT_FOUND = "04"
    CONFLICT = "05"
    INVALID_SESSION = "06"
    BAD_REQUEST = "12"

    def __init__(self):
        self._slots = self._build_seed_slots()
        self._orders: Dict[str, Dict] = {}
        self._sessions: Dict[str, datetime] = {}

    @staticmethod
    def _build_seed_slots() -> Dict[str, List[Dict]]:
        base_day = datetime.now() + timedelta(days=1)
        date_str = base_day.strftime("%Y-%m-%d")
        return {
            "dept_001": [
                {
                    "slot_id": "s_001",
                    "doctor": "张医生",
                    "time": "09:00",
                    "fee": 50,
                    "date": date_str,
                    "status": "available",
                },
                {
                    "slot_id": "s_002",
                    "doctor": "王医生",
                    "time": "10:00",
                    "fee": 50,
                    "date": date_str,
                    "status": "available",
                },
            ],
            "dept_002": [
                {
                    "slot_id": "s_003",
                    "doctor": "李医生",
                    "time": "09:30",
                    "fee": 80,
                    "date": date_str,
                    "status": "available",
                }
            ],
        }

    def _encode_request(self, action: str, payload: Dict, session_id: str = "") -> str:
        """
        Simulate old framed protocol:
        HIS|v1|ACTION|SESSION|<json payload>
        """
        return f"HIS|v1|{action}|{session_id}|{json.dumps(payload, ensure_ascii=False)}"

    def _decode_request(self, frame: str) -> Tuple[str, str, Dict]:
        parts = frame.split("|", 4)
        if len(parts) != 5 or parts[0] != "HIS":
            raise ValueError("invalid_legacy_frame")
        action = parts[2]
        session_id = parts[3]
        payload = json.loads(parts[4]) if parts[4] else {}
        return action, session_id, payload

    def _encode_response(self, code: str, message: str, payload: Dict | List | None = None) -> str:
        body = {"code": code, "message": message, "payload": payload or {}}
        return json.dumps(body, ensure_ascii=False)

    @staticmethod
    def _decode_response(frame: str) -> Dict:
        return json.loads(frame)

    def _is_session_valid(self, session_id: str) -> bool:
        if not session_id:
            return False
        expire_at = self._sessions.get(session_id)
        if not expire_at:
            return False
        if datetime.now() >= expire_at:
            self._sessions.pop(session_id, None)
            return False
        return True

    def _auth(self, payload: Dict) -> str:
        client_id = str(payload.get("client_id", "")).strip()
        if not client_id:
            return self._encode_response(self.BAD_REQUEST, "missing_client_id")
        session_id = secrets.token_hex(12)
        self._sessions[session_id] = datetime.now() + timedelta(minutes=30)
        return self._encode_response(self.SUCCESS, "ok", {"session_id": session_id})

    def _query_slots(self, payload: Dict) -> str:
        dept_id = str(payload.get("dept_id", "")).strip()
        if not dept_id:
            return self._encode_response(self.BAD_REQUEST, "missing_dept_id")
        slots = self._slots.get(dept_id, [])
        available = [s for s in slots if s["status"] == "available"]
        return self._encode_response(self.SUCCESS, "ok", {"slots": available})

    def _lock_slot(self, payload: Dict) -> str:
        slot_id = str(payload.get("slot_id", "")).strip()
        patient_id = str(payload.get("patient_id", "")).strip()
        if not slot_id or not patient_id:
            return self._encode_response(self.BAD_REQUEST, "missing_slot_or_patient")

        target_slot: Optional[Dict] = None
        for slots in self._slots.values():
            for slot in slots:
                if slot["slot_id"] == slot_id:
                    target_slot = slot
                    break
            if target_slot:
                break

        if not target_slot:
            return self._encode_response(self.NOT_FOUND, "slot_not_found")
        if target_slot["status"] != "available":
            return self._encode_response(self.CONFLICT, "slot_conflicted")

        target_slot["status"] = "locked"
        order_id = f"ord_{int(datetime.now().timestamp())}"
        self._orders[order_id] = {
            "order_id": order_id,
            "slot_id": slot_id,
            "patient_id": patient_id,
            "status": "pending_payment",
            "amount": target_slot["fee"],
            "created_at": datetime.now().isoformat(),
        }
        return self._encode_response(
            self.SUCCESS,
            "ok",
            {
                "order_id": order_id,
                "slot_info": target_slot,
                "payment_required": target_slot["fee"],
            },
        )

    def _confirm_appointment(self, payload: Dict) -> str:
        order_id = str(payload.get("order_id", "")).strip()
        if not order_id:
            return self._encode_response(self.BAD_REQUEST, "missing_order_id")
        order = self._orders.get(order_id)
        if not order:
            return self._encode_response(self.NOT_FOUND, "order_not_found")

        order["status"] = "confirmed"
        order["payment_time"] = datetime.now().isoformat()
        logger.info(
            "legacy_his_confirmed",
            order_id=order_id,
            patient_id=order["patient_id"],
            amount=order["amount"],
        )
        return self._encode_response(
            self.SUCCESS,
            "ok",
            {
                "order_id": order_id,
                "meet_location": "门诊 A 楼 302 室",
                "time": "请提前 15 分钟候诊",
            },
        )

    def _dispatch_legacy(self, frame: str) -> str:
        action, session_id, payload = self._decode_request(frame)

        if action == "AUTH":
            return self._auth(payload)

        if not self._is_session_valid(session_id):
            return self._encode_response(self.INVALID_SESSION, "session_invalid")

        if action == "QUERY_SLOTS":
            return self._query_slots(payload)
        if action == "LOCK_SLOT":
            return self._lock_slot(payload)
        if action == "CONFIRM_APPOINTMENT":
            return self._confirm_appointment(payload)
        return self._encode_response(self.BAD_REQUEST, "unknown_action")

    def call(self, action: str, payload: Dict, session_id: str = "") -> Dict:
        """
        MCP bridge entrypoint: encode legacy request, dispatch, decode response.
        """
        req = self._encode_request(action=action, payload=payload, session_id=session_id)
        resp = self._dispatch_legacy(req)
        data = self._decode_response(resp)
        return data

    def auth(self, client_id: str = "smart_hospital_agent") -> Dict:
        return self.call("AUTH", {"client_id": client_id})


class HISService:
    """
    HIS service facade used by LangGraph tools.

    Modes:
    - legacy_sim: route through legacy MCP bridge (default)
    - mock_direct: legacy-compatible direct mode, still returns same contract
    """

    _bridge = LegacyHISMCPBridge()
    _session_id: Optional[str] = None

    @classmethod
    def _direct_call(cls, action: str, payload: Dict) -> Dict:
        """
        Direct in-process mock mode (no legacy session framing).
        Keeps the same response schema as legacy bridge.
        """
        if action == "QUERY_SLOTS":
            dept_id = str(payload.get("dept_id", "")).strip()
            slots = cls._bridge._slots.get(dept_id, [])
            available = [s for s in slots if s["status"] == "available"]
            return {"code": LegacyHISMCPBridge.SUCCESS, "message": "ok", "payload": {"slots": available}}

        if action == "LOCK_SLOT":
            return json.loads(cls._bridge._lock_slot(payload))

        if action == "CONFIRM_APPOINTMENT":
            return json.loads(cls._bridge._confirm_appointment(payload))

        return {"code": LegacyHISMCPBridge.BAD_REQUEST, "message": "unknown_action", "payload": {}}

    @classmethod
    def _ensure_session(cls) -> Optional[str]:
        if settings.HIS_MCP_MODE != "legacy_sim":
            return None
        if cls._session_id:
            # quick probe via a harmless call; if invalid, re-auth
            probe = cls._bridge.call("QUERY_SLOTS", {"dept_id": "dept_001"}, session_id=cls._session_id)
            if probe.get("code") != LegacyHISMCPBridge.INVALID_SESSION:
                return cls._session_id

        auth_res = cls._bridge.auth()
        if auth_res.get("code") == LegacyHISMCPBridge.SUCCESS:
            cls._session_id = auth_res.get("payload", {}).get("session_id")
            return cls._session_id
        logger.error("his_auth_failed", response=auth_res)
        return None

    @classmethod
    def _legacy_call(cls, action: str, payload: Dict) -> Dict:
        if settings.HIS_MCP_MODE == "mock_direct":
            return cls._direct_call(action, payload)

        sid = cls._ensure_session()
        res = cls._bridge.call(action, payload, session_id=sid or "")
        # auto-retry once on invalid session
        if res.get("code") == LegacyHISMCPBridge.INVALID_SESSION:
            cls._session_id = None
            sid = cls._ensure_session()
            res = cls._bridge.call(action, payload, session_id=sid or "")
        return res

    @staticmethod
    def _as_error(message: str) -> Dict:
        return {"status": "error", "message": message}

    @staticmethod
    @tool
    def get_department_slots(dept_id: str) -> List[Dict]:
        """
        查询指定科室可用号源。

        Compatible return for service subgraph:
        [{slot_id, doctor, time, fee, date, status}, ...]
        """
        res = HISService._legacy_call("QUERY_SLOTS", {"dept_id": dept_id})
        if res.get("code") != LegacyHISMCPBridge.SUCCESS:
            logger.warning("his_query_slots_failed", dept_id=dept_id, response=res)
            return []
        return list(res.get("payload", {}).get("slots", []))

    @staticmethod
    @tool
    def lock_slot(slot_id: str, patient_id: str) -> Dict:
        """
        锁定号源，返回待支付订单。
        """
        res = HISService._legacy_call("LOCK_SLOT", {"slot_id": slot_id, "patient_id": patient_id})
        if res.get("code") != LegacyHISMCPBridge.SUCCESS:
            return HISService._as_error(str(res.get("message", "lock_failed")))
        payload = res.get("payload", {})
        return {
            "status": "success",
            "order_id": payload.get("order_id"),
            "slot_info": payload.get("slot_info", {}),
            "payment_required": payload.get("payment_required"),
        }

    @staticmethod
    @tool
    def confirm_appointment(order_id: str) -> Dict:
        """
        确认支付后完成预约。
        """
        res = HISService._legacy_call("CONFIRM_APPOINTMENT", {"order_id": order_id})
        if res.get("code") != LegacyHISMCPBridge.SUCCESS:
            return HISService._as_error(str(res.get("message", "confirm_failed")))
        payload = res.get("payload", {})
        return {"status": "success", "message": "预约确认成功", "details": payload}


his_tools = [HISService.get_department_slots, HISService.lock_slot, HISService.confirm_appointment]
