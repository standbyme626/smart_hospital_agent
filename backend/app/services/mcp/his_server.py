from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# 模拟数据 (Mock Data)
# 注意：在生产环境中，这将被真实的 HIS 数据库或通过 MCP 协议连接的远程服务取代。
# 本文件作为 MCP Server 的参考实现 (Reference Implementation)。

MOCK_SLOTS = {
    "dept_001": [  # 心内科 (Internal Medicine / Cardiology)
        {"slot_id": "s_001", "doctor": "张医生", "time": "09:00", "fee": 50, "date": "2026-02-09", "status": "available"},
        {"slot_id": "s_002", "doctor": "王医生", "time": "10:00", "fee": 50, "date": "2026-02-09", "status": "available"},
    ],
    "dept_002": [  # 外科 (Surgery)
        {"slot_id": "s_003", "doctor": "李医生", "time": "09:30", "fee": 80, "date": "2026-02-09", "status": "available"},
    ]
}

MOCK_ORDERS = {}

class SlotInfo(BaseModel):
    slot_id: str
    doctor: str
    time: str
    fee: float
    date: str
    status: str

class HISService:
    """
    模拟 HIS (医院信息系统) 的 MCP 服务端。
    Simulates an MCP Server for HIS.
    """
    
    @staticmethod
    @tool
    def get_department_slots(dept_id: str) -> List[Dict]:
        """
        查询指定科室的可用号源。
        
        Args:
            dept_id: 科室 ID (例如 'dept_001' 代表心内科, 'dept_002' 代表外科)。
        """
        # 在真实的 MCP Server 中，这里会查询数据库
        slots = MOCK_SLOTS.get(dept_id, [])
        return [s for s in slots if s["status"] == "available"]

    @staticmethod
    @tool
    def lock_slot(slot_id: str, patient_id: str) -> Dict:
        """
        临时锁定号源以便支付。
        
        Args:
            slot_id: 号源 ID。
            patient_id: 患者 ID。
        """
        # 在所有科室中查找号源
        target_slot = None
        dept_found = None
        
        for dept, slots in MOCK_SLOTS.items():
            for s in slots:
                if s["slot_id"] == slot_id:
                    target_slot = s
                    dept_found = dept
                    break
            if target_slot:
                break
        
        if not target_slot:
            return {"status": "error", "message": "未找到号源"}
        
        if target_slot["status"] != "available":
            return {"status": "error", "message": "号源已被占用"}
        
        # 锁定号源
        target_slot["status"] = "locked"
        order_id = f"ord_{int(datetime.now().timestamp())}"
        
        MOCK_ORDERS[order_id] = {
            "order_id": order_id,
            "slot_id": slot_id,
            "patient_id": patient_id,
            "status": "pending_payment",
            "amount": target_slot["fee"],
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "status": "success", 
            "order_id": order_id, 
            "slot_info": target_slot,
            "payment_required": target_slot["fee"]
        }

    @staticmethod
    @tool
    def confirm_appointment(order_id: str) -> Dict:
        """
        支付完成后确认预约。
        
        Args:
            order_id: 锁号时返回的订单 ID。
        """
        order = MOCK_ORDERS.get(order_id)
        if not order:
            return {"status": "error", "message": "订单不存在"}
        
        if order["status"] == "confirmed":
             return {"status": "success", "message": "订单已确认", "details": order}
             
        # 更新状态
        order["status"] = "confirmed"
        order["payment_time"] = datetime.now().isoformat()
        
        # 审计日志 (Anti-Lazy: Audit logging)
        print(f"[AUDIT] 预约确认: Order={order_id}, Patient={order['patient_id']}, Amount={order['amount']}")
        
        return {
            "status": "success", 
            "message": "预约确认成功", 
            "details": {
                "order_id": order_id,
                "meet_location": "门诊 A 楼 302 室", # 模拟地址
                "time": "请提前 15 分钟候诊"
            }
        }

his_tools = [HISService.get_department_slots, HISService.lock_slot, HISService.confirm_appointment]
