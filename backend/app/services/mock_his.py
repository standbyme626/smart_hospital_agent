from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timedelta
import random

# 数据模型定义 (简单字典模拟)
# Department: { "id": str, "name": str }
# Doctor: { "id": str, "name": str, "department_id": str, "title": str, "price": float }
# TimeSlot: { "id": str, "doctor_id": str, "start_time": str, "end_time": str, "is_booked": bool }
# Order: { "id": str, "patient_id": str, "slot_id": str, "amount": float, "status": str, "created_at": str }

class MockHISService:
    """
    模拟医院信息系统 (Hospital Information System)
    提供科室、医生、排班、预约、支付的核心业务数据支持。
    """
    
    def __init__(self):
        self._init_db()

    def _init_db(self):
        """初始化 Mock 数据"""
        
        # 1. 科室
        self.departments = [
            {"id": "dept_001", "name": "心血管内科"},
            {"id": "dept_002", "name": "神经内科"},
            {"id": "dept_003", "name": "呼吸内科"},
            {"id": "dept_004", "name": "骨科"},
            {"id": "dept_005", "name": "普通内科"},
        ]
        
        # 2. 医生 (随机生成一些)
        self.doctors = []
        titles = ["主任医师", "副主任医师", "主治医师"]
        prices = {"主任医师": 100.0, "副主任医师": 50.0, "主治医师": 20.0}
        
        doctor_names = {
            "dept_001": ["张心安", "李脉搏"],
            "dept_002": ["王脑清", "赵神经"],
            "dept_003": ["刘呼吸", "陈肺"],
            "dept_004": ["孙骨", "周关节"],
            "dept_005": ["吴全科", "郑通用"]
        }
        
        for dept_id, names in doctor_names.items():
            for name in names:
                title = random.choice(titles)
                self.doctors.append({
                    "id": f"doc_{uuid.uuid4().hex[:6]}",
                    "name": name,
                    "department_id": dept_id,
                    "title": title,
                    "price": prices[title]
                })

        # 3. 排班 (生成未来3天的排班)
        self.slots = []
        now = datetime.now()
        for doc in self.doctors:
            for day_offset in range(1, 4): # 未来3天
                date_base = now + timedelta(days=day_offset)
                # 上午 9:00 - 11:00
                for hour in range(9, 12):
                    self.slots.append({
                        "id": f"slot_{uuid.uuid4().hex[:8]}",
                        "doctor_id": doc["id"],
                        "start_time": date_base.replace(hour=hour, minute=0, second=0).isoformat(),
                        "end_time": date_base.replace(hour=hour+1, minute=0, second=0).isoformat(),
                        "is_booked": random.random() < 0.2 # 20% 概率已被预约
                    })

        # 4. 订单 (空)
        self.orders = {}

    # --- 查询接口 ---

    def get_departments(self) -> List[Dict]:
        return self.departments

    def get_doctors_by_department(self, department_name_or_id: str) -> List[Dict]:
        """根据科室ID或名称模糊查找医生"""
        target_id = None
        # 尝试匹配 ID
        for dept in self.departments:
            if dept["id"] == department_name_or_id:
                target_id = dept["id"]
                break
        
        # 尝试匹配 Name
        if not target_id:
            for dept in self.departments:
                if department_name_or_id in dept["name"]:
                    target_id = dept["id"]
                    break
        
        if not target_id:
            return []
            
        return [doc for doc in self.doctors if doc["department_id"] == target_id]

    def get_available_slots(self, doctor_id: str) -> List[Dict]:
        """获取某位医生的可用排班"""
        return [
            s for s in self.slots 
            if s["doctor_id"] == doctor_id and not s["is_booked"]
        ]
        
    def get_doctor_info(self, doctor_id: str) -> Optional[Dict]:
        for doc in self.doctors:
            if doc["id"] == doctor_id:
                return doc
        return None
        
    def get_slot_info(self, slot_id: str) -> Optional[Dict]:
        for slot in self.slots:
            if slot["id"] == slot_id:
                return slot
        return None

    # --- 交易接口 ---

    def create_booking_order(self, patient_id: str, slot_id: str) -> Dict[str, Any]:
        """创建预约订单"""
        slot = self.get_slot_info(slot_id)
        if not slot:
            return {"success": False, "message": "排班不存在"}
        if slot["is_booked"]:
            return {"success": False, "message": "该时段已被预约"}
            
        doctor = self.get_doctor_info(slot["doctor_id"])
        
        order_id = f"ord_{uuid.uuid4().hex[:10]}"
        order = {
            "id": order_id,
            "patient_id": patient_id,
            "slot_id": slot_id,
            "doctor_name": doctor["name"],
            "amount": doctor["price"],
            "status": "PENDING_PAYMENT", # PENDING_PAYMENT, PAID, CANCELLED
            "created_at": datetime.now().isoformat()
        }
        
        # 锁定排班 (简单处理：创建订单即锁定，实际应有超时释放机制)
        slot["is_booked"] = True
        self.orders[order_id] = order
        
        return {"success": True, "order": order}

    def pay_order(self, order_id: str) -> Dict[str, Any]:
        """支付订单"""
        order = self.orders.get(order_id)
        if not order:
            return {"success": False, "message": "订单不存在"}
            
        if order["status"] == "PAID":
            return {"success": True, "message": "订单已支付"}
            
        # 模拟支付成功
        order["status"] = "PAID"
        return {"success": True, "order": order}

# 单例实例
mock_his_service = MockHISService()
