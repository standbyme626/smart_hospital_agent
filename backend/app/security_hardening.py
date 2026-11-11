"""安全加固：入口 IP 白名单、分档限流、审计抽查工具函数。

设计原则：
- **默认不改变现有行为**：白名单留空 = 不限制；分档值为 0 = 回退全局限流。
- **失败关闭**：白名单一旦配置，来源 IP 解析不出来或不匹配就拒绝（403），
  避免"配置写错了反而全放开"。
- 所有判定集中在本模块，中间件只做编排，便于单测。
"""
from __future__ import annotations

import ipaddress
from typing import Any

CHAT_PATHS = ("/api/v1/chat/stream",)
ADMIN_PATHS = ("/api/v1/admin", "/api/v1/ops", "/api/v1/eval/run")


def parse_networks(raw: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """把 "1.2.3.4, 10.0.0.0/8" 解析成网段列表；非法项直接抛 ValueError（配置错误要早暴露）。"""
    networks: list[Any] = []
    for chunk in (raw or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        networks.append(ipaddress.ip_network(item, strict=False))
    return networks


def client_ip(request: Any, trust_proxy: bool) -> str:
    """取真实来源 IP。信任反代头时优先 X-Real-IP / X-Forwarded-For 第一段。"""
    if trust_proxy:
        real = (request.headers.get("x-real-ip") or "").strip()
        if real:
            return real
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def ip_allowed(ip: str, networks: list[Any]) -> bool:
    """白名单判定：解析失败按不通过处理（fail-closed）。"""
    if not networks:
        return True
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(address in network for network in networks)


def deny_all_networks() -> list[Any]:
    """配置非法时的兜底：只允许一个不可能命中的网段（等价于全拒绝）。"""
    return [ipaddress.ip_network("255.255.255.255/32")]


def rate_tier(path: str) -> tuple[str, str]:
    """返回 (限流桶名, 分档 key)。chat / admin / general 三档。"""
    if any(path.startswith(prefix) for prefix in CHAT_PATHS):
        return ("chat", "chat")
    if any(path.startswith(prefix) for prefix in ADMIN_PATHS):
        return ("admin", "admin")
    return ("general", "general")


def tier_limit(settings: Any, tier: str) -> int:
    """分档阈值：0 或未配置时回退全局 RATE_LIMIT_PER_MINUTE。"""
    global_limit = int(getattr(settings, "RATE_LIMIT_PER_MINUTE", 60) or 60)
    mapping = {
        "chat": getattr(settings, "RATE_LIMIT_CHAT_PER_MINUTE", 0),
        "admin": getattr(settings, "RATE_LIMIT_ADMIN_PER_MINUTE", 0),
    }
    value = int(mapping.get(tier, 0) or 0)
    return value if value > 0 else global_limit
