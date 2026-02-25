import asyncio
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamingAudit")

class StreamingAuditInterceptor:
    """
    [Phase 5.2] 流式增量审计拦截器
    核心逻辑：
    1. 维护一个滑动窗口缓存（Sliding Window Buffer）。
    2. 每次有新 Token 产生时，将其加入缓存。
    3. 在缓存中实时检测敏感词、违禁语、虚假诊断等。
    4. 一旦命中拦截规则，立即抛出异常中断生成，防止不合规内容输出给用户。
    """
    
    def __init__(self, window_size: int = 50):
        self.buffer = ""
        self.full_content = "" # 保留全量内容用于状态追踪
        self.window_size = window_size
        self.in_thinking_block = False # 是否处于 DPO 模型的思考块中
        # 预定义高风险词库
        self.blacklist = [
            r"自杀", r"农药", r"必死", # 心理健康风险
            r"保证治愈", r"包治百病", r"神药", # 医疗欺诈
            r"内部渠道", r"私下交易", # 违规导流
            r"傻逼", r"垃圾", # 辱骂
            r"TEST_BLOCK", # 专门用于测试的拦截词
        ]
        self.patterns = [re.compile(p) for p in self.blacklist]

    async def __call__(self, chunk: str):
        """流式回调接口，增加对 <think> 标签的适配"""
        if not chunk:
            return
            
        self.buffer += chunk
        self.full_content += chunk
        
        # 状态追踪：检测是否进入或退出思考块
        if "<think>" in chunk:
            self.in_thinking_block = True
            logger.info("🔍 [Audit] Entered thinking block, relaxing rules...")
        if "</think>" in chunk:
            self.in_thinking_block = False
            logger.info("✅ [Audit] Exited thinking block, enforcing strict rules...")

        # 保持滑动窗口大小
        if len(self.buffer) > self.window_size * 2:
            self.buffer = self.buffer[-self.window_size:]
            
        # 实时审计逻辑
        if not self.in_thinking_block:
            # 仅在非思考块（即输出给用户的文本）中执行严格拦截
            for pattern in self.patterns:
                if pattern.search(self.buffer):
                    logger.error(f"🚨 [Audit] Content intercepted! Rule matched: {pattern.pattern}")
                    raise ValueError(f"Content security violation: {pattern.pattern}")
        else:
            # 在思考块中，我们只记录日志，不抛出异常中断
            for pattern in self.patterns:
                if pattern.search(self.buffer):
                    logger.warning(f"⚠️ [Audit] Sensitive word in <think>: {pattern.pattern} (Allowed in reasoning)")

async def test_streaming_audit():
    auditor = StreamingAuditInterceptor(window_size=20)
    
    # 模拟正常的流
    print("Testing normal stream...")
    try:
        await auditor("你好")
        await auditor("，我是")
        await auditor("智能医疗助手。")
        print("✅ Normal stream passed.")
    except Exception as e:
        print(f"❌ Unexpected interception: {e}")

    # 模拟违规流
    print("\nTesting blocked stream...")
    try:
        await auditor("这个药")
        await auditor("简直是")
        await auditor("神药")
        await auditor("，包治百病。")
        print("❌ Failed to intercept!")
    except ValueError as e:
        print(f"✅ Intercepted successfully: {e}")

if __name__ == "__main__":
    asyncio.run(test_streaming_audit())
