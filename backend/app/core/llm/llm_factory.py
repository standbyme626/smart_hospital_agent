import asyncio
import logging
import uuid
import threading
import openai
import os
import json
from typing import Optional, List, Any, ClassVar, Dict, Union
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.core.llm.model_manager import model_manager
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult, Generation, ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

# 配置日志
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_base_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    value = value.rstrip("/")
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def _build_node_urls_from_env() -> List[str]:
    raw = os.getenv("DEEPSEEK_NODE_URLS", "")
    urls = [_normalize_base_url(x) for x in raw.split(",") if x.strip()]
    # De-duplicate while preserving order
    dedup: List[str] = []
    seen = set()
    for item in urls:
        if item and item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup

def _translate_openai_error(e: Exception) -> str:
    """将 OpenAI/HTTP 错误转换为易读的中文提示"""
    # 优先使用 status_code 属性
    if hasattr(e, "status_code"):
        code = e.status_code
        if code == 401: return f"认证失败 (401) - API Key 无效 ({e})"
        if code == 403: return f"拒绝访问 (403) - 账户余额不足或无权限 ({e})"
        if code == 404: return f"模型不存在 (404) - 请检查模型名称 ({e})"
        if code == 429: return f"请求过多 (429) - 触发并发/频率限制 ({e})"
        if code >= 500: return f"服务暂时不可用 ({code}) - 云厂商服务波动 ({e})"
        if code == 400: return f"请求参数错误 (400) - 上下文过长或参数格式错误 ({e})"

    err_str = str(e).lower()
    # 使用更严格的字符串匹配防止误判 (如 request_id 包含 403)
    if "error code: 401" in err_str or "authentication" in err_str:
        return f"认证失败 (401) - API Key 无效或过期 ({e})"
    if "error code: 403" in err_str:
        return f"拒绝访问 (403) - 账户余额不足或无权限 ({e})"
    if "error code: 429" in err_str or "rate limit" in err_str:
        return f"请求过多 (429) - 触发并发/频率限制 ({e})"
    if "error code: 404" in err_str or "not found" in err_str:
        return f"模型不存在 (404) - 请检查模型名称 ({e})"
    if "error code: 400" in err_str:
        return f"请求参数错误 (400) - 上下文过长或参数格式错误 ({e})"
    if "error code: 5" in err_str:
        return f"服务暂时不可用 (5xx) - 云厂商服务波动 ({e})"
        
    return f"未知错误: {e}"

# [V6.5.6] 强力锁定全局 OpenAI 配置，防止任何 SDK 默认行为流向 OpenAI 官网
target_url = os.getenv("OPENAI_API_BASE") or settings.OPENAI_API_BASE
if target_url:
    openai.base_url = target_url
    # 同时也设置环境变量，因为某些库 user-agent 只读环境变量
    os.environ["OPENAI_API_BASE"] = target_url
    os.environ["OPENAI_BASE_URL"] = target_url # [V6.5.8] Dual lock
    logger.info(f"🔒 [LLM] Global OpenAI Base URL locked to: {target_url}")

# [V6.5.9] Monkey Patching for Ultimate Safety (As per Reference)
# 拦截所有 OpenAI Client 的创建，强制注入阿里云配置
_original_openai_init = openai.OpenAI.__init__
_original_async_openai_init = openai.AsyncOpenAI.__init__

def _patched_openai_init(self, **kwargs):
    # 强制覆盖 base_url
    target = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # [V8.3] Allow localhost for Ollama/llama-server
    base_url = kwargs.get("base_url") or ""
    if "localhost" in str(base_url) or "127.0.0.1" in str(base_url):
            # Do not override if it's local
            pass
    elif not base_url or "openai.com" in str(base_url):
        kwargs["base_url"] = target
        # logger.debug(f"🛡️ [MonkeyPatch] Intercepted OpenAI client creation. Forced Base URL: {target}")
    
    # 确保 API Key 存在
    if not kwargs.get("api_key"):
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        
    _original_openai_init(self, **kwargs)

def _patched_async_openai_init(self, **kwargs):
    target = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # [V8.3] Allow localhost for Ollama/llama-server
    base_url = kwargs.get("base_url") or ""
    if "localhost" in str(base_url) or "127.0.0.1" in str(base_url):
            # Do not override if it's local
            pass
    elif not base_url or "openai.com" in str(base_url):
        kwargs["base_url"] = target
        # logger.debug(f"🛡️ [MonkeyPatch] Intercepted AsyncOpenAI client creation. Forced Base URL: {target}")

    if not kwargs.get("api_key"):
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY

    _original_async_openai_init(self, **kwargs)

openai.OpenAI.__init__ = _patched_openai_init
openai.AsyncOpenAI.__init__ = _patched_async_openai_init
logger.info("🛡️ [LLM] OpenAI Client constructors have been patched for safety.")

# [V6.6.0] Monkey Patch 增强版：同时拦截并修正模型名称
# 拦截所有 completion create 调用，确保模型名称正确
try:
    _original_chat_create = openai.resources.chat.completions.Completions.create
    _original_async_chat_create = openai.resources.chat.completions.AsyncCompletions.create

    def _patched_chat_create(self, *args, **kwargs):
        logger.info(f"🛡️ [MonkeyPatch-Sync] Intercepted create call for model: {kwargs.get('model')}")
        # 1. 修正模型名称
        if kwargs.get("model") == "smart-rotating-llm":
            # [V6.6.1] 动态从配置获取，不再硬编码 qwen-plus
            kwargs["model"] = settings.OPENAI_MODEL_NAME_DYN
            
        # [Fix] Do NOT override model if it is already set to a specific candidate by SmartRotatingLLM
        # logic: SmartRotatingLLM sets specific model name (e.g. qwen-plus). 
        # If we overwrite it with settings.OPENAI_MODEL_NAME_DYN every time, rotation fails.
        # kwargs["model"] = settings.OPENAI_MODEL_NAME_DYN  <-- This was the bug causing 0.0 score (repeated failures on same model)
        # [V6.6.4] DashScope 限制：非流式调用必须禁用 enable_thinking
        # [V8.5] 修复逻辑：如果是流式调用，不应该禁用，但要确保参数正确传递
        # [Fix] Remove enable_thinking if not streaming to avoid conflicts
        if not kwargs.get("stream", False):
            extra_body = kwargs.get("extra_body")
            if isinstance(extra_body, dict) and "enable_thinking" in extra_body:
                del extra_body["enable_thinking"]
                logger.info(f"🛡️ [MonkeyPatch-Sync] Removed enable_thinking for non-stream call.")
            kwargs["extra_body"] = extra_body
        
        try:
            # [Debug] Log Input Messages
            # valid_msgs = kwargs.get("messages", [])
            # if valid_msgs:
            #    logger.info(f"📤 [MonkeyPatch-Sync] Input Messages (Preview): {str(valid_msgs)[:500]}")

            response = _original_chat_create(self, *args, **kwargs)
            
            # [Debug] Log Response Content
            try:
                content = response.choices[0].message.content
                logger.info(f"📥 [MonkeyPatch-Sync] Response Content: {content}")
            except Exception:
                pass
                
            return response
        except Exception as e:
            logger.error(f"❌ [MonkeyPatch-Sync] Inner call failed: {e}")
            raise e

    async def _patched_async_chat_create(self, *args, **kwargs):
        logger.info(f"🛡️ [MonkeyPatch-Async] Intercepted create call for model: {kwargs.get('model')}")
        if kwargs.get("model") == "smart-rotating-llm":
            # [V6.6.1] 动态从配置获取，不再硬编码 qwen-plus
            kwargs["model"] = settings.OPENAI_MODEL_NAME_DYN
        # [V6.6.4] DashScope 限制：非流式调用必须禁用 enable_thinking
        # [Fix] Remove enable_thinking if not streaming to avoid conflicts
        if not kwargs.get("stream", False):
            extra_body = kwargs.get("extra_body")
            if isinstance(extra_body, dict) and "enable_thinking" in extra_body:
                del extra_body["enable_thinking"]
                logger.info(f"🛡️ [MonkeyPatch-Async] Removed enable_thinking for non-stream call.")
            kwargs["extra_body"] = extra_body
            
        try:
            return await _original_async_chat_create(self, *args, **kwargs)
        except Exception as e:
            friendly_error = _translate_openai_error(e)
            logger.error(f"🚨 [LLM_ERROR] Async Call Failed: {friendly_error}")
            raise e

    openai.resources.chat.completions.Completions.create = _patched_chat_create
    openai.resources.chat.completions.AsyncCompletions.create = _patched_async_chat_create
    logger.info("🛡️ [LLM] OpenAI Chat Completion methods have been patched for model name correction.")
except Exception as e:
    logger.error(f"❌ [MonkeyPatch] Failed to patch Completions.create: {e}")

from app.core.security.pii import PIIMasker # [Task 8.2]

class SmartRotatingLLM(BaseChatModel):
    """
    终极自愈 LLM 适配器。
    修复 DashScope 400 错误，增强 ToolMessage 处理。
    集成 PII 自动脱敏。
    """
    _BLACKLISTED_KEYS: ClassVar[set] = set() 
    _LOCAL_LOCK: ClassVar[asyncio.Lock] = asyncio.Lock() 
    _SYNC_LOCAL_LOCK: ClassVar[threading.Lock] = threading.Lock() 
    _NODE_RR_LOCK: ClassVar[threading.Lock] = threading.Lock()
    _NODE_RR_INDEX: ClassVar[int] = 0
    
    temperature: float = 0.0
    streaming: bool = False
    max_tokens: Optional[int] = None
    prefer_local: bool = False
    allow_local: bool = Field(default_factory=lambda: settings.ENABLE_LOCAL_FALLBACK) # [Config] Controlled by env
    model_name: str = Field(default_factory=lambda: settings.OPENAI_MODEL_NAME_DYN) # [V6.6.1] Dynamic from env
    stop: Optional[List[str]] = None 
    supports_stop_words: bool = True 
    enable_pii_masking: bool = True # [Task 8.2] Default enabled

    model_config = {
        "extra": "allow" 
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_quota_error(self, error: Exception) -> bool:
        """
        判断是否为额度耗尽/欠费错误 (403, 429)
        注意：区分“整个账号没钱/Key无效”和“特定模型Free Tier用完”。
        """
        error_str = str(error).lower()
        status_code = getattr(error, "status_code", None)
        
        # [Special Case] Aliyun "Free Tier of the model exhausted" -> 这是模型层面的错误，不是Key层面的（Key可能还能调其他模型）
        if "free tier of the model has been exhausted" in error_str:
            return False

        # 1. 明确的状态码检查
        if status_code in [429, 402]:
            return True
            
        # 2. 关键词匹配 (更严格)
        # 阿里云/DashScope 特有错误
        if "datainspectionfailed" in error_str:
             # 内容安全拦截，不应轮询 Key，但为了不崩溃先视为 True? 
             # 不，内容拦截换 Key 没用。应该直接抛出。
             # 这里只处理 Key 相关的
             return False
             
        if "out of capacity" in error_str: # Azure/OpenAI
            return True
            
        if "insufficient_quota" in error_str:
            return True
            
        if "billing" in error_str and "account" in error_str:
            return True
            
        # 避免 "404" 误判为 "403" (如果字符串里恰好有 403)
        # 使用正则匹配 "error code: 403" 
        import re
        if re.search(r"error.*code.*403", error_str):
            return True
            
        # [Fix] 之前的逻辑导致 404 被误判，如果 status_code 明确是 404，直接 False
        if status_code == 404:
            return False

        if status_code == 403:
            return True
            
        return False

    def _convert_dict_to_message(self, m: Any) -> BaseMessage:
        """内部辅助：将字典安全转换为 LangChain 消息对象"""
        if isinstance(m, BaseMessage):
            return m
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                return SystemMessage(content=content)
            elif role == "assistant":
                return AIMessage(content=content, tool_calls=m.get("tool_calls", []))
            elif role == "tool":
                return ToolMessage(content=content, tool_call_id=m.get("tool_call_id", ""))
            else:
                return HumanMessage(content=content)
        return HumanMessage(content=str(m))

    @classmethod
    def _ordered_deepseek_node_urls(cls) -> List[str]:
        if not _env_flag("DEEPSEEK_FALLBACK_ENABLED", False):
            return []

        urls = _build_node_urls_from_env()
        if not urls:
            return []

        strategy = (os.getenv("DEEPSEEK_LB_STRATEGY", "round_robin") or "round_robin").strip().lower()
        if strategy != "round_robin":
            return urls

        with cls._NODE_RR_LOCK:
            start = cls._NODE_RR_INDEX % len(urls)
            cls._NODE_RR_INDEX += 1
        return urls[start:] + urls[:start]

    @staticmethod
    def _node_model_name() -> str:
        return (os.getenv("DEEPSEEK_NODE_MODEL", "deepseek-r1:8b") or "deepseek-r1:8b").strip()

    @staticmethod
    def _apply_llm_env_options(params: Dict[str, Any]) -> Dict[str, Any]:
        out = params.copy()
        extra_body = out.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}

        # Unified think switch from env (default off).
        extra_body.setdefault("think", _env_flag("LLM_THINK_ENABLED", False))

        # Keep a stable context window; configurable via env.
        options = extra_body.get("options")
        if not isinstance(options, dict):
            options = {}
        try:
            options.setdefault("num_ctx", int(os.getenv("LLM_NUM_CTX", "4096")))
        except Exception:
            options.setdefault("num_ctx", 4096)
        extra_body["options"] = options

        out["extra_body"] = extra_body
        return out

    def call(self, messages: List[Any], **kwargs: Any) -> str:
        """兼容 CrewAI 的 call 方法"""
        logger.info(f"🎤 [LLM-Call] CrewAI invoked call() with {len(messages)} messages.")
        try:
            api_keys = settings.API_KEY_CANDIDATES or []
            active_keys = [k for k in api_keys if k and k not in self._BLACKLISTED_KEYS and k != "sk-no-key-configured"]
            
            if not active_keys:
                if self.allow_local:
                    logger.warning("⚠️ [LLM] (Call) No active API keys. Falling back to Local.")
                    return self._call_local(messages, **kwargs)
                else:
                    raise RuntimeError("No available LLM keys and Local fallback disabled for this task.")

            lc_messages = [self._convert_dict_to_message(m) for m in messages]
            result = self._generate(lc_messages, **kwargs)
            content = result.generations[0].message.content
            logger.info(f"✅ [LLM-Call] Returning content (len={len(content)}): {content[:50]}...")
            return content

        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["401", "authentication", "invalid_api_key", "unauthorized"]):
                if self.allow_local:
                    logger.error(f"❌ [LLM] (Call) Auth error: {e}. Forcing local fallback.")
                    return self._call_local(messages, **kwargs)
                else:
                    raise RuntimeError(f"Auth error: {e} and Local fallback disabled for this task.")
            raise e

    def _call_local(self, messages: List[Any], **kwargs: Any) -> str:
        """内部辅助：执行本地 call 调用"""
        with self._SYNC_LOCAL_LOCK:
            local_llm = get_enhanced_local_llm(self.temperature, self.max_tokens)
            params = kwargs.copy()
            for k in ["callbacks", "run_manager"]: params.pop(k, None)
            res = local_llm.invoke(messages, **params)
            return res.content

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        logger.info(f"🎤 [LLM-Invoke] invoke() called.")
        messages = input if isinstance(input, list) else [input]
        lc_messages = [self._convert_dict_to_message(m) for m in messages]
        res = self._generate(lc_messages, **kwargs)
        return res.generations[0].message

    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        messages = input if isinstance(input, list) else [input]
        lc_messages = [self._convert_dict_to_message(m) for m in messages]
        res = await self._agenerate(lc_messages, **kwargs)
        return res.generations[0].message

    def _normalize_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """归一化逻辑：彻底解决 DashScope 400 Field Required 错误"""
        normalized = []
        for msg in messages:
            if isinstance(msg, SystemMessage): role = "system"
            elif isinstance(msg, AIMessage): role = "assistant"
            elif isinstance(msg, ToolMessage): role = "tool"
            elif isinstance(msg, HumanMessage): role = "user"
            else: role = getattr(msg, "role", "user")
            
            raw_content = getattr(msg, "content", None)
            if raw_content is None and hasattr(msg, "dict"):
                 raw_content = msg.dict().get("content")
            
            if raw_content is None: content = " "
            else:
                content = str(raw_content)
                if not content.strip(): content = " "
            
            m_dict = {"role": role, "content": content}

            if role == "assistant":
                tool_calls = getattr(msg, "tool_calls", [])
                if not tool_calls:
                    additional_kwargs = getattr(msg, "additional_kwargs", {})
                    tool_calls = additional_kwargs.get("tool_calls", [])
                
                if tool_calls:
                    m_dict_tool_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id")
                            func = tc.get("function", {})
                            name = tc.get("name") or func.get("name")
                            args = tc.get("args") or tc.get("arguments") or func.get("arguments")
                        else:
                            tc_id = getattr(tc, "id", None)
                            name = getattr(tc, "name", None)
                            args = getattr(tc, "args", {})

                        if not tc_id: tc_id = f"call_{uuid.uuid4().hex[:12]}"
                        if isinstance(args, dict): args_str = json.dumps(args, ensure_ascii=False)
                        else: args_str = str(args) if args else "{}"

                        m_dict_tool_calls.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args_str}
                        })
                    m_dict["tool_calls"] = m_dict_tool_calls

            if role == "tool":
                tool_call_id = getattr(msg, "tool_call_id", None)
                if not tool_call_id and hasattr(msg, "dict"):
                    tool_call_id = msg.dict().get("tool_call_id")
                if not tool_call_id: tool_call_id = "unknown_call_id"
                m_dict["tool_call_id"] = tool_call_id

            normalized.append(m_dict)
        return normalized

    async def _retry_with_fixed_content(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        logger.warning("🔄 [LLM] Triggering strict content repair for DashScope compatibility...")
        fixed_messages = []
        for msg in messages:
            content = getattr(msg, "content", " ")
            if not content or str(content).strip() == "": content = " "
            
            if isinstance(msg, AIMessage):
                new_msg = AIMessage(
                    content=content,
                    tool_calls=getattr(msg, "tool_calls", []),
                    additional_kwargs=getattr(msg, "additional_kwargs", {}),
                    id=getattr(msg, "id", None)
                )
                fixed_messages.append(new_msg)
            elif isinstance(msg, ToolMessage):
                new_msg = ToolMessage(content=content, tool_call_id=getattr(msg, "tool_call_id", "unknown"))
                fixed_messages.append(new_msg)
            elif isinstance(msg, SystemMessage): fixed_messages.append(SystemMessage(content=content))
            else: fixed_messages.append(HumanMessage(content=content))
        
        return await self._agenerate(fixed_messages, stop=stop, is_retry=True, **kwargs)

    def _convert_to_chat_result(self, openai_response) -> ChatResult:
        generations = []
        for choice in openai_response.choices:
            message = choice.message

            # Normalize provider-specific tool payloads to plain dicts
            function_call_raw = getattr(message, "function_call", None)
            function_call_safe = None
            if function_call_raw:
                if isinstance(function_call_raw, dict):
                    function_call_safe = function_call_raw
                elif hasattr(function_call_raw, "model_dump"):
                    function_call_safe = function_call_raw.model_dump()
                elif hasattr(function_call_raw, "dict"):
                    function_call_safe = function_call_raw.dict()
                else:
                    function_call_safe = {"value": str(function_call_raw)}

            raw_tool_calls = getattr(message, "tool_calls", None) or []
            additional_tool_calls = []
            normalized_tool_calls = []
            for tool_call in raw_tool_calls:
                tc_id = getattr(tool_call, "id", None)
                tc_type = getattr(tool_call, "type", "function")
                fn = getattr(tool_call, "function", None)
                fn_name = getattr(fn, "name", None) if fn is not None else None
                fn_args = getattr(fn, "arguments", None) if fn is not None else None

                if isinstance(fn_args, dict):
                    fn_args_json = json.dumps(fn_args, ensure_ascii=False)
                    args_dict = fn_args
                elif isinstance(fn_args, str):
                    fn_args_json = fn_args
                    try:
                        args_dict = json.loads(fn_args)
                    except Exception:
                        args_dict = {}
                else:
                    fn_args_json = "{}"
                    args_dict = {}

                additional_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": tc_type,
                        "function": {
                            "name": fn_name,
                            "arguments": fn_args_json,
                        },
                    }
                )

                normalized_tool_calls.append(
                    {
                        "name": fn_name,
                        "args": args_dict,
                        "id": tc_id,
                        "type": "tool_call",
                    }
                )

            additional_kwargs = {}
            if function_call_safe is not None:
                additional_kwargs["function_call"] = function_call_safe
            if additional_tool_calls:
                additional_kwargs["tool_calls"] = additional_tool_calls

            ai_message_kwargs = {
                "content": message.content or " ",
                "additional_kwargs": additional_kwargs,
            }
            if normalized_tool_calls:
                ai_message_kwargs["tool_calls"] = normalized_tool_calls
            
            ai_message = AIMessage(**ai_message_kwargs)
            generation = ChatGeneration(message=ai_message, generation_info={"finish_reason": choice.finish_reason})
            generations.append(generation)
        
        token_usage = getattr(openai_response, "usage", {})
        if hasattr(token_usage, "dict"): token_usage = token_usage.dict()
        elif hasattr(token_usage, "model_dump"): token_usage = token_usage.model_dump()
            
        return ChatResult(generations=generations, llm_output={
            "token_usage": token_usage,
            "model_name": openai_response.model
        })

    def _try_deepseek_nodes_sync(
        self,
        normalized_messages: List[Dict[str, Any]],
        stop: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[ChatResult]:
        urls = self._ordered_deepseek_node_urls()
        if not urls:
            return None

        model_name = self._node_model_name()
        node_params = self._apply_llm_env_options(params)
        api_key = settings.OPENAI_API_KEY or "sk-node-fallback"
        last_error: Optional[Exception] = None

        for node_url in urls:
            try:
                logger.warning("🔁 [LLM-Node] Sync fallback attempt: %s model=%s", node_url, model_name)
                client = openai.OpenAI(api_key=api_key, base_url=node_url)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=normalized_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop,
                    stream=False,
                    **node_params,
                )
                logger.info("✅ [LLM-Node] Sync fallback success: %s", node_url)
                return self._convert_to_chat_result(completion)
            except Exception as e:
                last_error = e
                logger.warning("⚠️ [LLM-Node] Sync fallback failed: %s error=%s", node_url, e)

        if last_error is not None:
            logger.error("❌ [LLM-Node] All sync node fallbacks failed: %s", last_error)
        return None

    async def _try_deepseek_nodes_async(
        self,
        normalized_messages: List[Dict[str, Any]],
        stop: Optional[List[str]],
        params: Dict[str, Any],
    ) -> Optional[ChatResult]:
        urls = self._ordered_deepseek_node_urls()
        if not urls:
            return None

        model_name = self._node_model_name()
        node_params = self._apply_llm_env_options(params)
        api_key = settings.OPENAI_API_KEY or "sk-node-fallback"
        last_error: Optional[Exception] = None

        for node_url in urls:
            try:
                logger.warning("🔁 [LLM-Node] Async fallback attempt: %s model=%s", node_url, model_name)
                client = openai.AsyncOpenAI(api_key=api_key, base_url=node_url)
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=normalized_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop,
                    stream=False,
                    **node_params,
                )
                logger.info("✅ [LLM-Node] Async fallback success: %s", node_url)
                return self._convert_to_chat_result(completion)
            except Exception as e:
                last_error = e
                logger.warning("⚠️ [LLM-Node] Async fallback failed: %s error=%s", node_url, e)

        if last_error is not None:
            logger.error("❌ [LLM-Node] All async node fallbacks failed: %s", last_error)
        return None

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Any:
        """Bind tool definitions to this chat model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, **kwargs)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        is_retry = kwargs.pop("is_retry", False)
        if not messages: return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[System Error] No messages provided."))])
        
        params = kwargs.copy()
        for k in ["callbacks", "run_manager"]: params.pop(k, None)

        # [V6.6.3] 尊重 prefer_local 标志，直接进入本地路径以节省云端超时等待
        if self.prefer_local and self.allow_local:
            logger.info("🚀 [LLM] Prefer Local is enabled. Skipping cloud attempts.")
            return self._call_local_as_result(messages, stop, **params)
        
        # [Task 8.2] PII Masking
        if self.enable_pii_masking:
            messages = PIIMasker.mask_messages(messages)

        normalized_messages = self._normalize_messages(messages)
        api_keys = settings.API_KEY_CANDIDATES or []
        
        # [V8.4] 支持多模型轮询
        # 如果当前 model_name 不在候选列表中，则将其作为首选
        models = settings.MODEL_CANDIDATES_LIST
        if self.model_name not in models:
             models = [self.model_name] + models
        
        # models = model_manager.get_all_candidates() or []
        # if self.model_name in models: models = [self.model_name] + [m for m in models if m != self.model_name]

        cloud_primary_enabled = _env_flag("CLOUD_PRIMARY_ENABLED", True)
        if cloud_primary_enabled:
            for k_idx, api_key in enumerate(api_keys):
                if not api_key or api_key in self._BLACKLISTED_KEYS:
                    continue
                for model_name in models:  # Inner loop: Try all models with this key
                    try:
                        logger.info(f"🔄 [LLM-Sync] Attempting model: {model_name} with key suffix: ...{api_key[-4:]}")
                        target_base_url = os.getenv("OPENAI_API_BASE") or settings.OPENAI_API_BASE
                        client = openai.OpenAI(api_key=api_key, base_url=target_base_url)
                        call_params = self._apply_llm_env_options(params)

                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=normalized_messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            stop=stop,
                            stream=False,
                            **call_params,  # Pass extra params like tools
                        )
                        res = self._convert_to_chat_result(completion)
                        logger.info(f"✅ [LLM-Sync] Generated: {res.generations[0].message.content[:100]}...")
                        return res
                    except Exception as e:
                        err_str = str(e).lower()

                        # [Fallback Strategy]
                        # Check status code first
                        status_code = getattr(e, "status_code", None)
                        is_key_error = False

                        if status_code in [401, 403]:
                            is_key_error = True
                        elif any(x in err_str for x in ["authentication", "quota", "exhausted", "limitexceeded"]):
                            is_key_error = True
                        elif "error code: 403" in err_str or "error code: 401" in err_str:
                            is_key_error = True

                        # 1. 401/403/Quota -> Key Issue -> Break inner loop to switch Key
                        if is_key_error:
                            friendly_error = _translate_openai_error(e)
                            logger.error(f"❌ [LLM-Sync] Key ...{api_key[-4:]} 额度耗尽/失效: {friendly_error}. 拉黑该 Key.")
                            self._BLACKLISTED_KEYS.add(api_key)
                            break

                        # 2. 400/404/429/5xx -> Model Issue or Rate Limit -> Continue to next Model
                        is_model_error = False
                        if status_code in [400, 404, 429, 500, 502, 503]:
                            is_model_error = True
                        elif any(x in err_str for x in ["model", "found", "access", "permission"]):
                            is_model_error = True
                        elif "error code: 400" in err_str or "error code: 404" in err_str or "error code: 429" in err_str:
                            is_model_error = True
                        # [Special Case] Free Tier Exhausted is a model error (try next model), not a key error
                        elif "free tier of the model has been exhausted" in err_str:
                            is_model_error = True

                        if is_model_error:
                            friendly_error = _translate_openai_error(e)
                            logger.warning(f"⚠️ [LLM-Sync] 模型 {model_name} 调用失败: {friendly_error}. 正在尝试下一个模型...")
                            continue

                        # 3. Other errors -> Log and continue (Aggressive Fallback)
                        logger.error(f"❌ [LLM-Sync] 未知错误 {model_name}: {e}. 正在尝试下一个模型...")
                        continue
        else:
            logger.warning("⏭️ [LLM-Sync] CLOUD_PRIMARY_ENABLED=false, skipping cloud provider.")

        # Cloud failed or disabled -> try DeepSeek node pool
        node_result = self._try_deepseek_nodes_sync(normalized_messages, stop, params)
        if node_result is not None:
            return node_result
        
        if not self.allow_local:
            # logger.critical("🛑 [LLM] All cloud options failed and Local fallback is DISABLED.")
            # raise RuntimeError("No available LLM keys and Local fallback disabled for this task.")
            logger.warning("⚠️ [LLM] All cloud options failed. Falling back to MOCK for testing.")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[MOCK] Cloud LLM failed. This is a mock response for testing."))])
        return self._call_local_as_result(messages, stop, **params)

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        is_retry = kwargs.pop("is_retry", False)
        if not messages: return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[Error] No messages."))])
        
        params = kwargs.copy()
        for k in ["callbacks", "run_manager"]: params.pop(k, None)

        # [V6.6.3] 尊重 prefer_local 标志，直接进入本地路径
        if self.prefer_local and self.allow_local:
            logger.info("🚀 [LLM] Prefer Local (Async) is enabled. Skipping cloud attempts.")
            return await self._call_local_async_as_result(messages, stop, is_fallback=False, **params)
        
        # [Task 8.2] PII Masking
        if self.enable_pii_masking:
            messages = PIIMasker.mask_messages(messages)

        normalized_messages = self._normalize_messages(messages)
        api_keys = settings.API_KEY_CANDIDATES or []
        active_keys = [k for k in api_keys if k and k not in self._BLACKLISTED_KEYS and k != "sk-no-key-configured"]
        
        # [V8.4] 支持多模型轮询
        models = settings.MODEL_CANDIDATES_LIST
        if self.model_name not in models:
             models = [self.model_name] + models
             
        # models = model_manager.get_all_candidates() or []
        # if self.model_name in models: models = [self.model_name] + [m for m in models if m != self.model_name]

        cloud_primary_enabled = _env_flag("CLOUD_PRIMARY_ENABLED", True)
        if cloud_primary_enabled:
            for k_idx, api_key in enumerate(api_keys):
                if not api_key or api_key in self._BLACKLISTED_KEYS:
                    continue
                for model_name in models:
                    try:
                        logger.info(f"🔄 [LLM-Async] Attempting model: {model_name} with key suffix: ...{api_key[-4:]}")
                        async_client = openai.AsyncOpenAI(
                            api_key=api_key,
                            base_url=os.getenv("OPENAI_API_BASE") or settings.OPENAI_API_BASE,
                        )
                        call_params = self._apply_llm_env_options(params)

                        completion = await async_client.chat.completions.create(
                            model=model_name,
                            messages=normalized_messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            stop=stop,
                            stream=False,
                            **call_params,  # Pass extra params like tools
                        )
                        return self._convert_to_chat_result(completion)
                    except Exception as e:
                        err_str = str(e).lower()
                        if "400" in err_str and not is_retry:
                            # Try to fix content first for 400 errors (often format issues)
                            try:
                                return await self._retry_with_fixed_content(messages, stop=stop, **kwargs)
                            except Exception:
                                pass  # If fix fails, proceed to fallback logic below

                        # [Fallback Strategy]
                        # Check status code first
                        status_code = getattr(e, "status_code", None)

                        # 1. 401/403/Quota -> Key Issue -> Break inner loop to switch Key
                        # [Refactored] Use _is_quota_error method for consistent checking
                        is_key_error = False
                        if status_code in [401]:
                            is_key_error = True
                        elif self._is_quota_error(e):
                            is_key_error = True
                        elif "error code: 401" in err_str:
                            is_key_error = True

                        if is_key_error:
                            friendly_error = _translate_openai_error(e)
                            logger.error(f"❌ [LLM-Async] Key ...{api_key[-4:]} 额度耗尽/失效: {friendly_error}. 拉黑该 Key.")
                            self._BLACKLISTED_KEYS.add(api_key)
                            break

                        # 2. 400/404/429/5xx -> Model Issue or Rate Limit -> Continue to next Model
                        # Also fallback for 404/5xx or specific keywords
                        is_model_error = False
                        if status_code in [400, 404, 429, 500, 502, 503]:
                            is_model_error = True
                        elif any(x in err_str for x in ["model", "found", "access", "permission"]):
                            is_model_error = True
                        elif "error code: 400" in err_str or "error code: 404" in err_str or "error code: 429" in err_str:
                            is_model_error = True
                        # [Special Case] Free Tier Exhausted is a model error (try next model), not a key error
                        elif "free tier of the model has been exhausted" in err_str:
                            is_model_error = True

                        if is_model_error:
                            friendly_error = _translate_openai_error(e)
                            logger.warning(f"⚠️ [LLM-Async] 模型 {model_name} 调用失败: {friendly_error}. 正在尝试下一个模型...")
                            continue

                        # 3. Other errors -> Log and continue
                        logger.error(f"❌ [LLM-Async] 未知错误 {model_name}: {e}. 正在尝试下一个模型...")
                        continue
        else:
            logger.warning("⏭️ [LLM-Async] CLOUD_PRIMARY_ENABLED=false, skipping cloud provider.")

        # Cloud failed or disabled -> try DeepSeek node pool
        node_result = await self._try_deepseek_nodes_async(normalized_messages, stop, params)
        if node_result is not None:
            return node_result
        
        if not self.allow_local:
            # logger.critical("🛑 [LLM] All cloud options failed and Local fallback is DISABLED.")
            # raise RuntimeError("No available LLM keys and Local fallback disabled for this task.")
            logger.warning("⚠️ [LLM-Async] All cloud options failed. Falling back to MOCK for testing.")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[MOCK] Cloud LLM failed. This is a mock response for testing."))])
        return await self._call_local_async_as_result(messages, stop, **params)

    def _call_local_as_result(self, messages, stop, **params) -> ChatResult:
        with self._SYNC_LOCAL_LOCK:
            local_llm = get_enhanced_local_llm(self.temperature, self.max_tokens)
            gen_kwargs = params.copy()
            if stop: gen_kwargs["stop"] = stop
            res_msg = local_llm.invoke(messages, **gen_kwargs)
            return ChatResult(generations=[ChatGeneration(message=res_msg)])

    async def _call_local_async_as_result(self, messages, stop, is_fallback: bool = True, **params) -> ChatResult:
        if not self.allow_local:
             raise RuntimeError("❌ [Security] Local LLM usage is explicitly disabled by configuration.")

        async with self._LOCAL_LOCK:
            if is_fallback:
                logger.error("🚨 [LLM] All cloud options failed. Falling back to Local.")
                # [Fix] Insert warning system message for user visibility
                warning_msg = AIMessage(content="【⚠️ 系统提示】\n检测到云端专家服务（DashScope）暂时不可用（API 额度耗尽或连接失败）。\n系统已自动切换至本地应急模型（Local Qwen3-0.6B）为您提供基础服务。\n请注意：本地模型仅具备基础对话能力，无法进行复杂的专家会诊。\n\n")
                return ChatResult(generations=[ChatGeneration(message=warning_msg)])
            else:
                logger.info("🤖 [LLM] Executing Local Model (Preferred Mode).")
            
            local_llm = get_enhanced_local_llm(self.temperature, self.max_tokens)
            gen_kwargs = params.copy()
            if stop: gen_kwargs["stop"] = stop
            res_msg = await local_llm.ainvoke(messages, **gen_kwargs)
            return ChatResult(generations=[ChatGeneration(message=res_msg)])

    @property
    def _llm_type(self) -> str:
        return "smart-rotating-llm"

def get_enhanced_local_llm(temperature: float = 0.3, max_tokens: Optional[int] = None):
    from app.core.models.local_slm import local_slm
    from app.core.models.slm_adapter import LocalSLMAdapter
    return LocalSLMAdapter(service=local_slm, temperature=temperature, max_tokens=max_tokens)

def get_fast_llm(temperature: float = 0.0, task_type: str = "general", **kwargs):
    """
    获取一个高性能、低延迟的 LLM 实例（优先云端，允许本地回退）。
    支持通过 kwargs 传递 prefer_local 等参数。
    """
    prefer_local = kwargs.pop("prefer_local", False)
    allow_local = kwargs.pop("allow_local", settings.ENABLE_LOCAL_FALLBACK) # [Config] Default from settings
    
    # Temperature is env-controlled first; fallback to call argument.
    try:
        effective_temperature = float(os.getenv("TEMPERATURE", str(temperature)))
    except Exception:
        effective_temperature = float(temperature)

    return SmartRotatingLLM(
        temperature=effective_temperature,
        prefer_local=prefer_local,
        allow_local=allow_local,
        model_name=settings.OPENAI_MODEL_NAME_DYN,
        **kwargs
    )

def get_smart_llm(temperature: float = 0.7, **kwargs):
    """
    获取一个更具创造性、更“聪明”的 LLM 实例。
    """
    return get_fast_llm(temperature=temperature, **kwargs)

def get_judge_llm():
    """获取专门用于评分/判断的 LLM（固定 0 温度）"""
    return get_fast_llm(temperature=0.0, task_type="judgment")
