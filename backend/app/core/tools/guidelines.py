from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.rag.retriever import MedicalRetriever
import structlog

import structlog
import hashlib
import json
from app.core.config import settings
from app.core.llm.llm_factory import get_fast_llm
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.tools.medical_tools import get_retriever

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def lookup_guideline(query: str, top_k: int = 3) -> str:
    """
    检索医学指南或知识库 (Async)。
    
    Args:
        query: 用户的自然语言查询问题
        top_k: 返回的条目数量
        
    Returns:
        str: 格式化后的检索结果文本
    """
    import asyncio
    try:
        logger.info("tool_call.lookup_guideline", query=query)
        retriever = get_retriever()
        
        # [Async] Wrap sync retriever call
        loop = asyncio.get_running_loop()
        # [Isolation] 指定 diagnosis 意图，优先查病例库
        results = await loop.run_in_executor(None, lambda: retriever.search(query, top_k=top_k, intent="medical_diagnosis"))
        
        if not results:
            return "未找到相关指南信息。"
            
        # [L2 Cache] Context Compression & Caching
        redis = retriever.redis_client
        fast_llm = get_fast_llm() 
        processed_results = []
        
        for res in results:
            content = res.get("content", "")
            doc_id = str(res.get("id", "unknown"))
            
            # 1. 计算 Content Hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            summary_key = f"rag30:summary:{content_hash}"
            
            # 2. 查 Cache (Sync Redis for now, accept blocking for small reads or wrap)
            # 为了极致性能，建议 submit 这里的 IO 到线程池，但 Redis 读取通常微秒级
            # cached_summary = await loop.run_in_executor(None, redis.get, summary_key)
            cached_summary = redis.get(summary_key)
            
            if cached_summary:
                try:
                    res["content"] = json.loads(cached_summary)
                    res["source"] = f"{res.get('source','unknown')}_cached"
                    processed_results.append(res)
                    continue
                except:
                    pass
            
            # 3. Cache Miss - 执行压缩
            # 如果是 Encyclopedia (已经由 Retriever 截断)，直接存储
            # 如果是 Case (Augmented)，进行结构化提取
            new_content = content
            if "hua_" in doc_id or "[百科]" in content:
                # 已经是截断过的，无需再 LLM 处理，但可以进一步清洗
                pass
            else:
                # Case Data: 使用 LLM 提取关键信息
                # Graceful Degradation: Try/Except
                try:
                    prompt = f"""提取以下病例的关键信息(症状、诊断、治疗)，控制在100字以内。\n\n{content[:1000]}"""
                    # [Async] Use ainvoke
                    summary_resp = await fast_llm.ainvoke([HumanMessage(content=prompt)])
                    new_content = f"[病例摘要] {summary_resp.content}"
                except Exception as e:
                    logger.warning(f"Summary Generation Failed: {e}")
                    new_content = content[:300] + "..." # 降级为截断
            
            # 4. 回写 Cache (1小时过期)
            # redis.setex(summary_key, 3600, json.dumps(new_content))
            redis.setex(summary_key, 3600, json.dumps(new_content))
            
            res["content"] = new_content
            processed_results.append(res)
            
        results = processed_results
            
        # 构造给 LLM 看的文本视图
        llm_view_lines = []
        doc_details = []
        
        import json
        
        for i, res in enumerate(results):
            content = res.get("content", "")
            score = res.get("score", 0.0)
            metadata = res.get("metadata", {})
            
            view_text = f"【参考资料 {i+1}】(可信度: {score:.2f})\n{content}\n"
            llm_view_lines.append(view_text)
            
            # 记录详细信息供前端透视
            doc_details.append({
                "content": content,
                "metadata": metadata,
                "score": score
            })
            
        llm_text = "\n".join(llm_view_lines)
        
        # 返回复合结构 (hacky way to pass context)
        output_data = {
            "summary": llm_text, # 给 LLM 看的主体
            "_raw_docs": doc_details # 前端透视用 (隐藏字段)
        }
        
        return json.dumps(output_data, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_error.lookup_guideline", error=str(e))
        raise e
