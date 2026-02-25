from typing import Dict, Any, Optional
import time

class CostTracker:
    """
    [Pain Point #23] Resource Observability & Cost Audit
    Tracks token usage and estimates costs for various models.
    """
    
    # Pricing per 1M tokens (CNY/USD mixed, normalized to CNY for this project context if needed, 
    # but usually we keep original currency or normalized units)
    # Here assuming standard public cloud pricing (approximate)
    PRICING = {
        # Alibaba Qwen (Input/Output per 1M tokens in CNY)
        "qwen-turbo": {"input": 0.0, "output": 0.0}, # Free tier often
        "qwen-plus": {"input": 4.0, "output": 12.0},
        "qwen-max": {"input": 20.0, "output": 60.0},
        
        # OpenAI (Input/Output per 1M tokens in USD) -> Converted to CNY approx x7.2
        "gpt-4o": {"input": 5.0 * 7.2, "output": 15.0 * 7.2},
        "gpt-3.5-turbo": {"input": 0.5 * 7.2, "output": 1.5 * 7.2},
        
        # Local Models (Cost is 0, but we track Time/VRAM effectively)
        "local_slm": {"input": 0.0, "output": 0.0},
        "qwen3-0.6b": {"input": 0.0, "output": 0.0},
    }

    @staticmethod
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate estimated cost in CNY.
        """
        model_key = model_name.lower()
        # Fallback to qwen-plus if unknown cloud model, or local if it looks local
        if model_key not in CostTracker.PRICING:
            if "local" in model_key or "qwen3" in model_key:
                pricing = {"input": 0.0, "output": 0.0}
            else:
                pricing = CostTracker.PRICING["qwen-plus"] # Default fallback
        else:
            pricing = CostTracker.PRICING[model_key]
            
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)

    @staticmethod
    def create_usage_record(
        node_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        duration_s: float
    ) -> Dict[str, Any]:
        cost = CostTracker.calculate_cost(model_name, input_tokens, output_tokens)
        return {
            "timestamp": time.time(),
            "node": node_name,
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration_s": round(duration_s, 3),
            "estimated_cost_cny": cost
        }
