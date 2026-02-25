import functools
import asyncio
import concurrent.futures
from typing import Callable, Any, Dict, Coroutine

def with_timeout(func: Callable, timeout_seconds: float = 30.0) -> Callable:
    """
    Decorator/Wrapper to enforce a timeout on a function execution.
    Supports both sync and async functions.
    
    Args:
        func: The function to wrap (node).
        timeout_seconds: Timeout in seconds.

    Returns:
        Wrapped function.
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                print(f"Node {func.__name__} timed out after {timeout_seconds}s")
                return {"messages": [], "diagnosis_report": "System timeout.", "status": "timeout"}
        return wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # For sync functions in LangGraph, we rely on internal logic or thread pool.
            # But here, to enforce strict node timeout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    print(f"Node {func.__name__} timed out after {timeout_seconds}s")
                    return {"messages": [], "diagnosis_report": "System timeout.", "status": "timeout"}
        return wrapper
