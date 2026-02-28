from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class LangfuseBridge:
    """Best-effort Langfuse bridge with no-op fallback when SDK/config is missing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._client: Any = None
        self._enabled = False
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._init_client()

    def _init_client(self) -> None:
        if not bool(getattr(settings, "LANGFUSE_ENABLED", False)):
            return
        public_key = str(getattr(settings, "LANGFUSE_PUBLIC_KEY", "") or "").strip()
        secret_key = str(getattr(settings, "LANGFUSE_SECRET_KEY", "") or "").strip()
        if not public_key or not secret_key:
            logger.warning("langfuse_disabled_missing_keys")
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=str(getattr(settings, "LANGFUSE_HOST", "http://127.0.0.1:3000") or "").strip(),
                environment=str(getattr(settings, "LANGFUSE_ENVIRONMENT", "dev") or "").strip(),
            )
            self._enabled = True
            logger.info("langfuse_enabled")
        except Exception as exc:
            logger.warning("langfuse_sdk_unavailable", error=str(exc))
            self._client = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def ensure_trace(
        self,
        *,
        request_id: str,
        session_id: str = "",
        user_intent: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        trace_id = str(request_id or "").strip() or f"req-{uuid.uuid4().hex}"
        with self._lock:
            if trace_id in self._traces:
                return trace_id

            trace_obj: Any = None
            if self.enabled:
                try:
                    trace_obj = self._client.trace(
                        id=trace_id,
                        name="chat_request",
                        session_id=session_id or None,
                        metadata=metadata or {},
                        input={
                            "request_id": trace_id,
                            "intent": user_intent or "",
                        },
                    )
                except Exception as exc:
                    logger.warning("langfuse_trace_create_failed", request_id=trace_id, error=str(exc))

            self._traces[trace_id] = {
                "trace": trace_obj,
                "spans": {},
                "started_at": time.time(),
                "metadata": dict(metadata or {}),
            }
            return trace_id

    def annotate_trace(self, request_id: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not metadata:
            return
        trace_id = str(request_id or "").strip()
        if not trace_id:
            return
        with self._lock:
            entry = self._traces.get(trace_id)
            if not entry:
                entry = {"trace": None, "spans": {}, "started_at": time.time(), "metadata": {}}
                self._traces[trace_id] = entry
            entry_meta = entry.get("metadata")
            if not isinstance(entry_meta, dict):
                entry_meta = {}
            entry_meta.update(metadata)
            entry["metadata"] = entry_meta

            trace_obj = entry.get("trace")
            if self.enabled and trace_obj is not None:
                try:
                    trace_obj.update(metadata=entry_meta)
                except Exception as exc:
                    logger.warning("langfuse_trace_update_failed", request_id=trace_id, error=str(exc))

    def start_span(
        self,
        *,
        request_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        trace_id = str(request_id or "").strip()
        if not trace_id:
            return ""
        span_id = f"span-{uuid.uuid4().hex}"
        with self._lock:
            entry = self._traces.get(trace_id)
            if not entry:
                self.ensure_trace(request_id=trace_id)
                entry = self._traces.get(trace_id, {"trace": None, "spans": {}})

            span_obj: Any = None
            trace_obj = entry.get("trace")
            if self.enabled and trace_obj is not None:
                try:
                    span_obj = trace_obj.span(
                        id=span_id,
                        name=name,
                        metadata=metadata or {},
                        input=input_data or {},
                    )
                except Exception as exc:
                    logger.warning(
                        "langfuse_span_create_failed",
                        request_id=trace_id,
                        span=name,
                        error=str(exc),
                    )

            spans = entry.get("spans")
            if not isinstance(spans, dict):
                spans = {}
            spans[span_id] = {
                "name": name,
                "span": span_obj,
                "started_at": time.time(),
                "metadata": dict(metadata or {}),
            }
            entry["spans"] = spans
            self._traces[trace_id] = entry
        return span_id

    def end_span(
        self,
        *,
        request_id: str,
        span_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
    ) -> None:
        trace_id = str(request_id or "").strip()
        if not trace_id or not span_id:
            return
        with self._lock:
            entry = self._traces.get(trace_id)
            if not entry:
                return
            spans = entry.get("spans")
            if not isinstance(spans, dict):
                return
            span_entry = spans.pop(span_id, None)
            if not isinstance(span_entry, dict):
                return
            span_obj = span_entry.get("span")
            if self.enabled and span_obj is not None:
                try:
                    kwargs: Dict[str, Any] = {}
                    if metadata:
                        kwargs["metadata"] = metadata
                    if output:
                        kwargs["output"] = output
                    span_obj.end(**kwargs)
                except Exception as exc:
                    logger.warning(
                        "langfuse_span_end_failed",
                        request_id=trace_id,
                        span_id=span_id,
                        error=str(exc),
                    )

    def finish_trace(
        self,
        *,
        request_id: str,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        trace_id = str(request_id or "").strip()
        if not trace_id:
            return
        with self._lock:
            entry = self._traces.pop(trace_id, None)
        if not entry:
            return

        trace_obj = entry.get("trace")
        merged_meta: Dict[str, Any] = {}
        raw_meta = entry.get("metadata")
        if isinstance(raw_meta, dict):
            merged_meta.update(raw_meta)
        if metadata:
            merged_meta.update(metadata)
        if self.enabled and trace_obj is not None:
            try:
                trace_obj.update(output=output or {}, metadata=merged_meta)
                self._client.flush()
            except Exception as exc:
                logger.warning("langfuse_trace_finish_failed", request_id=trace_id, error=str(exc))


langfuse_bridge = LangfuseBridge()
