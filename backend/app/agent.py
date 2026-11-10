from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncGenerator
from typing import Any

from .config import Settings
from .conversation import trim_history
from .model_gateway import ChatResult, ModelGateway
from .policy import (
    apply_urgency_floor,
    crisis_protocol,
    normalize_department,
    rule_triage,
    safety_gate,
    split_member_prefix,
)
from .prompts import AGENT_SYSTEM, TRIAGE_SYSTEM
from .retrieval import RetrievalService
from .session_store import SessionStore
from .tools import ToolContext, ToolRegistry

URGENCIES = {"normal", "urgent", "critical"}
INTENTS = {"MEDICAL_CONSULT", "CRISIS", "INFO", "GREETING", "REGISTRATION"}
MIN_CITATION_SCORE = float(os.getenv("MIN_CITATION_SCORE", "0.2"))


def apply_evidence_state(
    report: dict[str, Any],
    citations: list[dict[str, Any]],
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Keep only relevant retrieved citations and never auto-fill report citations."""
    threshold = MIN_CITATION_SCORE if min_score is None else min_score
    valid = [item for item in citations if float(item.get("score") or 0) >= threshold]
    valid_ids = {str(item.get("id")) for item in valid}
    raw = [str(item) for item in (report.get("citations") or [])]
    kept: list[str] = []
    for citation_id in raw:
        if citation_id in valid_ids and citation_id not in kept:
            kept.append(citation_id)
    if not kept:
        state = "partial" if valid else "none"
    elif len(kept) != len(raw):
        state = "partial"
    else:
        state = "supported"
    report["citations"] = kept
    report["evidence_state"] = state
    return valid


class MedicalAgent:
    def __init__(
        self,
        settings: Settings,
        gateway: ModelGateway,
        retrieval: RetrievalService,
        tools: ToolRegistry,
        sessions: SessionStore,
    ):
        self.settings = settings
        self.gateway = gateway
        self.retrieval = retrieval
        self.tools = tools
        self.sessions = sessions

    async def stream(
        self,
        *,
        session_id: str,
        message: str,
        user_id: str = "anonymous",
        request_id: str,
        title: str | None = None,
        member_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        member_context: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self.sessions.ensure(session_id, user_id, title=(title or message)[:40], member_id=member_id)
        legacy_context, current_message = split_member_prefix(message)
        background = member_context or legacy_context or None
        yield await self._emit(session_id, "status", {"stage": "triage", "detail": "正在分诊"})

        triage = await self._triage(
            current_message,
            request_id,
            session_id,
            history=history,
            user_id=user_id,
            member_context=background,
        )
        triage_model = triage.pop("_model_meta", None)
        yield await self._emit(session_id, "status", {"stage": "triage_done", "detail": triage})
        yield await self._emit(
            session_id,
            "thought",
            {
                "text": f"推荐科室：{triage['department']}；紧急程度：{triage['urgency']}。",
                "department": triage["department"],
            },
        )

        if triage["intent"] == "CRISIS" or triage["urgency"] == "critical":
            async for event in self._crisis_flow(
                session_id, current_message, triage, request_id, user_id, model_meta=triage_model
            ):
                yield event
            return

        context = ToolContext(
            session_id=session_id,
            request_id=request_id,
            user_id=user_id,
            member_id=member_id,
            user_text=current_message,
        )
        system_prompt = AGENT_SYSTEM
        extra_system: list[dict[str, Any]] = []
        if background:
            if member_context:
                extra_system = [{"role": "system", "content": background}]
            else:
                system_prompt = system_prompt + "\n\n就诊人背景信息（不是当前主诉，仅供背景参考）：\n" + background
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *extra_system,
            *trim_history(history or []),
            {
                "role": "user",
                "content": (
                    f"分诊结果：科室={triage['department']}，紧急度={triage['urgency']}，"
                    f"症状摘要={triage.get('symptoms', '')}。\n用户原始输入：{current_message}"
                ),
            },
        ]
        final_content = ""
        last_result: ChatResult | None = None
        for _round in range(6):
            result = await self.gateway.chat(
                messages,
                tools=self.tools.definitions(),
                task="diagnosis",
                temperature=0.2,
                max_tokens=3000,
                session_id=session_id,
                reasoning_effort="high",
            )
            last_result = result
            await self._record(result, request_id, session_id, "diagnosis", user_id=user_id)
            if not result.tool_calls:
                final_content = result.content or ""
                break
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                    }
                    for call in result.tool_calls
                ],
            }
            messages.append(assistant_message)
            for call in result.tool_calls:
                yield await self._emit(
                    session_id,
                    "thought",
                    {"text": f"调用工具：{call.name}", "tool": call.name},
                )
                tool_result = await self.tools.execute(call.name, call.arguments, context)
                if call.name == "submit_report" and context.report:
                    if not str(context.report.get("summary") or "").strip():
                        context.report["summary"] = (result.content or "").strip() or self._default_summary(triage)
                    if float(context.report.get("confidence") or 0) <= 0:
                        context.report["confidence"] = float(triage.get("confidence") or 0.5)
                    if not call.arguments.get("citations"):
                        context.report["citations"] = []
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )
                yield await self._emit(
                    session_id,
                    "status",
                    {"stage": "tool", "tool": call.name, "ok": bool(tool_result.get("ok")), "detail": self._summarize(tool_result)},
                )
                if call.name in {"retrieve_evidence", "lookup_guideline"} and tool_result.get("ok"):
                    context.citations = self._filter_citations(context.citations)
                    async for event in self._emit_citations(session_id, context):
                        yield event
                if call.name == "check_drug_interaction" and tool_result.get("ok"):
                    yield await self._emit(
                        session_id,
                        "status",
                        {
                            "stage": "drug_check",
                            "highest_severity": tool_result.get("highest_severity", "none"),
                            "interactions": tool_result.get("interactions", []),
                        },
                    )
                if tool_result.get("data") and call.name in {"create_order", "lock_slot"}:
                    yield await self._emit(session_id, "booking", {"data": tool_result["data"]})
                if context.report:
                    break
            if context.report:
                break

        report = context.report or self._fallback_report(triage, final_content, context)
        if not str(report.get("summary") or "").strip():
            report["summary"] = self._default_summary(triage)
        context.citations = self._filter_citations(context.citations)
        valid_citations = apply_evidence_state(report, context.citations)
        safety = safety_gate(report, context.drug_result)
        report["department"] = normalize_department(report.get("department"))
        report_ids = set(report["citations"])
        final_citations = [item for item in valid_citations if str(item["id"]) in report_ids]
        async for event in self._emit_citations(session_id, context):
            yield event
        for chunk in self._chunks(report.get("summary", "")):
            yield await self._emit(session_id, "token", {"text": chunk})
        yield await self._emit(
            session_id,
            "final",
            {
                "report": report,
                "safety": safety,
                "triage": triage,
                "citations": final_citations,
                "booking": context.booking,
                "model": self._model_event(last_result or triage_model),
            },
        )
        await self.sessions.audit(
            request_id=request_id,
            session_id=session_id,
            actor=user_id,
            action="chat.completed",
            detail={"triage": triage, "report": report, "safety": safety},
        )

    async def _crisis_flow(
        self,
        session_id: str,
        message: str,
        triage: dict[str, Any],
        request_id: str,
        user_id: str = "anonymous",
        *,
        model_meta: ChatResult | dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        protocol = crisis_protocol(triage.get("crisis_subtype"))
        report = {
            "department": "急诊科",
            "urgency": "critical",
            "summary": protocol["summary"],
            "advice": list(protocol["advice"]),
            "medications": [],
            "citations": [],
            "confidence": None,
            "rule_triggered": True,
            "crisis_subtype": protocol["subtype"],
            "crisis_terms": list(triage.get("crisis_terms") or []),
            "evidence_state": "none",
            "followup_question": "",
        }
        safety = safety_gate(report, None)
        for chunk in self._chunks(report["summary"]):
            yield await self._emit(session_id, "token", {"text": chunk})
        yield await self._emit(
            session_id,
            "final",
            {
                "report": report,
                "safety": safety,
                "triage": triage,
                "citations": [],
                "booking": None,
                "model": self._model_event(model_meta),
            },
        )
        await self.sessions.audit(
            request_id=request_id,
            session_id=session_id,
            actor=user_id,
            action="chat.crisis",
            detail={"triage": triage, "report": report},
        )

    async def _triage(
        self,
        message: str,
        request_id: str,
        session_id: str,
        history: list[dict[str, Any]] | None = None,
        user_id: str = "anonymous",
        member_context: str | None = None,
    ) -> dict[str, Any]:
        current = split_member_prefix(message)[1]
        user_history = [
            item
            for item in (history or [])
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        system_prompt = TRIAGE_SYSTEM
        if member_context:
            system_prompt = (
                system_prompt
                + "\n\n就诊人背景（不是当前主诉，不得据此判断本次危机或科室）：\n"
                + member_context
            )
        triage_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        recent = trim_history(history or [], max_chars=2000)[-2:]
        if recent:
            transcript = "\n".join(
                f"{'用户' if item.get('role') == 'user' else '助手'}：{item.get('content', '')}" for item in recent
            )
            triage_messages.append(
                {
                    "role": "system",
                    "content": f"最近对话（仅用于理解当前追问，分诊对象仍是最后一条用户输入）：\n{transcript}",
                }
            )
        triage_messages.append({"role": "user", "content": current})
        try:
            result = await self.gateway.chat(
                triage_messages,
                task="triage",
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
                session_id=session_id,
                reasoning_effort="low",
            )
            await self._record(result, request_id, session_id, "triage", user_id=user_id)
            parsed = self._parse_json(result.content)
            if not parsed:
                raise ValueError("triage json parse failed")
            raw_department = str(parsed.get("department") or "").strip()
            normalized = normalize_department(raw_department)
            if normalized == "全科" and raw_department not in {"", "全科", "General Practice"}:
                normalized = rule_triage(current, history=user_history)["department"]
            parsed["department"] = normalized
            urgency = str(parsed.get("urgency") or "normal").strip().lower()
            parsed["urgency"] = urgency if urgency in URGENCIES else "normal"
            intent = str(parsed.get("intent") or "MEDICAL_CONSULT").strip().upper()
            parsed["intent"] = intent if intent in INTENTS else "MEDICAL_CONSULT"
            parsed["confidence"] = float(parsed.get("confidence") or 0.5)
            parsed["symptoms"] = str(parsed.get("symptoms") or current[:120])
            parsed = apply_urgency_floor(parsed, current, history=user_history)
            parsed["_model_meta"] = {"provider": result.provider, "model": result.model}
            return parsed
        except Exception:
            fallback = rule_triage(current, history=user_history)
            fallback["_model_meta"] = {"provider": None, "model": None, "rule_triggered": True}
            return fallback

    def _default_summary(self, triage: dict[str, Any]) -> str:
        return (
            f"根据目前信息，建议优先就诊 {triage.get('department', '全科')}。"
            "请携带既往病历和用药清单；若症状加重或出现胸痛、呼吸困难、意识不清等危机信号，请立即前往急诊或拨打 120。"
            "以上为分诊建议，不能替代医生面诊与确诊。"
        )

    def _fallback_report(self, triage: dict[str, Any], content: str, context: ToolContext) -> dict[str, Any]:
        summary = content.strip() or (
            f"根据目前信息，建议优先就诊 {triage['department']}。"
            "请补充症状持续时间、诱因、伴随症状、既往史和用药史，以便进一步判断。"
        )
        return {
            "department": triage["department"],
            "urgency": triage["urgency"],
            "summary": summary,
            "advice": ["如症状加重或出现危机信号，立即前往急诊", "就诊时携带既往病历和用药清单"],
            "medications": [],
            "citations": [],
            "confidence": float(triage.get("confidence") or 0.4),
            "followup_question": "" if content.strip() else "请补充症状持续时间、诱因、伴随症状、既往史和用药情况。",
        }

    @staticmethod
    def _filter_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in citations if float(item.get("score") or 0) >= MIN_CITATION_SCORE]

    @staticmethod
    def _model_event(meta: ChatResult | dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(meta, ChatResult):
            provider, model = meta.provider, meta.model
        elif isinstance(meta, dict):
            provider, model = meta.get("provider"), meta.get("model")
        else:
            provider, model = None, None
        event: dict[str, Any] = {"provider": provider, "model": model}
        if provider == "ollama":
            event["degraded"] = True
            event["degradation_reason"] = "primary model unavailable; fell back to local Ollama without tools"
        elif provider is None:
            event["rule_triggered"] = True
        return event

    async def _emit_citations(self, session_id: str, context: ToolContext) -> AsyncGenerator[dict[str, Any], None]:
        for item in context.citations:
            key = str(item.get("id"))
            if key in context.emitted_citations:
                continue
            if len(context.emitted_citations) >= 6:
                break
            context.emitted_citations.add(key)
            yield await self._emit(session_id, "citation", item)

    async def _emit(self, session_id: str, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        seq = await self.sessions.append(session_id, event_type, data)
        return {"seq": seq, "type": event_type, "data": data}

    async def _record(
        self,
        result: ChatResult,
        request_id: str,
        session_id: str,
        task: str,
        *,
        user_id: str,
    ) -> None:
        await self.sessions.record_model_call(
            request_id=request_id,
            session_id=session_id,
            provider=result.provider,
            model=result.model,
            task=task,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost=result.cost,
            latency_ms=result.latency_ms,
            user_id=user_id,
        )

    def _summarize(self, result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return str(result.get("error", "tool failed"))[:160]
        if "count" in result:
            return f"{result.get('count')} 条结果"
        data = result.get("data")
        if isinstance(data, list):
            return f"{len(data)} 条结果"
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False, default=str)[:180]
        return str(data or "ok")[:160]

    def _parse_json(self, content: str) -> dict[str, Any] | None:
        text = (content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

    def _chunks(self, text: str, size: int = 48) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        chunks: list[str] = []
        current = ""
        for char in text:
            current += char
            if len(current) >= size or char in "。！？；\n":
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)
        return chunks
