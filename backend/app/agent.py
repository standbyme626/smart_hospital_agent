from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from .config import Settings
from .conversation import trim_history
from .model_gateway import ChatResult, ModelGateway
from .policy import DEPARTMENT_NAMES, detect_crisis, normalize_department, rule_triage, safety_gate
from .prompts import AGENT_SYSTEM, TRIAGE_SYSTEM
from .retrieval import RetrievalService
from .session_store import SessionStore
from .tools import ToolContext, ToolRegistry

URGENCIES = {"normal", "urgent", "critical"}
INTENTS = {"MEDICAL_CONSULT", "CRISIS", "INFO", "GREETING", "REGISTRATION"}


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
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self.sessions.ensure(session_id, user_id, title=(title or message)[:40], member_id=member_id)
        yield await self._emit(session_id, "status", {"stage": "triage", "detail": "正在分诊"})

        triage = await self._triage(message, request_id, session_id, history=history)
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
            async for event in self._crisis_flow(session_id, message, triage, request_id, user_id):
                yield event
            return

        context = ToolContext(
            session_id=session_id,
            request_id=request_id,
            user_id=user_id,
            member_id=member_id,
            user_text=message,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM},
            *trim_history(history or []),
            {
                "role": "user",
                "content": (
                    f"分诊结果：科室={triage['department']}，紧急度={triage['urgency']}，"
                    f"症状摘要={triage.get('symptoms', '')}。\n用户原始输入：{message}"
                ),
            },
        ]
        final_content = ""
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
            await self._record(result, request_id, session_id, "diagnosis")
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

        if not context.citations:
            context.citations = await self.retrieval.search(message, top_k=3)
        report = context.report or self._fallback_report(triage, final_content, context)
        if not str(report.get("summary") or "").strip():
            report["summary"] = self._default_summary(triage)
        valid_ids = {str(item["id"]) for item in context.citations}
        report_citations = [str(item) for item in (report.get("citations") or []) if str(item) in valid_ids]
        if not report_citations:
            report_citations = list(valid_ids)
        report["citations"] = report_citations
        report["department"] = normalize_department(report.get("department"))
        safety = safety_gate(report, context.drug_result)
        final_citations = [item for item in context.citations if str(item["id"]) in set(report_citations)]
        if not final_citations:
            final_citations = context.citations[:5]
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
                "model": {"provider": "opencode-go", "model": self.settings.LLM_MODEL},
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
    ) -> AsyncGenerator[dict[str, Any], None]:
        evidence = await self.retrieval.search(message, top_k=3)
        context = ToolContext(session_id=session_id, request_id=request_id, user_id=user_id, user_text=message)
        context.citations = evidence
        report = {
            "department": "急诊科",
            "urgency": "critical",
            "summary": (
                "你描述的情况可能属于急危重症，请立即拨打 120 或前往最近的急诊科。"
                "在等待救援时保持安静、不要自行服药；如果出现意识丧失、呼吸停止，请立即进行心肺复苏并呼叫旁人协助。"
            ),
            "advice": ["立即拨打 120", "不要自行驾车", "保持呼吸道通畅", "携带既往病历和用药清单"],
            "medications": [],
            "citations": [item["id"] for item in evidence],
            "confidence": 0.95,
            "followup_question": "",
        }
        safety = safety_gate(report, None)
        async for event in self._emit_citations(session_id, context):
            yield event
        for chunk in self._chunks(report["summary"]):
            yield await self._emit(session_id, "token", {"text": chunk})
        yield await self._emit(
            session_id,
            "final",
            {"report": report, "safety": safety, "triage": triage, "citations": evidence, "booking": None},
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
    ) -> dict[str, Any]:
        triage_messages: list[dict[str, Any]] = [{"role": "system", "content": TRIAGE_SYSTEM}]
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
        triage_messages.append({"role": "user", "content": message})
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
            await self._record(result, request_id, session_id, "triage")
            parsed = self._parse_json(result.content)
            if not parsed:
                raise ValueError("triage json parse failed")
            department = str(parsed.get("department") or "").strip()
            if department not in DEPARTMENT_NAMES and department != "General Practice":
                parsed["department"] = rule_triage(message)["department"]
            urgency = str(parsed.get("urgency") or "normal").lower()
            parsed["urgency"] = urgency if urgency in URGENCIES else "normal"
            intent = str(parsed.get("intent") or "MEDICAL_CONSULT").upper()
            parsed["intent"] = intent if intent in INTENTS else "MEDICAL_CONSULT"
            parsed["confidence"] = float(parsed.get("confidence") or 0.5)
            parsed["symptoms"] = str(parsed.get("symptoms") or message[:120])
            parsed["department"] = normalize_department(parsed.get("department"))
            crisis, terms = detect_crisis(message)
            if crisis:
                parsed.update(
                    {
                        "intent": "CRISIS",
                        "urgency": "critical",
                        "department": "急诊科",
                        "reason": "命中危机关键词：" + "、".join(terms),
                    }
                )
            return parsed
        except Exception:
            return rule_triage(message)

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
            "citations": [item["id"] for item in context.citations],
            "confidence": float(triage.get("confidence") or 0.4),
            "followup_question": "" if content.strip() else "请补充症状持续时间、诱因、伴随症状、既往史和用药情况。",
        }

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

    async def _record(self, result: ChatResult, request_id: str, session_id: str, task: str) -> None:
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
