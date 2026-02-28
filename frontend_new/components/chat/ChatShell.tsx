"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";

import { Composer } from "@/components/chat/Composer";
import { MessageList } from "@/components/chat/MessageList";
import { streamChat } from "@/lib/stream";
import type { BookingPayload, ChatMessage, SlotItem } from "@/lib/types";

interface TraceItem {
  id: string;
  ts: number;
  label: string;
  level: "info" | "success" | "error";
}

interface BookingFlowState {
  open: boolean;
  step: 1 | 2 | 3;
  slot: SlotItem | null;
  orderId: string;
  amount: number | string | "";
  waitingLock: boolean;
  waitingPay: boolean;
  error: string;
  details: Record<string, unknown> | null;
}

const BOOT_TEXT = "你好，我是智慧医院助理。请描述症状，或先查询号源后预约挂号。";
const SESSION_PLACEHOLDER = "session-pending";
const FEATURE_PROMPTS: Record<string, string> = {
  symptom: "我想做症状咨询，请先完成分诊并给出就诊建议。",
  booking: "我想预约挂号，请先帮我查询可预约号源。",
  inspection: "我想做检查检验咨询，请给出建议的检查项目和注意事项。",
  medication: "我想咨询用药，请给出适应症、禁忌和风险提示。"
};

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}`;
}

function createAssistantMessage(id: string): ChatMessage {
  return { id, role: "assistant", text: "", thoughts: [], slots: [] };
}

function createUserMessage(content: string): ChatMessage {
  return {
    id: `u-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role: "user",
    text: content,
    thoughts: [],
    slots: []
  };
}

function createBootMessages(): ChatMessage[] {
  return [
    {
      id: "boot",
      role: "assistant",
      text: BOOT_TEXT,
      thoughts: [],
      slots: []
    }
  ];
}

function createInitialBookingState(): BookingFlowState {
  return {
    open: false,
    step: 1,
    slot: null,
    orderId: "",
    amount: "",
    waitingLock: false,
    waitingPay: false,
    error: "",
    details: null
  };
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function parseBookingPayload(payload: BookingPayload | undefined): {
  orderId: string;
  amount: number | string | "";
  details: Record<string, unknown> | null;
} {
  const orderId = String(payload?.order_id || "");
  const amount = payload?.payment_required ?? "";
  const details =
    payload?.details && typeof payload.details === "object" ? payload.details : payload && typeof payload === "object" ? payload : null;
  return { orderId, amount, details };
}

export function ChatShell() {
  const [sessionId, setSessionId] = useState<string>(SESSION_PLACEHOLDER);
  const [messages, setMessages] = useState<ChatMessage[]>(createBootMessages);
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState("就绪");
  const [error, setError] = useState("");
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [bookingFlow, setBookingFlow] = useState<BookingFlowState>(createInitialBookingState);
  const [autoFollow, setAutoFollow] = useState(true);
  const [clock, setClock] = useState(0);
  const [lastTurnMs, setLastTurnMs] = useState(0);

  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLElement>(null);
  const sessionStartRef = useRef<number>(0);
  const turnStartRef = useRef<number | null>(null);

  useEffect(() => {
    const now = Date.now();
    setSessionId((prev) => (prev === SESSION_PLACEHOLDER ? createSessionId() : prev));
    setClock(now);
    sessionStartRef.current = now;
    const timer = window.setInterval(() => setClock(Date.now()), 500);
    return () => {
      window.clearInterval(timer);
    };
  }, []);

  const totalMs = sessionStartRef.current ? clock - sessionStartRef.current : 0;
  const activeTurnMs = turnStartRef.current ? clock - turnStartRef.current : 0;

  const appendTrace = (label: string, level: TraceItem["level"] = "info") => {
    const item: TraceItem = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      ts: Date.now(),
      label,
      level
    };
    setTrace((prev) => [...prev.slice(-17), item]);
  };

  const scrollToBottom = useCallback((force = false) => {
    const container = listRef.current;
    if (!container) {
      return;
    }
    if (!force && !autoFollow) {
      return;
    }
    container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [autoFollow]);

  useEffect(() => {
    scrollToBottom(false);
  }, [messages, scrollToBottom]);

  const pushUserAndAssistant = (content: string): string => {
    const assistantId = `a-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setMessages((prev) => [...prev, createUserMessage(content), createAssistantMessage(assistantId)]);
    return assistantId;
  };

  const applyBookingEvent = (eventType: string, payload: BookingPayload | undefined) => {
    const parsed = parseBookingPayload(payload);
    if (eventType === "booking_preview") {
      appendTrace("挂号预览已返回", "info");
      setBookingFlow((prev) => ({
        ...prev,
        open: true,
        step: 2,
        waitingLock: false,
        orderId: parsed.orderId || prev.orderId,
        amount: parsed.amount !== "" ? parsed.amount : prev.amount
      }));
      setStatus("号源已锁定，等待支付");
      return;
    }

    if (eventType === "payment_required") {
      appendTrace("进入支付环节", "info");
      setBookingFlow((prev) => ({
        ...prev,
        open: true,
        step: 2,
        waitingLock: false,
        waitingPay: false,
        orderId: parsed.orderId || prev.orderId,
        amount: parsed.amount !== "" ? parsed.amount : prev.amount,
        error: ""
      }));
      setStatus("待支付");
      return;
    }

    if (eventType === "booking_confirmed") {
      appendTrace("支付成功，预约确认", "success");
      setBookingFlow((prev) => ({
        ...prev,
        open: true,
        step: 3,
        waitingPay: false,
        error: "",
        details: parsed.details
      }));
      setStatus("预约成功");
      return;
    }

    if (eventType === "booking_error") {
      appendTrace("挂号流程失败", "error");
      setBookingFlow((prev) => ({
        ...prev,
        open: true,
        waitingLock: false,
        waitingPay: false,
        error: String(payload?.message || "挂号流程失败，请重试。")
      }));
      setStatus("挂号异常");
    }
  };

  const send = async (content: string) => {
    if (pending) {
      return;
    }

    let activeSessionId = sessionId;
    if (activeSessionId === SESSION_PLACEHOLDER) {
      activeSessionId = createSessionId();
      setSessionId(activeSessionId);
      sessionStartRef.current = Date.now();
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const assistantId = pushUserAndAssistant(content);
    setPending(true);
    setError("");
    setStatus("请求后端流式响应...");
    turnStartRef.current = Date.now();
    appendTrace(`发送请求: ${content.slice(0, 48)}`, "info");
    scrollToBottom(true);

    const patchAssistant = (patch: (msg: ChatMessage) => ChatMessage) => {
      setMessages((prev) => prev.map((msg) => (msg.id === assistantId ? patch(msg) : msg)));
    };

    const finishTurn = (finalStatus: string) => {
      if (turnStartRef.current) {
        setLastTurnMs(Date.now() - turnStartRef.current);
      }
      turnStartRef.current = null;
      setPending(false);
      setStatus(finalStatus);
    };

    try {
      await streamChat({
        message: content,
        sessionId: activeSessionId,
        signal: controller.signal,
        onEvent: (event) => {
          if (event.type === "thought") {
            const text = event.content || "处理中...";
            setStatus(text);
            appendTrace(text, "info");
            patchAssistant((msg) => {
              if (!event.content || msg.thoughts.includes(event.content)) {
                return msg;
              }
              return { ...msg, thoughts: [...msg.thoughts, event.content] };
            });
            return;
          }

          if (event.type === "token") {
            setStatus("输出回答中...");
            patchAssistant((msg) => ({ ...msg, text: `${msg.text}${event.content}` }));
            return;
          }

          if (event.type === "final") {
            patchAssistant((msg) => {
              if (!msg.text && event.content) {
                return { ...msg, text: event.content };
              }
              return msg;
            });
            return;
          }

          if (event.type === "status") {
            const text = event.content || "状态更新";
            setStatus(text);
            const rewriteFallback = Boolean(event.meta?.rewrite_fallback);
            if (rewriteFallback) {
              appendTrace(`rewrite fallback: ${String(event.meta?.fallback_reason || "unknown")}`, "info");
            }
            return;
          }

          if (event.type === "ping") {
            return;
          }

          if (event.type === "doctor_slots") {
            setStatus("接收到号源信息");
            appendTrace("号源卡片已更新", "info");
            patchAssistant((msg) => ({ ...msg, slots: event.slots || [] }));
            return;
          }

          if (event.type === "booking_preview" || event.type === "payment_required" || event.type === "booking_confirmed" || event.type === "booking_error") {
            applyBookingEvent(event.type, event.booking);
            return;
          }

          if (event.type === "error") {
            const text = event.content || "后端返回错误";
            setError(text);
            appendTrace(text, "error");
          }
        },
        onDone: () => {
          appendTrace("流式响应完成", "success");
          finishTurn("完成");
        },
        onError: (reason) => {
          appendTrace(reason, "error");
          setError(reason);
          finishTurn("异常");
        }
      });
    } catch (e) {
      const text = e instanceof Error ? e.message : String(e);
      appendTrace(text, "error");
      setError(text);
      finishTurn("异常");
    }
  };

  const resetSession = () => {
    abortRef.current?.abort();
    setSessionId(createSessionId());
    setMessages(createBootMessages());
    setPending(false);
    setStatus("就绪");
    setError("");
    setTrace([]);
    setBookingFlow(createInitialBookingState());
    setAutoFollow(true);
    setLastTurnMs(0);
    setClock(Date.now());
    sessionStartRef.current = Date.now();
    turnStartRef.current = null;
  };

  const runFeatureAction = (feature: keyof typeof FEATURE_PROMPTS) => {
    const prompt = FEATURE_PROMPTS[feature];
    appendTrace(`功能入口触发: ${prompt.slice(0, 28)}...`, "info");
    void send(prompt);
  };

  const handleListScroll = () => {
    const container = listRef.current;
    if (!container) {
      return;
    }
    const gap = container.scrollHeight - container.scrollTop - container.clientHeight;
    setAutoFollow(gap < 64);
  };

  const openBooking = (slot: SlotItem) => {
    setBookingFlow({
      open: true,
      step: 1,
      slot,
      orderId: "",
      amount: "",
      waitingLock: false,
      waitingPay: false,
      error: "",
      details: null
    });
    appendTrace(`准备预约号源 ${String(slot.slot_id || "")}`, "info");
  };

  const submitBooking = () => {
    const slotId = String(bookingFlow.slot?.slot_id || "");
    if (!slotId) {
      setBookingFlow((prev) => ({ ...prev, error: "缺少 slot_id，无法预约。" }));
      return;
    }
    setBookingFlow((prev) => ({ ...prev, step: 2, waitingLock: true, error: "" }));
    void send(`BOOK:${slotId}`);
  };

  const submitPayment = () => {
    if (!bookingFlow.orderId) {
      setBookingFlow((prev) => ({ ...prev, error: "缺少订单号，无法支付。" }));
      return;
    }
    setBookingFlow((prev) => ({ ...prev, waitingPay: true, error: "" }));
    void send(`PAY:${bookingFlow.orderId}`);
  };

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>智慧医院 Agent 控制台</h1>
          <p>Session: {sessionId}</p>
        </div>
        <div className={`badge ${pending ? "busy" : "idle"}`}>{status}</div>
      </header>

      <section className="workspace">
        <aside className="side-left">
          <h2>功能入口</h2>
          <button type="button" className="nav-btn primary" onClick={resetSession}>
            新对话
          </button>
          <button type="button" className="nav-btn" onClick={() => runFeatureAction("symptom")}>
            症状咨询
          </button>
          <button type="button" className="nav-btn" onClick={() => runFeatureAction("booking")}>
            预约挂号
          </button>
          <button type="button" className="nav-btn" onClick={() => runFeatureAction("inspection")}>
            检查检验
          </button>
          <button type="button" className="nav-btn" onClick={() => runFeatureAction("medication")}>
            用药咨询
          </button>
          <Link className="nav-btn rag-link" href="/rag">
            RAG 医学文档检索
          </Link>

          <div className="side-card">
            <h3>系统状态</h3>
            <p>Backend: 8001</p>
            <p>Frontend: 3000</p>
            <p>协议: SSE 流式</p>
            <p>总耗时: {formatDuration(totalMs)}</p>
          </div>
        </aside>

        <section className="center-panel">
          {error ? <p className="error">{error}</p> : null}

          <MessageList messages={messages} pending={pending} onBook={openBooking} listRef={listRef} onScroll={handleListScroll} />

          {!autoFollow ? (
            <button type="button" className="scroll-btn" onClick={() => scrollToBottom(true)}>
              回到底部
            </button>
          ) : null}

          <Composer pending={pending} onSubmitText={(text) => void send(text)} />
        </section>

        <aside className="side-right">
          <h2>运行过程</h2>
          <p>会话耗时: {formatDuration(totalMs)}</p>
          <p>本轮耗时: {pending ? formatDuration(activeTurnMs) : formatDuration(lastTurnMs)}</p>
          <div className="trace-list" aria-live="polite">
            {trace.length === 0 ? <p className="trace-empty">等待请求...</p> : null}
            {trace.map((item) => (
              <article key={item.id} className={`trace-item ${item.level}`}>
                <time>{new Date(item.ts).toLocaleTimeString()}</time>
                <p>{item.label}</p>
              </article>
            ))}
          </div>
        </aside>
      </section>

      {bookingFlow.open ? (
        <div className="booking-modal-mask" role="dialog" aria-modal="true">
          <section className="booking-modal">
            <header>
              <h3>挂号流程</h3>
              <button type="button" onClick={() => setBookingFlow(createInitialBookingState())}>
                关闭
              </button>
            </header>

            {bookingFlow.step === 1 ? (
              <div className="booking-step">
                <h4>Step 1/3 核对号源</h4>
                <p>医生: {String(bookingFlow.slot?.doctor || "未提供")}</p>
                <p>日期: {String(bookingFlow.slot?.date || "未提供")}</p>
                <p>时间: {String(bookingFlow.slot?.time || "未提供")}</p>
                <p>费用: {String(bookingFlow.slot?.fee ?? "未提供")}</p>
                <div className="booking-actions">
                  <button type="button" onClick={() => setBookingFlow(createInitialBookingState())}>
                    取消
                  </button>
                  <button type="button" className="primary" onClick={submitBooking} disabled={pending}>
                    确认挂号
                  </button>
                </div>
              </div>
            ) : null}

            {bookingFlow.step === 2 ? (
              <div className="booking-step">
                <h4>Step 2/3 支付确认</h4>
                <p>订单号: {bookingFlow.orderId || "等待生成..."}</p>
                <p>支付金额: {bookingFlow.amount === "" ? "等待返回..." : `¥${bookingFlow.amount}`}</p>
                {bookingFlow.waitingLock ? <p className="hint">正在锁定号源，请稍候...</p> : null}
                {bookingFlow.error ? <p className="error inline">{bookingFlow.error}</p> : null}
                <div className="booking-actions">
                  <button
                    type="button"
                    onClick={submitBooking}
                    disabled={pending || bookingFlow.waitingLock || !bookingFlow.slot?.slot_id}
                  >
                    重试锁号
                  </button>
                  <button
                    type="button"
                    className="primary"
                    onClick={submitPayment}
                    disabled={pending || bookingFlow.waitingLock || bookingFlow.waitingPay || !bookingFlow.orderId}
                  >
                    {bookingFlow.waitingPay ? "支付处理中..." : "确认支付"}
                  </button>
                </div>
              </div>
            ) : null}

            {bookingFlow.step === 3 ? (
              <div className="booking-step">
                <h4>Step 3/3 挂号成功</h4>
                <p>订单号: {bookingFlow.orderId || "未提供"}</p>
                <p>就诊信息: {String(bookingFlow.details?.time || "请提前到院候诊")}</p>
                <p>地点: {String(bookingFlow.details?.meet_location || "门诊大厅")}</p>
                <div className="booking-actions">
                  <button type="button" className="primary" onClick={() => setBookingFlow(createInitialBookingState())}>
                    完成
                  </button>
                </div>
              </div>
            ) : null}
          </section>
        </div>
      ) : null}
    </main>
  );
}
