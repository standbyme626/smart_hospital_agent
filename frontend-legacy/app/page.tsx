"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  CalendarClock,
  CheckCircle2,
  ChevronRight,
  Loader2,
  Menu,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserRound,
  X,
} from "lucide-react";
import clsx from "clsx";
import { Sidebar } from "@/components/Sidebar";
import { ReportCard } from "@/components/ReportCard";
import {
  UnauthorizedError,
  getMe,
  getSessions,
  streamChat,
} from "@/lib/api";
import type { ChatMessage, Citation, Session } from "@/lib/types";

const examples = [
  "头痛伴发热两天，该挂什么科？",
  "高血压患者能吃阿司匹林吗？",
  "孩子发烧 38.5 度，需要马上去医院吗？",
  "我想挂心内科的号，最近总是心悸。",
];

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function statusText(data: Record<string, unknown>): string {
  const stage = String(data.stage || "");
  if (stage === "triage") return "正在分诊";
  if (stage === "triage_done") {
    const detail = data.detail as Record<string, unknown> | undefined;
    if (detail) return `分诊完成 · ${detail.department || ""} · ${detail.urgency || ""}`;
    return "分诊完成";
  }
  if (stage === "tool") return `工具 ${data.tool || ""} · ${data.ok ? "完成" : "失败"}`;
  if (stage === "drug_check") return `药物检查 · ${data.highest_severity || "none"}`;
  return stage || "处理中";
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [statusLines, setStatusLines] = useState<string[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [authError, setAuthError] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await getSessions());
    } catch (err) {
      if (err instanceof UnauthorizedError) setAuthError(true);
    }
  }, []);

  useEffect(() => {
    getMe().catch((err) => {
      if (err instanceof UnauthorizedError) setAuthError(true);
    });
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, statusLines]);

  const updateMessage = useCallback((id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((message) => (message.id === id ? { ...message, ...patch } : message)));
  }, []);

  const send = useCallback(
    async (raw?: string) => {
      const text = (raw ?? input).trim();
      if (!text || streaming) return;
      setInput("");
      setError(null);
      setSidebarOpen(false);
      setStatusLines([]);
      setCitations([]);
      setSelectedCitation(null);
      const userMessage: ChatMessage = { id: newId(), role: "user", content: text };
      const assistantId = newId();
      const assistantMessage: ChatMessage = { id: assistantId, role: "assistant", content: "" };
      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setStreaming(true);
      let streamed = "";
      let streamedCitations: Citation[] = [];
      try {
        for await (const event of streamChat({ message: text, session_id: sessionId ?? undefined })) {
          if (event.type === "session") {
            const id = String(event.data.session_id || "");
            if (id) setSessionId(id);
          } else if (event.type === "status") {
            setStatusLines((prev) => [...prev, statusText(event.data)]);
          } else if (event.type === "thought") {
            const thought = String(event.data.text || "").trim();
            if (thought) setStatusLines((prev) => [...prev, thought]);
          } else if (event.type === "citation") {
            const citation = event.data as unknown as Citation;
            if (!streamedCitations.some((item) => String(item.id) === String(citation.id))) {
              streamedCitations = [...streamedCitations, citation];
              setCitations(streamedCitations);
              updateMessage(assistantId, { citations: streamedCitations });
            }
          } else if (event.type === "token") {
            streamed += String(event.data.text || "");
            updateMessage(assistantId, { content: streamed });
          } else if (event.type === "final") {
            const report = event.data.report as ChatMessage["report"];
            const finalCitations = (event.data.citations as unknown as Citation[]) || streamedCitations;
            const safety = event.data.safety as ChatMessage["safety"];
            const triage = event.data.triage as ChatMessage["triage"];
            updateMessage(assistantId, {
              content: report?.summary || streamed,
              report,
              citations: finalCitations,
              safety,
              triage,
            });
            setCitations(finalCitations);
            setStatusLines((prev) => [...prev, "报告已生成"]);
            refreshSessions();
          } else if (event.type === "error") {
            const message = String(event.data.message || "服务异常");
            setError(message);
            updateMessage(assistantId, { error: message });
          }
        }
      } catch (err) {
        if (err instanceof UnauthorizedError) {
          setAuthError(true);
        } else {
          const message = err instanceof Error ? err.message : "请求失败";
          setError(message);
          updateMessage(assistantId, { error: message });
        }
      } finally {
        setStreaming(false);
        textareaRef.current?.focus();
      }
    },
    [input, streaming, sessionId, refreshSessions, updateMessage],
  );

  const newChat = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    setStatusLines([]);
    setCitations([]);
    setSelectedCitation(null);
    setError(null);
    setSidebarOpen(false);
    textareaRef.current?.focus();
  }, []);

  const lastCitations = useMemo(() => citations.slice(0, 8), [citations]);
  const showEmpty = messages.length === 0;

  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar
        sessions={sessions}
        activeSession={sessionId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNew={newChat}
        onSelect={(id) => {
          setSessionId(id);
          setMessages([]);
          setStatusLines([]);
          setCitations([]);
        }}
      />

      <div className="flex min-w-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col">
          <header className="flex items-center gap-3 border-b border-line bg-panel/70 px-4 py-3 backdrop-blur">
            <button className="btn-ghost h-9 w-9 p-0 lg:hidden" onClick={() => setSidebarOpen(true)} aria-label="打开侧栏">
              <Menu size={16} />
            </button>
            <div className="min-w-0 flex-1">
              <h1 className="truncate text-sm font-semibold text-white">智能分诊与就医咨询</h1>
              <p className="truncate text-[11px] text-slate-400">DeepSeek V4.1 Flash · Qwen3-8B 向量/重排 · 证据可追溯</p>
            </div>
            <span className="hidden items-center gap-1.5 rounded-full border border-mint/30 bg-mint/10 px-2.5 py-1 text-[11px] text-mint sm:inline-flex">
              <span className="h-1.5 w-1.5 rounded-full bg-mint pulse-dot" />
              服务在线
            </span>
          </header>

          {authError ? (
            <div className="border-b border-amber/30 bg-amber/10 px-4 py-2.5 text-xs text-amber">
              需要登录后才能使用。请前往 <a className="underline" href="/login">登录页</a>。
            </div>
          ) : null}
          {error ? (
            <div className="flex items-center gap-2 border-b border-rose/30 bg-rose/10 px-4 py-2.5 text-xs text-rose">
              <AlertTriangle size={14} />
              {error}
            </div>
          ) : null}

          <div className="scrollbar-thin flex-1 overflow-y-auto px-4 py-6">
            <div className="mx-auto max-w-3xl space-y-5">
              {showEmpty ? (
                <div className="py-10">
                  <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs text-primary">
                    <Sparkles size={13} />
                    分子医学 · 循证分诊
                  </div>
                  <h2 className="mt-5 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                    先说症状，<span className="text-primary">再决定挂什么科。</span>
                  </h2>
                  <p className="mt-3 max-w-xl text-sm leading-6 text-slate-400">
                    大医会先做分诊与危机识别，再检索医学证据、检查药物相互作用，最后给出可追溯的就医建议。
                    需要挂号时，可以继续完成号源锁定与支付。
                  </p>
                  <div className="mt-6 grid gap-2 sm:grid-cols-2">
                    {examples.map((example) => (
                      <button
                        key={example}
                        className="surface-2 group flex items-center justify-between px-3.5 py-3 text-left text-sm text-slate-300 transition hover:border-primary/40 hover:text-white"
                        onClick={() => send(example)}
                      >
                        <span>{example}</span>
                        <ChevronRight size={15} className="text-slate-500 group-hover:text-primary" />
                      </button>
                    ))}
                  </div>
                  <div className="mt-6 flex flex-wrap gap-2 text-[11px] text-slate-500">
                    <span className="chip"><ShieldCheck size={12} /> 危机快车道</span>
                    <span className="chip"><PillIcon /> 药物相互作用</span>
                    <span className="chip"><BookOpen size={12} /> 证据引用</span>
                    <span className="chip"><CalendarClock size={12} /> MCP 挂号支付</span>
                  </div>
                </div>
              ) : null}

              {messages.map((message) => (
                <div key={message.id}>
                  {message.role === "user" ? (
                    <div className="flex justify-end gap-3">
                      <div className="max-w-[85%] rounded-2xl rounded-tr-md bg-primary px-4 py-3 text-sm leading-6 text-slate-950">
                        {message.content}
                      </div>
                      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-panel2 text-slate-300">
                        <UserRound size={15} />
                      </div>
                    </div>
                  ) : (
                    <div className="flex gap-3">
                      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary2 text-slate-950">
                        <Stethoscope size={15} />
                      </div>
                      <div className="min-w-0 flex-1 space-y-3">
                        {message.report ? (
                          <ReportCard
                            report={message.report}
                            citations={message.citations || []}
                            safety={message.safety}
                            onCitation={setSelectedCitation}
                          />
                        ) : (
                          <div className="surface px-4 py-3">
                            {message.content ? (
                              <div className="markdown">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                              </div>
                            ) : (
                              <div className="flex items-center gap-2 text-sm text-slate-400">
                                <Loader2 size={15} className="animate-spin" />
                                正在生成回答…
                              </div>
                            )}
                            {message.error ? <p className="mt-2 text-xs text-rose">{message.error}</p> : null}
                          </div>
                        )}
                        {message.triage ? (
                          <div className="flex flex-wrap gap-2 text-[11px] text-slate-500">
                            <span className="chip">分诊：{message.triage.department}</span>
                            <span className="chip">紧急度：{message.triage.urgency}</span>
                            <span className="chip">意图：{message.triage.intent}</span>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {streaming ? (
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Loader2 size={14} className="animate-spin text-primary" />
                  {statusLines.length ? statusLines[statusLines.length - 1] : "正在处理…"}
                </div>
              ) : null}

              {statusLines.length && !streaming ? (
                <details className="surface-2 px-3 py-2 text-xs text-slate-400">
                  <summary className="cursor-pointer select-none text-slate-300">执行过程（{statusLines.length} 步）</summary>
                  <ul className="mt-2 space-y-1">
                    {statusLines.map((line, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <CheckCircle2 size={12} className="mt-0.5 shrink-0 text-mint" />
                        <span>{line}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              ) : null}
              <div ref={endRef} />
            </div>
          </div>

          <div className="border-t border-line bg-panel/70 px-4 py-3 backdrop-blur">
            <div className="mx-auto max-w-3xl">
              <div className="surface flex items-end gap-2 p-2">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      send();
                    }
                  }}
                  rows={1}
                  placeholder="描述症状、持续时间、用药情况，或直接说要挂什么科…"
                  className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-500"
                />
                <button
                  className="btn-primary h-10 w-10 shrink-0 p-0"
                  disabled={streaming || !input.trim()}
                  onClick={() => send()}
                  aria-label="发送"
                >
                  {streaming ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                </button>
              </div>
              <p className="mt-1.5 text-center text-[10px] text-slate-500">
                Enter 发送 · Shift+Enter 换行 · 本产品不提供确诊与处方，危机情况请立即拨打 120
              </p>
            </div>
          </div>
        </main>

        <aside className="hidden w-80 shrink-0 border-l border-line bg-panel/50 xl:flex xl:flex-col">
          <div className="border-b border-line px-4 py-3">
            <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-slate-400">
              <BookOpen size={14} className="text-primary" />
              证据面板
            </p>
          </div>
          <div className="scrollbar-thin flex-1 overflow-y-auto p-4">
            {selectedCitation ? (
              <div className="surface p-3">
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-medium text-white">{selectedCitation.title}</p>
                  <button className="text-slate-500 hover:text-white" onClick={() => setSelectedCitation(null)} aria-label="关闭">
                    <X size={14} />
                  </button>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] text-slate-400">
                  {selectedCitation.department ? <span className="chip">{selectedCitation.department}</span> : null}
                  {selectedCitation.source ? <span className="chip">{selectedCitation.source}</span> : null}
                  {typeof selectedCitation.score === "number" ? (
                    <span className="chip">score {selectedCitation.score.toFixed(4)}</span>
                  ) : null}
                </div>
                <p className="mt-3 whitespace-pre-wrap text-xs leading-5 text-slate-300">{selectedCitation.snippet}</p>
              </div>
            ) : (
              <div className="space-y-2">
                {lastCitations.length === 0 ? (
                  <p className="text-xs leading-6 text-slate-500">
                    检索到的医学证据会显示在这里。每条建议都可以点击引用编号回看原文片段。
                  </p>
                ) : (
                  lastCitations.map((citation, index) => (
                    <button
                      key={`${citation.id}-${index}`}
                      className="surface-2 w-full px-3 py-2.5 text-left transition hover:border-primary/40"
                      onClick={() => setSelectedCitation(citation)}
                    >
                      <p className="line-clamp-2 text-xs text-slate-200">{citation.title}</p>
                      <p className="mt-1 text-[10px] text-slate-500">
                        {citation.department || "未知科室"} · {citation.source || "evidence"}
                      </p>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
          <div className="border-t border-line p-4 text-[10px] leading-5 text-slate-500">
            <p className="flex items-center gap-1.5 text-slate-400">
              <Search size={12} />
              混合检索：向量 + BM25 + RRF
            </p>
            <p className="mt-1 flex items-center gap-1.5 text-slate-400">
              <ShieldCheck size={12} />
              药物相互作用与妊娠禁忌检查
            </p>
            <p className="mt-1 flex items-center gap-1.5 text-slate-400">
              <Activity size={12} />
              模型调用与成本写入审计
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}

function PillIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="8" width="18" height="8" rx="4" transform="rotate(-45 12 12)" />
    </svg>
  );
}
