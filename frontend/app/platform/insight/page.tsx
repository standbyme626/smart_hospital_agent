"use client";

import { useCallback, useEffect, useState } from "react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import { statusOf } from "@/lib/use-admin-load";
import {
  getAcceptance,
  getAuditAggregate,
  getModelSummary,
  getSessionTrace,
  listSessions,
  type AcceptanceData,
  type AuditAggregate,
  type ModelSummary,
  type SessionItem,
  type SessionTrace,
} from "@/lib/api-admin-ext";

type Tab = "model" | "trace" | "audit" | "accept";

const TABS: Array<{ key: Tab; label: string; screen: string }> = [
  { key: "model", label: "模型网关", screen: "a-model" },
  { key: "trace", label: "会话回放", screen: "a-trace" },
  { key: "audit", label: "配置审计", screen: "a-audit" },
  { key: "accept", label: "验收自测", screen: "a-accept" },
];

const STATUS_STYLE: Record<string, string> = {
  ready: "badge bg-emerald-50 text-emerald-700",
  partial: "badge bg-amber-50 text-amber-700",
  blocked: "badge bg-rose-50 text-rose-700",
  ok: "badge bg-emerald-50 text-emerald-700",
  warn: "badge bg-amber-50 text-amber-700",
  fail: "badge bg-rose-50 text-rose-700",
};

const PLAYBOOK_LABEL: Record<string, string> = { ready: "可跑", partial: "部分", blocked: "阻塞" };

function formatTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

export default function InsightPage() {
  const [tab, setTab] = useState<Tab>("model");
  const [model, setModel] = useState<ModelSummary | null>(null);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [trace, setTrace] = useState<SessionTrace | null>(null);
  const [audit, setAudit] = useState<AuditAggregate | null>(null);
  const [acceptance, setAcceptance] = useState<AcceptanceData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDenied(false);
    try {
      const [modelData, sessionData, auditData, acceptData] = await Promise.all([
        getModelSummary(7),
        listSessions(),
        getAuditAggregate(1),
        getAcceptance(),
      ]);
      setModel(modelData);
      setSessions(sessionData.items);
      setAudit(auditData);
      setAcceptance(acceptData);
    } catch (err) {
      if (statusOf(err) === "denied") setDenied(true);
      else setError((err as Error).message || "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const openTrace = async (sessionId: string) => {
    setError(null);
    try {
      setTrace(await getSessionTrace(sessionId));
    } catch (err) {
      setError((err as Error).message || "读取会话失败");
    }
  };

  return (
    <ConsoleShell end="platform" title="可观测与验收" subtitle="模型调用、会话回放、审计聚合与 25 条剧本看板">

      <div className="mt-3"><OfflineBanner /></div>

      <nav className="mt-4 flex flex-wrap gap-2">
        {TABS.map((item) => (
          <button
            key={item.key}
            onClick={() => setTab(item.key)}
            className={
              tab === item.key
                ? "rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white"
                : "rounded-full bg-white px-4 py-1.5 text-[12px] font-medium text-ink ring-1 ring-line"
            }
          >
            {item.label}
            <span className="ml-1 text-[10px] opacity-60">{item.screen}</span>
          </button>
        ))}
      </nav>

      {error ? <p className="mt-3 rounded-2xl bg-rose-50 px-4 py-2 text-[12px] text-rose-700">{error}</p> : null}

      {loading ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[0, 1, 2].map((i) => <div key={i} className="card h-28 animate-pulse bg-canvas" />)}
        </div>
      ) : denied ? (
        <div className="mt-4 card p-5"><StateBlock kind="denied" title="需要管理员权限" /></div>
      ) : (
        <div className="mt-4 space-y-4">
          {tab === "model" && model ? (
            <>
              <section className="card p-5">
                <h2 className="section-title">近 7 天调用与成本</h2>
                <div className="mt-3 grid gap-3 sm:grid-cols-4">
                  {[
                    { label: "调用次数", value: model.totals.calls },
                    { label: "输入 token", value: model.totals.prompt_tokens },
                    { label: "输出 token", value: model.totals.completion_tokens },
                    { label: "累计成本", value: model.totals.cost.toFixed(4) },
                  ].map((item) => (
                    <div key={item.label} className="rounded-2xl bg-canvas p-3">
                      <p className="text-[11px] text-muted">{item.label}</p>
                      <p className="mt-1 text-[20px] font-semibold text-ink">{item.value}</p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">按 provider / 模型 / 任务</h2></div>
                {model.items.length === 0 ? (
                  <div className="border-t border-line p-4"><StateBlock kind="empty" title="近 7 天没有模型调用" /></div>
                ) : (
                  <table className="w-full border-t border-line text-[12px]">
                    <thead className="bg-canvas text-muted">
                      <tr>
                        <th className="px-4 py-2 text-left">Provider</th>
                        <th className="px-4 py-2 text-left">模型</th>
                        <th className="px-4 py-2 text-left">任务</th>
                        <th className="px-4 py-2 text-right">调用</th>
                        <th className="px-4 py-2 text-right">成本</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {model.items.map((item) => (
                        <tr key={item.provider + item.model + item.task}>
                          <td className="px-4 py-2 text-ink">{item.provider}</td>
                          <td className="px-4 py-2 text-ink">{item.model}</td>
                          <td className="px-4 py-2 text-muted">{item.task}</td>
                          <td className="px-4 py-2 text-right text-ink">{item.calls}</td>
                          <td className="px-4 py-2 text-right text-muted">{item.cost.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </section>
            </>
          ) : null}

          {tab === "trace" ? (
            <div className="grid gap-4 lg:grid-cols-[280px,1fr]">
              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">最近会话</h2></div>
                {sessions.length === 0 ? (
                  <div className="border-t border-line p-4"><StateBlock kind="empty" title="还没有会话" /></div>
                ) : (
                  <ul className="divide-y divide-line border-t border-line">
                    {sessions.map((item) => (
                      <li key={item.id}>
                        <button onClick={() => void openTrace(item.id)} className="w-full px-4 py-2 text-left text-[12px] hover:bg-canvas">
                          <p className="truncate font-medium text-ink">{item.title}</p>
                          <p className="mt-1 text-[11px] text-muted">{item.messages} 条消息 · {formatTime(item.last_at)}</p>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section className="card p-4">
                {!trace ? (
                  <StateBlock kind="empty" title="选一个会话回放" description="左侧点会话，右侧按时间线还原消息、事件与审计" />
                ) : (
                  <>
                    <h2 className="section-title">{trace.session.title}</h2>
                    <p className="mt-1 text-[11px] text-muted">
                      会话 {trace.session.id} · 用户 {trace.session.user_tag} · {formatTime(trace.session.created_at)}
                    </p>
                    <div className="mt-3 space-y-2">
                      {trace.messages.map((message, index) => (
                        <div key={index} className="rounded-2xl bg-canvas p-3 text-[12px]">
                          <span className={message.role === "user" ? "badge bg-primary-soft text-primary" : "badge bg-white text-ink"}>
                            {message.role === "user" ? "用户" : "助手"}
                          </span>
                          <span className="ml-2 text-muted">{formatTime(message.created_at)}</span>
                          <p className="mt-1 whitespace-pre-wrap text-ink">{message.content}</p>
                        </div>
                      ))}
                    </div>
                    {trace.events.length ? (
                      <div className="mt-4">
                        <h3 className="text-[12px] font-medium text-ink">事件</h3>
                        <ul className="mt-2 space-y-1 text-[12px] text-muted">
                          {trace.events.map((event, index) => (
                            <li key={index}>{event.kind} · {formatTime(event.created_at)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {trace.audit.length ? (
                      <div className="mt-4">
                        <h3 className="text-[12px] font-medium text-ink">审计</h3>
                        <ul className="mt-2 space-y-1 text-[12px] text-muted">
                          {trace.audit.map((item, index) => (
                            <li key={index}>{item.action} · {item.actor || "—"} · {formatTime(item.created_at)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </>
                )}
              </section>
            </div>
          ) : null}

          {tab === "audit" && audit ? (
            <>
              <section className="card p-5">
                <h2 className="section-title">近 24 小时审计</h2>
                <p className="mt-2 text-[13px] text-ink">共 {audit.total} 条</p>
                <div className="mt-3 grid gap-4 sm:grid-cols-2">
                  <div>
                    <h3 className="text-[12px] font-medium text-ink">按动作</h3>
                    <ul className="mt-2 space-y-1 text-[12px]">
                      {audit.by_action.map((item) => (
                        <li key={item.action} className="flex items-center justify-between rounded-xl bg-canvas px-3 py-1.5">
                          <span className="text-muted">{item.action}</span>
                          <span className="text-ink">{item.count}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3 className="text-[12px] font-medium text-ink">按操作人</h3>
                    <ul className="mt-2 space-y-1 text-[12px]">
                      {audit.by_actor.map((item) => (
                        <li key={item.actor} className="flex items-center justify-between rounded-xl bg-canvas px-3 py-1.5">
                          <span className="text-muted">{item.actor}</span>
                          <span className="text-ink">{item.count}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </section>

              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">配置变更时间线</h2></div>
                {audit.config_changes.length === 0 ? (
                  <div className="border-t border-line p-4"><StateBlock kind="empty" title="还没有配置发布记录" /></div>
                ) : (
                  <ul className="divide-y divide-line border-t border-line">
                    {audit.config_changes.map((item, index) => (
                      <li key={index} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]">
                        <span className={item.action === "rollback" ? STATUS_STYLE.warn : STATUS_STYLE.ok}>
                          {item.action === "rollback" ? "回滚" : "发布"}
                        </span>
                        <span className="font-medium text-ink">{item.kind} / {item.key}</span>
                        <span className="text-muted">v{item.version}</span>
                        <span className="ml-auto text-[11px] text-muted">{item.actor || "—"} · {formatTime(item.created_at)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">最近审计条目</h2></div>
                <ul className="divide-y divide-line border-t border-line">
                  {audit.recent.map((item, index) => (
                    <li key={index} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]">
                      <span className="font-medium text-ink">{item.action}</span>
                      <span className="text-muted">{item.actor || "匿名"}</span>
                      <span className="ml-auto text-[11px] text-muted">{formatTime(item.created_at)}</span>
                    </li>
                  ))}
                </ul>
              </section>
            </>
          ) : null}

          {tab === "accept" && acceptance ? (
            <>
              <section className="card p-5">
                <div className="flex items-center justify-between">
                  <h2 className="section-title">系统自检</h2>
                  <span className="text-[11px] text-muted">
                    通过 {acceptance.summary.ok} · 关注 {acceptance.summary.warn} · 失败 {acceptance.summary.fail}
                  </span>
                </div>
                <ul className="mt-3 space-y-2">
                  {acceptance.checks.map((check) => (
                    <li key={check.name} className="flex flex-wrap items-center gap-2 rounded-2xl bg-canvas px-3 py-2 text-[12px]">
                      <span className={STATUS_STYLE[check.status] || STATUS_STYLE.warn}>
                        {check.status === "ok" ? "通过" : check.status === "warn" ? "关注" : "失败"}
                      </span>
                      <span className="font-medium text-ink">{check.name}</span>
                      <span className="text-muted">{check.detail}</span>
                    </li>
                  ))}
                </ul>
              </section>

              <section className="card overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3">
                  <h2 className="section-title">25 条验收剧本</h2>
                  <span className="text-[11px] text-muted">
                    可跑 {acceptance.playbook_summary.ready} · 部分 {acceptance.playbook_summary.partial} · 阻塞{" "}
                    {acceptance.playbook_summary.blocked}
                  </span>
                </div>
                <ul className="divide-y divide-line border-t border-line">
                  {acceptance.playbooks.map((item) => (
                    <li key={item.pid} className="px-4 py-2 text-[12px]">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-[11px] text-muted">{item.pid}</span>
                        <span className={STATUS_STYLE[item.status]}>{PLAYBOOK_LABEL[item.status]}</span>
                        <span className="font-medium text-ink">{item.name}</span>
                        <span className="badge bg-canvas text-muted">{item.area}</span>
                        <span className="ml-auto font-mono text-[10px] text-muted">{item.evidence}</span>
                      </div>
                      {item.note ? <p className="mt-1 text-[11px] text-muted">{item.note}</p> : null}
                    </li>
                  ))}
                </ul>
              </section>
            </>
          ) : null}
        </div>
      )}
    </ConsoleShell>
  );
}
