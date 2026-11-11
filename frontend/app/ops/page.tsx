"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, CheckCircle2, ClipboardList, HandHeart, ShieldCheck } from "lucide-react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import {
  claimOpsHandoff,
  getOpsFindings,
  getOpsHandoff,
  getOpsSummary,
  resolveOpsFinding,
  type OpsFinding,
  type OpsHandoff,
  type OpsSummary,
} from "@/lib/api";
import { relativeTime } from "@/lib/format";

type Tab = "overview" | "reconciliation" | "handoff";

const SEVERITY: Record<string, string> = {
  high: "bg-rose-50 text-rose-700",
  medium: "bg-amber-50 text-amber-700",
  low: "bg-canvas text-muted",
};

export default function OpsPage() {
  const [tab, setTab] = useState<Tab>("overview");
  const [summary, setSummary] = useState<OpsSummary | null>(null);
  const [findings, setFindings] = useState<OpsFinding[]>([]);
  const [handoff, setHandoff] = useState<OpsHandoff[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [denied, setDenied] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setFailed(false);
    setDenied(false);
    try {
      const [s, f, h] = await Promise.all([getOpsSummary(), getOpsFindings(), getOpsHandoff()]);
      setSummary(s);
      setFindings(f);
      setHandoff(h);
    } catch (error) {
      const status = (error as { status?: number } | null)?.status;
      if (status === 401 || status === 403) setDenied(true);
      else setFailed(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function resolve(id: string) {
    setBusyId(id);
    try {
      await resolveOpsFinding(id);
      setFindings((prev) => prev.map((item) => (item.id === id ? { ...item, status: "resolved" } : item)));
      setSummary((prev) => (prev ? { ...prev, open_findings: Math.max(0, prev.open_findings - 1) } : prev));
    } catch {
      // 失败保持原状态，可重试
    } finally {
      setBusyId(null);
    }
  }

  async function claim(id: string) {
    setBusyId(id);
    try {
      await claimOpsHandoff(id);
      setHandoff((prev) => prev.map((item) => (item.id === id ? { ...item, status: "claimed" } : item)));
      setSummary((prev) => (prev ? { ...prev, handoff_waiting: Math.max(0, prev.handoff_waiting - 1) } : prev));
    } catch {
      // 失败保持原状态
    } finally {
      setBusyId(null);
    }
  }

  const openFindings = findings.filter((item) => item.status === "open");
  const waitingHandoff = handoff.filter((item) => item.status === "requested");

  return (
    <ConsoleShell end="ops" title="运营 / 质控台" subtitle="支付异常、对账差异、人工接管、审计、分析、医保、危机与通知记录">

      <div className="space-y-4">
        <OfflineBanner />

        {denied ? (
          <div className="card p-3">
            <StateBlock kind="denied" title="当前账号无运营权限" description="运营台只对 ops / admin 角色开放" />
            <div className="mt-2 text-center"><Link href="/login/" className="text-[12px] font-medium text-primary">去登录</Link></div>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              {([
                { key: "overview", label: "概览" },
                { key: "reconciliation", label: "对账差异" },
                { key: "handoff", label: "人工接管" },
              ] as Array<{ key: Tab; label: string }>).map((item) => (
                <button
                  key={item.key}
                  onClick={() => setTab(item.key)}
                  className={tab === item.key ? "chip border-primary bg-primary text-white" : "chip hover:border-primary/40 hover:text-primary"}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {loading ? (
              <div className="grid gap-3 sm:grid-cols-4">
                {[0, 1, 2, 3].map((index) => <div key={index} className="h-24 animate-pulse rounded-2xl bg-white" />)}
              </div>
            ) : failed ? (
              <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
            ) : tab === "overview" ? (
              <>
                <div className="grid gap-3 sm:grid-cols-4">
                  {[
                    { label: "待审核", value: summary?.pending_reviews ?? 0, hint: "支付/临床复核" },
                    { label: "对账差异", value: summary?.open_findings ?? 0, hint: "未处理 finding" },
                    { label: "待接管", value: summary?.handoff_waiting ?? 0, hint: "人工介入队列" },
                    { label: "今日已支付", value: summary?.paid_today ?? 0, hint: "医院直收" },
                  ].map((item) => (
                    <div key={item.label} className="card p-4">
                      <p className="text-[12px] text-muted">{item.label}</p>
                      <p className="mt-1 text-[26px] font-semibold text-ink">{item.value}</p>
                      <p className="mt-1 text-[11px] text-muted">{item.hint}</p>
                    </div>
                  ))}
                </div>
                <div className="card p-5">
                  <h2 className="section-title">今天要处理的事</h2>
                  <ul className="mt-3 space-y-2 text-[13px] text-ink">
                    <li className="flex items-center gap-2 rounded-2xl bg-canvas px-3 py-2"><ClipboardList size={14} className="text-primary" /> 对账差异 {openFindings.length} 条待处理</li>
                    <li className="flex items-center gap-2 rounded-2xl bg-canvas px-3 py-2"><HandHeart size={14} className="text-primary" /> 人工接管 {waitingHandoff.length} 条待认领</li>
                    <li className="flex items-center gap-2 rounded-2xl bg-canvas px-3 py-2"><CheckCircle2 size={14} className="text-emerald-600" /> 正常支付自动放行，不进入人工队列</li>
                  </ul>
                </div>
              </>
            ) : tab === "reconciliation" ? (
              <section className="card overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3">
                  <h2 className="section-title">对账差异</h2>
                  <span className="text-[11px] text-muted">{openFindings.length} 条未处理</span>
                </div>
                <div className="border-t border-line">
                  {findings.length === 0 ? (
                    <div className="p-3"><StateBlock kind="empty" title="没有对账差异" description="HIS 与本地投影一致时会显示为空" /></div>
                  ) : (
                    <ul className="divide-y divide-line">
                      {findings.map((item) => (
                        <li key={item.id} className="flex flex-wrap items-center gap-3 px-4 py-3">
                          <span className={`badge ${SEVERITY[item.severity] || SEVERITY.low}`}>{item.severity === "high" ? "高" : item.severity === "medium" ? "中" : "低"}</span>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-[13px] font-medium text-ink">{item.kind}</p>
                            <p className="mt-0.5 truncate text-[11px] text-muted">{item.detail || (item.local_ref || "-") + " ↔ " + (item.his_ref || "-")}</p>
                          </div>
                          <span className="text-[11px] text-muted">{relativeTime(item.detected_at)}</span>
                          {item.status === "open" ? (
                            <button onClick={() => resolve(item.id)} disabled={busyId === item.id} className="rounded-full bg-primary px-3 py-1 text-[11px] font-medium text-white disabled:opacity-50">
                              {busyId === item.id ? "处理中…" : "标记已处理"}
                            </button>
                          ) : (
                            <span className="badge bg-emerald-50 text-emerald-700">已处理</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </section>
            ) : (
              <section className="card overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3">
                  <h2 className="section-title">人工接管</h2>
                  <span className="text-[11px] text-muted">{waitingHandoff.length} 条待认领</span>
                </div>
                <div className="border-t border-line">
                  {handoff.length === 0 ? (
                    <div className="p-3"><StateBlock kind="empty" title="没有待接管任务" description="低置信、用户要求人工、危机事件会进入这里" /></div>
                  ) : (
                    <ul className="divide-y divide-line">
                      {handoff.map((item) => (
                        <li key={item.id} className="flex flex-wrap items-center gap-3 px-4 py-3">
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-[13px] font-medium text-ink">会话 {item.ref_id}</p>
                            <p className="mt-0.5 text-[11px] text-muted">创建 {relativeTime(item.created_at)}{item.assignee ? " · 已由 " + item.assignee + " 接管" : ""}</p>
                          </div>
                          {item.status === "requested" ? (
                            <button onClick={() => claim(item.id)} disabled={busyId === item.id} className="rounded-full bg-primary px-3 py-1 text-[11px] font-medium text-white disabled:opacity-50">
                              {busyId === item.id ? "接管中…" : "接管"}
                            </button>
                          ) : (
                            <span className="badge bg-emerald-50 text-emerald-700">已接管</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </ConsoleShell>
  );
}
