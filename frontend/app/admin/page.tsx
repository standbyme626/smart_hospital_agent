"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  ClipboardList,
  Database,
  Gauge,
  Loader2,
  Play,
  RefreshCw,
  ServerCog,
  ShieldCheck,
  Users,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { ReviewQueue } from "@/components/ReviewQueue";
import { useApp } from "@/lib/app-context";
import {
  createUser,
  getAnalytics,
  getAudit,
  getEvalResults,
  getModelCalls,
  getRagStatus,
  getUsers,
  runEval,
  updateUser,
} from "@/lib/api";
import type { AdminUser, AuditEntry, EvalResult, ModelCall, RagStatus } from "@/lib/types";

type Tab = "overview" | "reviews" | "users" | "audit";

const TABS: { key: Tab; label: string; icon: typeof Activity }[] = [
  { key: "overview", label: "概览", icon: Gauge },
  { key: "reviews", label: "待审核", icon: ClipboardList },
  { key: "users", label: "用户", icon: Users },
  { key: "audit", label: "审计", icon: ShieldCheck },
];

export default function AdminPage() {
  const { user, showToast } = useApp();
  const [tab, setTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [calls, setCalls] = useState<ModelCall[]>([]);
  const [results, setResults] = useState<EvalResult[]>([]);
  const [analytics, setAnalytics] = useState<{ name: string; count: number }[]>([]);
  const [adminUsers, setAdminUsers] = useState<AdminUser[]>([]);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [rag, modelCalls, evalResults, events] = await Promise.all([
        getRagStatus(),
        getModelCalls(),
        getEvalResults(),
        getAnalytics(),
      ]);
      setStatus(rag);
      setCalls(modelCalls);
      setResults(evalResults);
      setAnalytics(events);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    try {
      setAdminUsers(await getUsers());
    } catch (err) {
      showToast(err instanceof Error ? err.message : "加载用户失败");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  const loadAudit = useCallback(async () => {
    setLoading(true);
    try {
      setAudit(await getAudit());
    } catch (err) {
      showToast(err instanceof Error ? err.message : "加载审计失败");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    if (user?.role !== "admin") return;
    if (tab === "overview") load();
    if (tab === "users") loadUsers();
    if (tab === "audit") loadAudit();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.role, tab]);

  async function run() {
    setLoading(true);
    try {
      await runEval(30);
      await load();
      showToast("评测完成");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "评测失败");
      setLoading(false);
    }
  }

  async function submitUser(event: React.FormEvent) {
    event.preventDefault();
    try {
      await createUser({ username: newUsername.trim(), password: newPassword, role: "patient" });
      setNewUsername("");
      setNewPassword("");
      showToast("患者账号已创建");
      await loadUsers();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "创建失败");
    }
  }

  async function toggleUser(target: AdminUser) {
    try {
      await updateUser(target.id, { enabled: !target.enabled });
      await loadUsers();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "更新失败");
    }
  }

  async function resetPassword(target: AdminUser) {
    const password = window.prompt(`为 ${target.username} 设置新密码（至少 6 位）`);
    if (!password) return;
    try {
      await updateUser(target.id, { password });
      showToast("密码已重置");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "重置失败");
    }
  }

  if (user?.role !== "admin") {
    return (
      <AppShell title="运营后台" subtitle="仅管理员可访问">
        <div className="card p-6 text-[13px] text-muted">当前账号没有管理员权限。请使用管理员账号登录。</div>
      </AppShell>
    );
  }

  return (
    <AppShell title="运营后台" subtitle="审核、用户、检索、模型与评测">
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {TABS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.key}
                onClick={() => setTab(item.key)}
                className={clsx("btn", tab === item.key ? "btn-primary" : "btn-ghost")}
              >
                <Icon size={14} />{item.label}
              </button>
            );
          })}
        </div>

        {tab === "reviews" ? <ReviewQueue /> : null}

        {tab === "users" ? (
          <section className="card overflow-hidden">
            <div className="border-b border-line px-4 py-3">
              <p className="flex items-center gap-1.5 text-[12px] font-medium text-muted">
                <Users size={14} className="text-primary" />用户管理
              </p>
              <form onSubmit={submitUser} className="mt-3 flex flex-wrap items-end gap-2">
                <label className="block">
                  <span className="label">用户名</span>
                  <input
                    className="input"
                    value={newUsername}
                    onChange={(event) => setNewUsername(event.target.value)}
                    minLength={1}
                    maxLength={64}
                    required
                  />
                </label>
                <label className="block">
                  <span className="label">初始密码</span>
                  <input
                    className="input"
                    type="password"
                    value={newPassword}
                    onChange={(event) => setNewPassword(event.target.value)}
                    minLength={6}
                    required
                  />
                </label>
                <button type="submit" className="btn-primary">创建患者账号</button>
                <button type="button" onClick={loadUsers} disabled={loading} className="btn-ghost">
                  <RefreshCw size={14} className={loading ? "animate-spin" : ""} />刷新
                </button>
              </form>
            </div>
            <div className="scrollbar-thin overflow-x-auto">
              <table className="w-full text-left text-[12px]">
                <thead className="bg-canvas text-muted">
                  <tr>
                    <th className="px-4 py-2 font-medium">用户名</th>
                    <th className="px-4 py-2 font-medium">角色</th>
                    <th className="px-4 py-2 font-medium">状态</th>
                    <th className="px-4 py-2 font-medium">创建时间</th>
                    <th className="px-4 py-2 font-medium">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {adminUsers.length === 0 ? (
                    <tr><td colSpan={5} className="px-4 py-6 text-center text-muted">暂无用户</td></tr>
                  ) : adminUsers.map((item) => (
                    <tr key={item.id} className="border-t border-line/60 text-ink">
                      <td className="px-4 py-2">{item.username}</td>
                      <td className="px-4 py-2">{item.role === "admin" ? "管理员" : "患者"}</td>
                      <td className="px-4 py-2">
                        <span className={clsx("badge", item.enabled ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700")}>
                          {item.enabled ? "启用" : "停用"}
                        </span>
                      </td>
                      <td className="px-4 py-2">{item.created_at ? new Date(item.created_at).toLocaleString() : "—"}</td>
                      <td className="px-4 py-2">
                        <div className="flex gap-2">
                          <button onClick={() => toggleUser(item)} className="btn-ghost">{item.enabled ? "停用" : "启用"}</button>
                          <button onClick={() => resetPassword(item)} className="btn-ghost">重置密码</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        {tab === "audit" ? (
          <section className="card overflow-hidden">
            <div className="flex items-center justify-between border-b border-line px-4 py-3">
              <p className="flex items-center gap-1.5 text-[12px] font-medium text-muted">
                <ShieldCheck size={14} className="text-emerald-500" />审计日志（最近 100 条）
              </p>
              <button onClick={loadAudit} disabled={loading} className="btn-ghost">
                <RefreshCw size={14} className={loading ? "animate-spin" : ""} />刷新
              </button>
            </div>
            <div className="scrollbar-thin overflow-x-auto">
              <table className="w-full text-left text-[12px]">
                <thead className="bg-canvas text-muted">
                  <tr>
                    <th className="px-4 py-2 font-medium">时间</th>
                    <th className="px-4 py-2 font-medium">操作者</th>
                    <th className="px-4 py-2 font-medium">动作</th>
                    <th className="px-4 py-2 font-medium">详情</th>
                  </tr>
                </thead>
                <tbody>
                  {audit.length === 0 ? (
                    <tr><td colSpan={4} className="px-4 py-6 text-center text-muted">暂无审计记录</td></tr>
                  ) : audit.map((item, index) => (
                    <tr key={index} className="border-t border-line/60 text-ink">
                      <td className="px-4 py-2">{new Date(item.created_at).toLocaleString()}</td>
                      <td className="px-4 py-2">{item.actor || "—"}</td>
                      <td className="px-4 py-2">{item.action}</td>
                      <td className="max-w-md truncate px-4 py-2 text-[11px] text-muted">{JSON.stringify(item.detail)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        {tab === "overview" ? (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="card p-4">
                <p className="flex items-center gap-1.5 text-[12px] text-muted"><Database size={13} className="text-primary" />文档数</p>
                <p className="mt-2 text-2xl font-semibold text-ink">{status?.documents ?? "—"}</p>
              </div>
              <div className="card p-4">
                <p className="flex items-center gap-1.5 text-[12px] text-muted"><ServerCog size={13} className="text-primary" />模型</p>
                <p className="mt-2 text-[13px] text-ink">
                  opencode {status?.models.opencode ? "✓" : "—"} · SiliconFlow {status?.models.siliconflow ? "✓" : "—"} · Ollama {status?.models.ollama ? "✓" : "—"}
                </p>
              </div>
              <div className="card p-4">
                <p className="flex items-center gap-1.5 text-[12px] text-muted"><ShieldCheck size={13} className="text-emerald-500" />MCP 医院系统</p>
                <p className="mt-2 text-[13px] text-ink">{status?.mcp ? "已连接（mock）" : "未连接"}</p>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <button onClick={run} disabled={loading} className="btn-primary"><Play size={15} />运行评测</button>
              <button onClick={load} disabled={loading} className="btn-ghost"><RefreshCw size={15} className={loading ? "animate-spin" : ""} />刷新</button>
            </div>

            <section className="card overflow-hidden">
              <div className="border-b border-line px-4 py-3 text-[12px] font-medium text-muted">模型调用</div>
              <div className="scrollbar-thin overflow-x-auto">
                <table className="w-full text-left text-[12px]">
                  <thead className="bg-canvas text-muted">
                    <tr>
                      <th className="px-4 py-2 font-medium">provider</th>
                      <th className="px-4 py-2 font-medium">model</th>
                      <th className="px-4 py-2 font-medium">task</th>
                      <th className="px-4 py-2 font-medium">calls</th>
                      <th className="px-4 py-2 font-medium">tokens</th>
                      <th className="px-4 py-2 font-medium">cost</th>
                      <th className="px-4 py-2 font-medium">avg ms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calls.length === 0 ? (
                      <tr><td colSpan={7} className="px-4 py-6 text-center text-muted">暂无调用记录</td></tr>
                    ) : calls.map((call, index) => (
                      <tr key={index} className="border-t border-line/60 text-ink">
                        <td className="px-4 py-2">{call.provider}</td>
                        <td className="px-4 py-2">{call.model}</td>
                        <td className="px-4 py-2">{call.task}</td>
                        <td className="px-4 py-2">{call.calls}</td>
                        <td className="px-4 py-2">{call.prompt_tokens}/{call.completion_tokens}</td>
                        <td className="px-4 py-2">${Number(call.cost).toFixed(6)}</td>
                        <td className="px-4 py-2">{call.avg_latency_ms}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <div className="grid gap-4 lg:grid-cols-2">
              <section className="card p-4">
                <p className="flex items-center gap-1.5 text-[12px] font-medium text-muted"><BarChart3 size={14} className="text-primary" />埋点（30 天）</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {analytics.length === 0 ? <p className="text-[12px] text-muted">暂无埋点数据</p> : analytics.map((item) => (
                    <span key={item.name} className="chip">{item.name} · {item.count}</span>
                  ))}
                </div>
              </section>
              <section className="card p-4">
                <p className="flex items-center gap-1.5 text-[12px] font-medium text-muted"><Gauge size={14} className="text-primary" />评测结果</p>
                <div className="mt-3 space-y-2">
                  {results.length === 0 ? <p className="text-[12px] text-muted">还没有评测记录</p> : results.slice(0, 3).map((item) => (
                    <div key={item.id} className="rounded-xl bg-canvas p-3">
                      <p className="text-[11px] text-muted">#{item.id} · {item.name}</p>
                      <pre className="scrollbar-thin mt-1 max-h-28 overflow-auto text-[10px] leading-4 text-ink">{JSON.stringify(item.metrics, null, 2)}</pre>
                    </div>
                  ))}
                </div>
              </section>
            </div>

            <div className="flex items-start gap-2 rounded-2xl border border-line bg-card p-3 text-[11px] leading-5 text-muted">
              <Activity size={14} className="mt-0.5 shrink-0 text-primary" />
              这是内部运营台，不属于消费者导航。支付申请必须在这里人工批准后才会调用医院 MCP 完成扣款。
            </div>
          </>
        ) : null}

        {loading && tab !== "overview" ? (
          <div className="flex items-center gap-2 text-[12px] text-muted"><Loader2 size={14} className="animate-spin" />加载中…</div>
        ) : null}
      </div>
    </AppShell>
  );
}
