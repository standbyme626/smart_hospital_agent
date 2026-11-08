"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Database,
  Gauge,
  Loader2,
  Play,
  RefreshCw,
  ServerCog,
  ShieldCheck,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useApp } from "@/lib/app-context";
import { getAnalytics, getEvalResults, getModelCalls, getRagStatus, runEval } from "@/lib/api";
import type { EvalResult, ModelCall, RagStatus } from "@/lib/types";

export default function AdminPage() {
  const { user, showToast } = useApp();
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [calls, setCalls] = useState<ModelCall[]>([]);
  const [results, setResults] = useState<EvalResult[]>([]);
  const [analytics, setAnalytics] = useState<{ name: string; count: number }[]>([]);
  const [loading, setLoading] = useState(false);

  async function load() {
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
  }

  useEffect(() => {
    if (user?.role === "admin") load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.role]);

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

  if (user?.role !== "admin") {
    return (
      <AppShell title="运营后台" subtitle="仅管理员可访问">
        <div className="card p-6 text-[13px] text-muted">当前账号没有管理员权限。请使用管理员账号登录。</div>
      </AppShell>
    );
  }

  return (
    <AppShell title="运营后台" subtitle="检索、模型、评测与埋点">
      <div className="space-y-4">
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
          这是内部运营台，不属于消费者导航。真实医院 MCP 接入后，模型调用、支付和随访数据会继续写入同一套审计表。
        </div>
      </div>
    </AppShell>
  );
}
