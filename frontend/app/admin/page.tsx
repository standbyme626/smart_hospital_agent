"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  ArrowLeft,
  Database,
  Gauge,
  Loader2,
  Play,
  RefreshCw,
  ServerCog,
  ShieldCheck,
} from "lucide-react";
import {
  UnauthorizedError,
  getEvalResults,
  getModelCalls,
  getRagStatus,
  runEval,
} from "@/lib/api";
import type { EvalResult, ModelCall, RagStatus } from "@/lib/types";

export default function AdminPage() {
  const router = useRouter();
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [calls, setCalls] = useState<ModelCall[]>([]);
  const [results, setResults] = useState<EvalResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleError = useCallback(
    (err: unknown) => {
      if (err instanceof UnauthorizedError) {
        router.push("/login");
        return;
      }
      setError(err instanceof Error ? err.message : "请求失败");
    },
    [router],
  );

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [rag, modelCalls, evalResults] = await Promise.all([
        getRagStatus(),
        getModelCalls(),
        getEvalResults(),
      ]);
      setStatus(rag);
      setCalls(modelCalls);
      setResults(evalResults);
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  useEffect(() => {
    load();
  }, [load]);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      await runEval(30);
      await load();
    } catch (err) {
      handleError(err);
      setLoading(false);
    }
  }

  return (
    <div className="min-h-dvh px-4 py-6 sm:px-6">
      <div className="mx-auto max-w-5xl">
        <div className="flex items-center justify-between">
          <Link href="/" className="inline-flex items-center gap-2 text-xs text-slate-400 hover:text-white">
            <ArrowLeft size={14} />
            返回问诊
          </Link>
          <span className="flex items-center gap-2 text-sm font-semibold text-white">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary2 text-slate-950">
              <Activity size={16} strokeWidth={2.5} />
            </span>
            运营后台
          </span>
        </div>

        {error ? <p className="mt-4 rounded-lg border border-rose/30 bg-rose/10 px-3 py-2 text-xs text-rose">{error}</p> : null}

        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          <div className="surface p-4">
            <p className="flex items-center gap-1.5 text-xs text-slate-400">
              <Database size={13} className="text-primary" />
              文档数
            </p>
            <p className="mt-2 text-2xl font-semibold text-white">{status?.documents ?? "—"}</p>
          </div>
          <div className="surface p-4">
            <p className="flex items-center gap-1.5 text-xs text-slate-400">
              <ServerCog size={13} className="text-primary" />
              模型
            </p>
            <p className="mt-2 text-sm text-slate-200">
              opencode {status?.models.opencode ? "✓" : "—"} · SiliconFlow {status?.models.siliconflow ? "✓" : "—"} · Ollama{" "}
              {status?.models.ollama ? "✓" : "—"}
            </p>
          </div>
          <div className="surface p-4">
            <p className="flex items-center gap-1.5 text-xs text-slate-400">
              <ShieldCheck size={13} className="text-mint" />
              MCP 医院系统
            </p>
            <p className="mt-2 text-sm text-slate-200">{status?.mcp ? "已连接（mock）" : "未连接"}</p>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <button className="btn-primary" onClick={run} disabled={loading}>
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
            运行评测
          </button>
          <button className="btn-ghost" onClick={load} disabled={loading}>
            <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
            刷新
          </button>
        </div>

        <div className="mt-6 surface overflow-hidden">
          <div className="border-b border-line px-4 py-3 text-xs font-medium uppercase tracking-wider text-slate-400">
            模型调用
          </div>
          <div className="scrollbar-thin overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-panel2 text-slate-400">
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
                  <tr>
                    <td colSpan={7} className="px-4 py-6 text-center text-slate-500">
                      暂无调用记录
                    </td>
                  </tr>
                ) : (
                  calls.map((call, index) => (
                    <tr key={index} className="border-t border-line/60 text-slate-300">
                      <td className="px-4 py-2">{call.provider}</td>
                      <td className="px-4 py-2">{call.model}</td>
                      <td className="px-4 py-2">{call.task}</td>
                      <td className="px-4 py-2">{call.calls}</td>
                      <td className="px-4 py-2">
                        {call.prompt_tokens}/{call.completion_tokens}
                      </td>
                      <td className="px-4 py-2">${Number(call.cost).toFixed(6)}</td>
                      <td className="px-4 py-2">{call.avg_latency_ms}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-4 surface p-4">
          <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-slate-400">
            <Gauge size={14} className="text-primary" />
            评测结果
          </p>
          <div className="mt-3 space-y-3">
            {results.length === 0 ? (
              <p className="text-xs text-slate-500">还没有评测记录，点击“运行评测”。</p>
            ) : (
              results.slice(0, 5).map((result) => (
                <div key={result.id} className="rounded-xl border border-line bg-panel2 p-3">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>
                      #{result.id} · {result.name}
                    </span>
                    <span>{new Date(result.created_at).toLocaleString()}</span>
                  </div>
                  <pre className="scrollbar-thin mt-2 max-h-48 overflow-auto text-[11px] leading-5 text-slate-300">
                    {JSON.stringify(result.metrics, null, 2)}
                  </pre>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
