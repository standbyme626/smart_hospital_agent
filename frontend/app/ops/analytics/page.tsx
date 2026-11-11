"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, BarChart3 } from "lucide-react";
import { OpsNav } from "@/components/OpsNav";
import { StateBlock } from "@/components/ui";
import { getOpsAnalytics, type OpsAnalytics } from "@/lib/api";

export default function OpsAnalyticsPage() {
  const [data, setData] = useState<OpsAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getOpsAnalytics().then(setData).catch(() => setFailed(true)).finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-dvh bg-canvas">
      <header className="border-b border-line bg-white/90">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
          <BarChart3 size={18} className="text-primary" />
          <div><p className="text-[15px] font-semibold text-ink">运营分析</p><p className="text-[11px] text-muted">规模与事件计数（真实数据）</p></div>
          <Link href="/" className="ml-auto inline-flex items-center gap-1 rounded-full bg-white px-3 py-1 text-[11px] text-muted ring-1 ring-line hover:text-primary"><ArrowLeft size={12} /> 患者端</Link>
        </div>
        <div className="mx-auto max-w-5xl px-4 pb-3"><OpsNav /></div>
      </header>
      <main className="mx-auto max-w-5xl space-y-3 px-4 py-5">
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={() => window.location.reload()} /></div>
        ) : data ? (
          <>
            <div className="grid gap-3 sm:grid-cols-4">
              {[
                { label: "用户", value: data.totals.users },
                { label: "会话", value: data.totals.sessions },
                { label: "报告", value: data.totals.reports },
                { label: "挂号", value: data.totals.bookings },
              ].map((item) => (
                <div key={item.label} className="card p-4"><p className="text-[12px] text-muted">{item.label}</p><p className="mt-1 text-[24px] font-semibold text-ink">{item.value}</p></div>
              ))}
            </div>
            <section className="card overflow-hidden">
              <div className="px-4 py-3"><h2 className="section-title">事件计数</h2></div>
              <table className="w-full text-left text-[12px]">
                <thead className="bg-canvas text-muted"><tr><th className="px-3 py-2">事件</th><th className="px-3 py-2">次数</th></tr></thead>
                <tbody>
                  {data.events.length === 0 ? <tr><td colSpan={2} className="px-3 py-3 text-muted">暂无事件</td></tr> : data.events.map((event) => (
                    <tr key={event.name} className="border-t border-line"><td className="px-3 py-2 text-ink">{event.name}</td><td className="px-3 py-2 text-muted">{event.count}</td></tr>
                  ))}
                </tbody>
              </table>
            </section>
          </>
        ) : null}
        <p className="px-1 text-[11px] leading-5 text-muted">漏斗与成本分析在后续版本接入；当前为真实事件聚合。</p>
      </main>
    </div>
  );
}
