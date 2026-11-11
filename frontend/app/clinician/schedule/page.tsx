"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, CalendarDays } from "lucide-react";
import { StateBlock } from "@/components/ui";
import { getMySchedules, type ScheduleItem } from "@/lib/api";

export default function ClinicianSchedulePage() {
  const [items, setItems] = useState<ScheduleItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [denied, setDenied] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    setDenied(false);
    try {
      setItems(await getMySchedules());
    } catch (error) {
      const status = (error as { status?: number } | null)?.status;
      if (status === 401 || status === 403) setDenied(true);
      else setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="min-h-dvh bg-canvas">
      <header className="sticky top-0 z-20 border-b border-line bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-4xl items-center gap-3 px-4 py-3">
          <CalendarDays size={18} className="text-primary" />
          <div><p className="text-[15px] font-semibold text-ink">我的排班</p><p className="text-[11px] text-muted">平台维护的排班（HIS 排班接口未开放）</p></div>
          <Link href="/clinician/" className="ml-auto rounded-full bg-white px-3 py-1 text-[11px] text-muted ring-1 ring-line hover:text-primary">返回工作台</Link>
        </div>
      </header>
      <main className="mx-auto max-w-4xl space-y-4 px-4 py-5">
        {denied ? <div className="card p-3"><StateBlock kind="denied" title="当前账号无医生权限" /></div> : loading ? (
          <div className="card h-32 animate-pulse" />
        ) : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="还没有排班" description="平台维护排班后显示在这里" /></div>
        ) : (
          <section className="card overflow-hidden">
            <table className="w-full text-left text-[12px]">
              <thead className="bg-canvas text-muted"><tr><th className="px-3 py-2">星期</th><th className="px-3 py-2">时段</th><th className="px-3 py-2">科室</th><th className="px-3 py-2">状态</th><th className="px-3 py-2">备注</th></tr></thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id} className="border-t border-line">
                    <td className="px-3 py-2 text-ink">{item.weekday}</td>
                    <td className="px-3 py-2 text-ink">{item.period === "am" ? "上午" : "下午"}</td>
                    <td className="px-3 py-2 text-muted">{item.department || "—"}</td>
                    <td className="px-3 py-2"><span className="badge">{item.status === "clinic" ? "出诊" : item.status}</span></td>
                    <td className="px-3 py-2 text-muted">{item.note || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
        <p className="px-1 text-[11px] leading-5 text-muted">排班为平台数据；停诊/加号需在后续版本接入审批流。</p>
      </main>
    </div>
  );
}
