"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, BedDouble } from "lucide-react";
import { StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import { getClinicianQueue, type ClinicianQueueItem } from "@/lib/api";

export default function ClinicianWardPage() {
  const [items, setItems] = useState<ClinicianQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setItems(await getClinicianQueue());
    } catch {
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <ConsoleShell end="clinician" title="移动查房" subtitle="手机端只看摘要">
      <div className="space-y-4">
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="今日没有患者" description="有挂号后这里会显示待查房患者" /></div>
        ) : items.map((item) => (
          <div key={item.id} className="card p-4">
            <div className="flex items-center gap-2">
              <span className="text-[14px] font-semibold text-ink">{item.member_name || item.member_id}</span>
              <span className={item.payment_status === "paid" ? "badge bg-emerald-50 text-emerald-700" : "badge bg-amber-50 text-amber-700"}>
                {item.payment_status === "paid" ? "已缴费" : "待缴费"}
              </span>
              <span className="ml-auto text-[11px] text-muted">{item.appointment_date || ""}</span>
            </div>
            <p className="mt-1 text-[12px] text-muted">{item.department || "未填科室"} · {item.doctor_name || "未指定"}</p>
            <Link href="/clinician/" className="mt-2 inline-block text-[12px] font-medium text-primary">查看患者 360 →</Link>
          </div>
        ))}
        <p className="px-1 text-[11px] leading-5 text-muted">移动端只读：签名、开医嘱请回桌面工作台。</p>
      </div>
    </ConsoleShell>
  );
}
