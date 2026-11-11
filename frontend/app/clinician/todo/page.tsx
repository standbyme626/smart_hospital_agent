"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ListChecks } from "lucide-react";
import { StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import { getClinicianQueue, type ClinicianQueueItem } from "@/lib/api";

export default function ClinicianTodoPage() {
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

  const unpaid = items.filter((item) => item.payment_status !== "paid");
  const unregistered = items.filter((item) => item.status !== "registered" && item.status !== "completed");

  return (
    <ConsoleShell end="clinician" title="移动待办" subtitle="待缴费与待接诊">
      <div className="space-y-4">
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="今天没有待办" description="有挂号后这里会出现待缴费/待接诊" /></div>
        ) : (
          <>
            <section className="card p-4">
              <h2 className="section-title">待缴费（{unpaid.length}）</h2>
              <ul className="mt-2 space-y-1 text-[12px] text-muted">
                {unpaid.length === 0 ? <li>无</li> : unpaid.map((item) => <li key={item.id}>{item.member_name || item.member_id} · {item.department || "未填科室"}</li>)}
              </ul>
            </section>
            <section className="card p-4">
              <h2 className="section-title">待接诊 / 待登记（{unregistered.length}）</h2>
              <ul className="mt-2 space-y-1 text-[12px] text-muted">
                {unregistered.length === 0 ? <li>无</li> : unregistered.map((item) => <li key={item.id}>{item.member_name || item.member_id} · {item.status}</li>)}
              </ul>
            </section>
          </>
        )}
        <p className="px-1 text-[11px] leading-5 text-muted">签名类操作请回桌面工作台完成。</p>
      </div>
    </ConsoleShell>
  );
}
