"use client";

import { useEffect, useState } from "react";
import { PackageCheck } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { getDispensing, type DispensingItem } from "@/lib/api";
import { relativeTime } from "@/lib/format";

const STEPS = [
  { key: "waiting_dispense", label: "已缴费" },
  { key: "dispensing", label: "药房配药" },
  { key: "ready", label: "可取药" },
  { key: "picked_up", label: "已取药" },
];

export default function PickupPage() {
  const [items, setItems] = useState<DispensingItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setItems(await getDispensing());
    } catch {
      setItems([]);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <AppShell title="取药进度" subtitle="缴费后药房配药状态" actions={<SourceBadge live />}>
      <div className="mx-auto max-w-3xl space-y-4">
        <OfflineBanner />
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="暂无取药记录" description="挂号缴费后，药房配药进度会显示在这里" /></div>
        ) : (
          items.map((item) => {
            const index = Math.max(0, STEPS.findIndex((step) => step.key === item.status));
            return (
              <section key={item.id} className="card p-5">
                <div className="flex items-center gap-2 text-primary">
                  <PackageCheck size={18} />
                  <h2 className="text-[16px] font-semibold">处方取药</h2>
                  <span className="ml-auto text-[11px] text-muted">{relativeTime(item.created_at)}</span>
                </div>
                <div className="mt-4 flex flex-wrap items-center gap-3">
                  {STEPS.map((step, i) => (
                    <div key={step.key} className="flex items-center gap-2">
                      <span className={i <= index ? "flex h-6 w-6 items-center justify-center rounded-full bg-primary text-[11px] text-white" : "flex h-6 w-6 items-center justify-center rounded-full bg-canvas text-[11px] text-muted"}>{i + 1}</span>
                      <span className={i <= index ? "text-[12px] text-ink" : "text-[12px] text-muted"}>{step.label}</span>
                      {i < STEPS.length - 1 ? <span className="h-px w-6 bg-line" /> : null}
                    </div>
                  ))}
                </div>
                <p className="mt-4 rounded-2xl bg-canvas px-3 py-2 text-[12px] text-muted">
                  {item.window ? "取药窗口：" + item.window : "取药窗口待分配"}
                  {item.ready_at ? " · 可取药时间 " + relativeTime(item.ready_at) : ""}
                </p>
              </section>
            );
          })
        )}
        <p className="text-center text-[11px] leading-5 text-muted">药房进度由平台记录；HIS 写接口未开通前不写入医院系统。</p>
      </div>
    </AppShell>
  );
}
