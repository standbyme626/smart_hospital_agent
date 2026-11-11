"use client";

import { useEffect, useState } from "react";
import { Clock3 } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { getQueueToday, type QueueToday } from "@/lib/api";

export default function QueuePage() {
  const [data, setData] = useState<QueueToday | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setData(await getQueueToday());
    } catch {
      setData(null);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <AppShell title="排队叫号" subtitle="今日候诊进度与到院流程" actions={<SourceBadge live />}>
      <div className="mx-auto max-w-3xl space-y-4">
        <OfflineBanner />
        {loading ? <div className="card h-28 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : !data?.mine ? (
          <div className="card p-3"><StateBlock kind="empty" title="今天没有取号记录" description="挂号缴费后会自动生成候诊号" /></div>
        ) : (
          <>
            <section className="card p-5">
              <div className="flex items-center gap-2 text-primary"><Clock3 size={18} /><h2 className="text-[16px] font-semibold">候诊叫号</h2></div>
              <div className="mt-5 flex flex-wrap items-end gap-8">
                <div>
                  <p className="text-[11px] text-muted">当前叫号</p>
                  <p className="text-[32px] font-semibold text-ink">{data.current_called?.ticket_no || "—"}</p>
                </div>
                <div>
                  <p className="text-[11px] text-muted">我的号码</p>
                  <p className="text-[32px] font-semibold text-primary">{data.mine.ticket_no}</p>
                </div>
                <div>
                  <p className="text-[11px] text-muted">前面等待</p>
                  <p className="text-[20px] font-semibold text-ink">{data.waiting_ahead} 人</p>
                </div>
              </div>
              <p className="mt-4 rounded-2xl bg-canvas px-3 py-2 text-[12px] text-muted">
                科室：{data.mine.department || "未指定"} · 状态：{data.mine.status === "waiting" ? "候诊中" : data.mine.status}
              </p>
            </section>
            <section className="card p-5">
              <h2 className="section-title">到院流程</h2>
              <div className="mt-3 flex flex-wrap items-center gap-3 text-[12px]">
                {["取号", "签到", "候诊", "就诊", "检查/取药"].map((step, index) => (
                  <div key={step} className="flex items-center gap-2">
                    <span className={index <= 2 ? "flex h-6 w-6 items-center justify-center rounded-full bg-primary text-[11px] text-white" : "flex h-6 w-6 items-center justify-center rounded-full bg-canvas text-[11px] text-muted"}>{index + 1}</span>
                    <span className={index <= 2 ? "text-ink" : "text-muted"}>{step}</span>
                    {index < 4 ? <span className="h-px w-4 bg-line" /> : null}
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[12px] leading-5 text-muted">地点：3 号楼 2 层 · 请提前 15 分钟到院签到；过号需重新排队。</p>
            </section>
          </>
        )}
        <p className="text-center text-[11px] leading-5 text-muted">叫号数据来自平台排队记录；接入 HIS 分诊叫号后自动同步。</p>
      </div>
    </AppShell>
  );
}
