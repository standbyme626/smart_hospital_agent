"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  CalendarClock,
  CheckCircle2,
  ChevronRight,
  FileText,
  HeartPulse,
  Pill,
  Stethoscope,
  Users,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { useApp } from "@/lib/app-context";
import { completeFollowup, getFollowups, getMedications, getReports, getTimeline } from "@/lib/api";
import { formatDate, kindMeta, relativeTime } from "@/lib/format";
import type { Followup, Medication, ReportRecord, TimelineEvent } from "@/lib/types";

export default function HealthPage() {
  const { activeMember, members, showToast } = useApp();
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [meds, setMeds] = useState<Medication[]>([]);
  const [followups, setFollowups] = useState<Followup[]>([]);
  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMember?.id]);

  async function load() {
    setLoading(true);
    try {
      const [t, m, f, r] = await Promise.all([
        getTimeline(activeMember?.id),
        getMedications(activeMember?.id),
        getFollowups("pending", activeMember?.id),
        getReports(activeMember?.id),
      ]);
      setTimeline(t);
      setMeds(m);
      setFollowups(f);
      setReports(r);
    } catch {
      // keep empty
    } finally {
      setLoading(false);
    }
  }

  async function finish(id: string) {
    try {
      await completeFollowup(id);
      showToast("已完成随访");
      await load();
    } catch {
      showToast("操作失败");
    }
  }

  const filtered = filter === "all" ? timeline : timeline.filter((item) => item.kind === filter);
  const activeMeds = meds.filter((item) => item.active);

  return (
    <AppShell title="健康档案" subtitle={`${activeMember?.name || "本人"}的健康时间线与家庭档案`}>
      <div className="mx-auto max-w-4xl space-y-4">
        <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: "用药中", value: activeMeds.length, icon: Pill, href: "/meds/" },
            { label: "待随访", value: followups.length, icon: CalendarClock, href: "/health/" },
            { label: "报告", value: reports.length, icon: FileText, href: "/reports/" },
            { label: "家庭成员", value: members.length, icon: Users, href: "/profile/" },
          ].map((item) => (
            <Link key={item.label} href={item.href} className="card p-4 transition hover:border-primary/30">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-soft text-primary"><item.icon size={16} /></span>
              <p className="mt-2 text-2xl font-semibold text-ink">{item.value}</p>
              <p className="text-[11px] text-muted">{item.label}</p>
            </Link>
          ))}
        </section>

        {followups.length ? (
          <section className="card p-5">
            <div className="flex items-center justify-between">
              <h2 className="section-title">待办随访</h2>
              <span className="text-[11px] text-muted">{followups.length} 项</span>
            </div>
            <div className="mt-3 space-y-2">
              {followups.map((item) => (
                <div key={item.id} className="flex items-center gap-3 rounded-2xl bg-canvas p-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-full bg-violet-50 text-violet-600"><CalendarClock size={15} /></span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13px] font-medium text-ink">{item.title}</p>
                    <p className="mt-0.5 text-[11px] text-muted">{item.due_date ? formatDate(item.due_date) : "待安排"} · {item.notes || "按医嘱随访"}</p>
                  </div>
                  <button onClick={() => finish(item.id)} className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-primary" aria-label="完成">
                    <CheckCircle2 size={15} />
                  </button>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        <section className="card p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="section-title">健康时间线</h2>
            <div className="flex flex-wrap gap-1.5">
              {[
                ["all", "全部"],
                ["consult", "问诊"],
                ["report", "报告"],
                ["medication", "用药"],
                ["booking", "挂号"],
                ["followup", "随访"],
              ].map(([value, label]) => (
                <button key={value} onClick={() => setFilter(value)} className={clsx("chip min-h-7 px-2.5 text-[11px]", filter === value ? "chip-active" : "")}>{label}</button>
              ))}
            </div>
          </div>

          <div className="mt-4 space-y-3">
            {loading ? <div className="h-20 animate-pulse rounded-2xl bg-canvas" /> : null}
            {!loading && filtered.length === 0 ? <p className="rounded-2xl bg-canvas p-4 text-[13px] text-muted">还没有记录。完成一次问诊或上传报告后，这里会自动生成时间线。</p> : null}
            {filtered.map((event) => {
              const meta = kindMeta[event.kind] || kindMeta.consult;
              return (
                <div key={event.id} className="relative flex gap-3 pl-1">
                  <span className={clsx("z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full", meta.color)}>
                    {event.kind === "report" ? <FileText size={14} /> : event.kind === "medication" ? <Pill size={14} /> : event.kind === "booking" ? <CalendarClock size={14} /> : event.kind === "followup" ? <HeartPulse size={14} /> : <Stethoscope size={14} />}
                  </span>
                  <div className="min-w-0 flex-1 rounded-2xl bg-canvas p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="badge bg-white text-muted">{meta.label}</span>
                      <span className="text-[11px] text-muted">{relativeTime(event.occurred_at)}</span>
                    </div>
                    <p className="mt-1.5 text-[13px] font-medium text-ink">{event.title}</p>
                    {typeof event.detail?.summary === "string" ? <p className="mt-1 line-clamp-2 text-[11px] leading-5 text-muted">{String(event.detail.summary)}</p> : null}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <h2 className="section-title">家庭成员</h2>
            <Link href="/profile/" className="text-[12px] text-primary">管理</Link>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {members.map((item) => (
              <div key={item.id} className="flex items-center gap-3 rounded-2xl bg-canvas p-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-soft text-primary"><Users size={15} /></span>
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] font-medium text-ink">{item.name} <span className="text-[11px] text-muted">{item.relation === "self" ? "本人" : item.relation}</span></p>
                  <p className="mt-0.5 truncate text-[11px] text-muted">{item.conditions || item.allergies || "暂无档案信息"}</p>
                </div>
                <ChevronRight size={15} className="text-muted" />
              </div>
            ))}
          </div>
        </section>
      </div>
    </AppShell>
  );
}
