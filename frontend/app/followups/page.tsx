"use client";

import { useEffect, useMemo, useState } from "react";
import { CalendarCheck2, CheckCircle2, Loader2, Plus } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { useApp } from "@/lib/app-context";
import { completeFollowup, createFollowup, getFollowups, track } from "@/lib/api";
import { formatDate } from "@/lib/format";
import type { Followup } from "@/lib/types";

type Filter = "pending" | "done" | "all";

export default function FollowupsPage() {
  const { activeMember, showToast } = useApp();
  const [items, setItems] = useState<Followup[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [filter, setFilter] = useState<Filter>("pending");
  const [title, setTitle] = useState("");
  const [dueDate, setDueDate] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setItems(await getFollowups(undefined, activeMember?.id));
    } catch {
      setItems([]);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    track("followup_view", { member_id: activeMember?.id || null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMember?.id]);

  const shown = useMemo(() => {
    if (filter === "pending") return items.filter((item) => item.status !== "done");
    if (filter === "done") return items.filter((item) => item.status === "done");
    return items;
  }, [items, filter]);

  async function add() {
    if (!title.trim()) {
      showToast("请填写随访事项");
      return;
    }
    setSaving(true);
    try {
      await createFollowup({
        title: title.trim(),
        due_date: dueDate || null,
        notes: notes.trim() || null,
        member_id: activeMember?.id,
      });
      setTitle("");
      setDueDate("");
      setNotes("");
      showToast("已添加随访");
      await load();
    } catch {
      showToast("保存失败，请稍后再试");
    } finally {
      setSaving(false);
    }
  }

  async function finish(item: Followup) {
    setBusyId(item.id);
    try {
      await completeFollowup(item.id);
      showToast("已完成随访");
      await load();
    } catch {
      showToast("操作失败，请稍后再试");
    } finally {
      setBusyId(null);
    }
  }

  const pendingCount = items.filter((item) => item.status !== "done").length;

  return (
    <AppShell
      title="随访"
      subtitle={pendingCount ? pendingCount + " 项待完成 · 复诊与自我监测计划" : "复诊与自我监测计划"}
      actions={<SourceBadge live />}
    >
      <div className="mx-auto max-w-3xl space-y-4">
        <h1 className="text-xl font-semibold text-ink lg:hidden">随访</h1>
        <OfflineBanner />

        <div className="flex flex-wrap gap-2">
          {([
            { key: "pending", label: "待完成" },
            { key: "done", label: "已完成" },
            { key: "all", label: "全部" },
          ] as Array<{ key: Filter; label: string }>).map((item) => (
            <button
              key={item.key}
              onClick={() => setFilter(item.key)}
              className={filter === item.key ? "chip border-primary bg-primary text-white" : "chip hover:border-primary/40 hover:text-primary"}
            >
              {item.label}
            </button>
          ))}
        </div>

        <section className="card p-5">
          <h2 className="section-title">添加随访</h2>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <label className="block sm:col-span-2">
              <span className="label">随访事项</span>
              <input className="input" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="例如：2 周后复诊心内科" />
            </label>
            <label className="block">
              <span className="label">计划日期</span>
              <input type="date" className="input" value={dueDate} onChange={(event) => setDueDate(event.target.value)} />
            </label>
            <label className="block">
              <span className="label">备注</span>
              <input className="input" value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="例如：带上化验单" />
            </label>
          </div>
          <button onClick={add} disabled={saving} className="btn-primary mt-4">
            {saving ? <Loader2 size={15} className="animate-spin" /> : <Plus size={15} />}
            添加
          </button>
        </section>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <h2 className="section-title">随访计划</h2>
            <span className="text-[11px] text-muted">{loading || failed ? "—" : shown.length + " 项"}</span>
          </div>
          <div className="mt-3 space-y-2">
            {loading ? <div className="h-16 animate-pulse rounded-2xl bg-canvas" /> : null}
            {!loading && failed ? <StateBlock kind="error" onRetry={load} /> : null}
            {!loading && !failed && shown.length === 0 ? (
              <StateBlock
                kind="empty"
                title={filter === "pending" ? "没有待完成的随访" : "暂无随访记录"}
                description="添加复诊或自我监测事项，到期会有提醒"
              />
            ) : null}
            {shown.map((item) => {
              const done = item.status === "done";
              return (
                <div key={item.id} className="flex items-center gap-3 rounded-2xl bg-canvas p-3">
                  <span className={done ? "flex h-9 w-9 items-center justify-center rounded-full bg-emerald-50 text-emerald-600" : "flex h-9 w-9 items-center justify-center rounded-full bg-primary-soft text-primary"}>
                    <CalendarCheck2 size={16} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className={done ? "truncate text-[14px] text-muted line-through" : "truncate text-[14px] font-medium text-ink"}>{item.title}</p>
                    <p className="mt-0.5 text-[11px] text-muted">
                      {item.due_date ? formatDate(item.due_date) : "未设置日期"}
                      {item.notes ? " · " + item.notes : ""}
                    </p>
                  </div>
                  {done ? (
                    <span className="badge bg-emerald-50 text-emerald-700">已完成</span>
                  ) : (
                    <button
                      onClick={() => finish(item)}
                      disabled={busyId === item.id}
                      className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-primary hover:bg-primary-soft disabled:opacity-40"
                      aria-label="完成随访"
                    >
                      {busyId === item.id ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle2 size={15} />}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <p className="px-1 text-[11px] leading-5 text-muted">
          随访计划由你或医生添加；到期提醒接入通知中心后会自动发送。当前为站内记录。
        </p>
      </div>
    </AppShell>
  );
}
