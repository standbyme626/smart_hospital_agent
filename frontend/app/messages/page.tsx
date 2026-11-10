"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { BellRing, ChevronRight } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, StateBlock } from "@/components/ui";
import { useApp } from "@/lib/app-context";
import { getHomeSummary, track } from "@/lib/api";
import { NOTICE_LABEL, buildNotices, type Notice, type NoticeKind } from "@/lib/notifications";
import { relativeTime } from "@/lib/format";
import type { HomeSummary } from "@/lib/types";

const FILTERS: Array<{ key: "all" | NoticeKind; label: string }> = [
  { key: "all", label: "全部" },
  { key: "visit", label: "就诊" },
  { key: "report", label: "报告" },
  { key: "medication", label: "用药" },
  { key: "followup", label: "随访" },
];

const READ_KEY = "sha_notice_read";

function loadRead(): Set<string> {
  try {
    const raw = window.localStorage.getItem(READ_KEY);
    const list = raw ? (JSON.parse(raw) as string[]) : [];
    return new Set(Array.isArray(list) ? list : []);
  } catch {
    return new Set();
  }
}

function saveRead(ids: Set<string>) {
  try {
    window.localStorage.setItem(READ_KEY, JSON.stringify(Array.from(ids).slice(-200)));
  } catch {
    // localStorage 不可用时忽略，已读状态仅影响角标
  }
}

export default function MessagesPage() {
  const { activeMember } = useApp();
  const [summary, setSummary] = useState<HomeSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [filter, setFilter] = useState<"all" | NoticeKind>("all");
  const [read, setRead] = useState<Set<string>>(new Set());

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      const data = await getHomeSummary(activeMember?.id);
      setSummary(data);
      track("messages_view", { member_id: activeMember?.id || null });
    } catch {
      setSummary(null);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setRead(loadRead());
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMember?.id]);

  const notices = useMemo(() => buildNotices(summary), [summary]);
  const shown = filter === "all" ? notices : notices.filter((item) => item.kind === filter);
  const unread = notices.filter((item) => !read.has(item.id)).length;

  function markAllRead() {
    const next = new Set(read);
    notices.forEach((item) => next.add(item.id));
    setRead(next);
    saveRead(next);
  }

  function openNotice(item: Notice) {
    const next = new Set(read);
    next.add(item.id);
    setRead(next);
    saveRead(next);
  }

  return (
    <AppShell
      title="消息"
      subtitle={unread ? unread + " 条未读 · 就诊、报告、用药与随访统一入口" : "就诊、报告、用药与随访统一入口"}
      actions={
        <div className="flex items-center gap-2">
          <span className="hidden rounded-full bg-primary-soft px-3 py-1 text-[11px] text-primary sm:inline-flex">真实数据</span>
          <button onClick={markAllRead} className="rounded-full bg-white px-3 py-1 text-[11px] font-medium text-primary ring-1 ring-line">
            全部已读
          </button>
        </div>
      }
    >
      <div className="space-y-4">
        <OfflineBanner />

        <div className="flex flex-wrap gap-2">
          {FILTERS.map((item) => (
            <button
              key={item.key}
              onClick={() => setFilter(item.key)}
              className={
                filter === item.key
                  ? "chip border-primary bg-primary text-white"
                  : "chip hover:border-primary/40 hover:text-primary"
              }
            >
              {item.label}
            </button>
          ))}
        </div>

        <section className="card overflow-hidden">
          {loading ? (
            <div className="space-y-2 p-4">
              <div className="h-14 animate-pulse rounded-2xl bg-canvas" />
              <div className="h-14 animate-pulse rounded-2xl bg-canvas" />
              <div className="h-14 animate-pulse rounded-2xl bg-canvas" />
            </div>
          ) : failed ? (
            <div className="p-3">
              <StateBlock kind="error" onRetry={load} />
            </div>
          ) : shown.length === 0 ? (
            <div className="p-3">
              <StateBlock
                kind="empty"
                title={filter === "all" ? "还没有消息" : "该分类下暂无消息"}
                description="完成问诊、上传报告或安排随访后，这里会出现提醒"
              />
            </div>
          ) : (
            <ul className="divide-y divide-line">
              {shown.map((item) => {
                const isRead = read.has(item.id);
                return (
                  <li key={item.id}>
                    <Link
                      href={item.href || "/messages/"}
                      onClick={() => openNotice(item)}
                      className="flex items-center gap-3 px-4 py-3 transition hover:bg-primary-soft/40"
                    >
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary-soft text-primary">
                        <BellRing size={15} />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-2">
                          <span className="badge">{NOTICE_LABEL[item.kind]}</span>
                          <span className={isRead ? "truncate text-[13px] text-muted" : "truncate text-[13px] font-medium text-ink"}>
                            {item.title}
                          </span>
                          {!isRead ? <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-rose-500" /> : null}
                        </span>
                        <span className="mt-1 block truncate text-[12px] text-muted">{item.detail}</span>
                      </span>
                      <span className="shrink-0 text-[11px] text-muted">{relativeTime(item.at)}</span>
                      <ChevronRight size={14} className="shrink-0 text-muted" />
                    </Link>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <p className="px-1 text-[11px] leading-5 text-muted">
          消息由现有就诊数据推导，已读状态保存在本机。接入通知中心后，将支持短信 / 微信 / 站内信多渠道与发送记录。
        </p>
      </div>
    </AppShell>
  );
}
