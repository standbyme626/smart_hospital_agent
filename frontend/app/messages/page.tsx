"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { BellRing, ChevronRight } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, StateBlock } from "@/components/ui";
import {
  getNotifications,
  markAllNotificationsRead,
  markNotificationRead,
  track,
  type NotificationItem,
} from "@/lib/api";
import { relativeTime } from "@/lib/format";

const KIND_LABEL: Record<string, string> = {
  followup: "随访",
  report: "报告",
  medication: "用药",
  appointment: "就诊",
};

const FILTERS: Array<{ key: string; label: string }> = [
  { key: "all", label: "全部" },
  { key: "appointment", label: "就诊" },
  { key: "report", label: "报告" },
  { key: "medication", label: "用药" },
  { key: "followup", label: "随访" },
];

export default function MessagesPage() {
  const router = useRouter();
  const [items, setItems] = useState<NotificationItem[]>([]);
  const [unread, setUnread] = useState(0);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [filter, setFilter] = useState("all");

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      const data = await getNotifications();
      setItems(data.items);
      setUnread(data.unread);
      track("messages_view", { total: data.items.length, unread: data.unread });
    } catch {
      setItems([]);
      setUnread(0);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const shown = useMemo(
    () => (filter === "all" ? items : items.filter((item) => item.kind === filter)),
    [items, filter],
  );

  async function markAll() {
    try {
      await markAllNotificationsRead();
      setItems((prev) => prev.map((item) => ({ ...item, read_at: item.read_at || new Date().toISOString() })));
      setUnread(0);
    } catch {
      // 失败时保持未读状态，下一次进入页面会重新拉取
    }
  }

  async function open(item: NotificationItem) {
    if (!item.read_at) {
      try {
        await markNotificationRead(item.id);
        setItems((prev) => prev.map((row) => (row.id === item.id ? { ...row, read_at: new Date().toISOString() } : row)));
        setUnread((count) => Math.max(0, count - 1));
      } catch {
        // 已读失败不阻塞跳转
      }
    }
    if (item.link) router.push(item.link);
  }

  return (
    <AppShell
      title="消息"
      subtitle={unread ? unread + " 条未读 · 就诊、报告、用药与随访统一入口" : "就诊、报告、用药与随访统一入口"}
      actions={
        <div className="flex items-center gap-2">
          <span className="hidden rounded-full bg-primary-soft px-3 py-1 text-[11px] text-primary sm:inline-flex">真实数据</span>
          <button onClick={markAll} className="rounded-full bg-white px-3 py-1 text-[11px] font-medium text-primary ring-1 ring-line">
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
              className={filter === item.key ? "chip border-primary bg-primary text-white" : "chip hover:border-primary/40 hover:text-primary"}
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
                description="安排随访或解读报告后，这里会出现提醒"
              />
            </div>
          ) : (
            <ul className="divide-y divide-line">
              {shown.map((item) => {
                const isRead = Boolean(item.read_at);
                return (
                  <li key={item.id}>
                    <button onClick={() => open(item)} className="flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-primary-soft/40">
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary-soft text-primary">
                        <BellRing size={15} />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-2">
                          <span className="badge">{KIND_LABEL[item.kind] || "消息"}</span>
                          <span className={isRead ? "truncate text-[13px] text-muted" : "truncate text-[13px] font-medium text-ink"}>
                            {item.title}
                          </span>
                          {!isRead ? <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-rose-500" /> : null}
                        </span>
                        {item.body ? <span className="mt-1 block truncate text-[12px] text-muted">{item.body}</span> : null}
                      </span>
                      <span className="shrink-0 text-[11px] text-muted">{relativeTime(item.created_at)}</span>
                      <ChevronRight size={14} className="shrink-0 text-muted" />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <p className="px-1 text-[11px] leading-5 text-muted">
          消息由后端通知表生成（随访安排、报告解读等触发）。短信 / 微信渠道接入后，会在这里显示发送结果。
        </p>
      </div>
    </AppShell>
  );
}
