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

/** 按自然日分组：长列表分组后更好扫读，也不会像一条无限长的带子 */
function dayGroup(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "更早";
  const today = new Date();
  const startOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const diffDays = Math.round((startOf(today) - startOf(date)) / 86_400_000);
  if (diffDays <= 0) return "今天";
  if (diffDays === 1) return "昨天";
  if (diffDays < 7) return "本周";
  return "更早";
}

export default function MessagesPage() {
  const router = useRouter();
  const [items, setItems] = useState<NotificationItem[]>([]);
  const [unread, setUnread] = useState(0);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [filter, setFilter] = useState("all");
  const [onlyUnread, setOnlyUnread] = useState(false);

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
    () =>
      items
        .filter((item) => (filter === "all" ? true : item.kind === filter))
        .filter((item) => (onlyUnread ? !item.read_at : true)),
    [items, filter, onlyUnread],
  );

  // 分组保持原始顺序（后端按时间倒序返回）
  const groups = useMemo(() => {
    const order = ["今天", "昨天", "本周", "更早"];
    const buckets = new Map<string, NotificationItem[]>();
    for (const item of shown) {
      const key = dayGroup(item.created_at);
      const bucket = buckets.get(key);
      if (bucket) bucket.push(item);
      else buckets.set(key, [item]);
    }
    return order.filter((key) => buckets.has(key)).map((key) => ({ key, rows: buckets.get(key) as NotificationItem[] }));
  }, [shown]);

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

  const hasAnyUnread = items.some((item) => !item.read_at);

  return (
    <AppShell
      title="消息"
      subtitle={unread ? unread + " 条未读 · 就诊、报告、用药与随访统一入口" : "就诊、报告、用药与随访统一入口"}
      actions={
        <div className="flex items-center gap-2">
          <span className="hidden rounded-full bg-primary-soft px-3 py-1 text-[11px] text-primary sm:inline-flex">真实数据</span>
          <button
            onClick={markAll}
            disabled={!hasAnyUnread}
            className="rounded-full bg-white px-3 py-1 text-[11px] font-medium text-primary ring-1 ring-line disabled:text-muted disabled:ring-line/60"
          >
            全部已读
          </button>
        </div>
      }
    >
      <div className="space-y-4">
        <OfflineBanner />

        {/* 列表宽度收在 48rem：13px 正文一行约 60 个汉字，不会拉成"长条" */}
        <div className="max-w-3xl space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            {FILTERS.map((item) => (
              <button
                key={item.key}
                onClick={() => setFilter(item.key)}
                className={filter === item.key ? "chip border-primary bg-primary text-white" : "chip hover:border-primary/40 hover:text-primary"}
              >
                {item.label}
              </button>
            ))}
            <button
              onClick={() => setOnlyUnread((prev) => !prev)}
              aria-pressed={onlyUnread}
              className={onlyUnread ? "chip border-primary bg-primary-soft text-primary" : "chip text-muted hover:text-primary"}
            >
              只看未读{unread ? " · " + unread : ""}
            </button>
            {shown.length ? (
              <span className="ml-auto text-[11px] text-muted">共 {shown.length} 条</span>
            ) : null}
          </div>

          <section className="card overflow-hidden">
            {loading ? (
              <div className="space-y-2 p-4">
                {[0, 1, 2].map((index) => (
                  <div key={index} className="h-20 animate-pulse rounded-2xl bg-canvas" />
                ))}
              </div>
            ) : failed ? (
              <div className="p-3">
                <StateBlock kind="error" onRetry={load} />
              </div>
            ) : shown.length === 0 ? (
              <div className="p-3">
                <StateBlock
                  kind="empty"
                  title={onlyUnread ? "没有未读消息" : filter === "all" ? "还没有消息" : "该分类下暂无消息"}
                  description={onlyUnread ? "消息都已读，点这里切回全部" : "安排随访或解读报告后，这里会出现提醒"}
                />
              </div>
            ) : (
              groups.map((group) => (
                <div key={group.key}>
                  <div className="flex items-center gap-2 bg-canvas/60 px-4 py-1.5">
                    <span className="text-[11px] font-medium text-muted">{group.key}</span>
                    <span className="text-[11px] text-muted/70">{group.rows.length} 条</span>
                  </div>
                  <ul className="divide-y divide-line">
                    {group.rows.map((item) => {
                      const isRead = Boolean(item.read_at);
                      return (
                        <li key={item.id}>
                          <button
                            onClick={() => open(item)}
                            className="group flex w-full items-start gap-3 px-4 py-3 text-left transition hover:bg-primary-soft/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                          >
                            <span
                              className={
                                isRead
                                  ? "mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-canvas text-muted"
                                  : "mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary-soft text-primary"
                              }
                            >
                              <BellRing size={14} />
                            </span>

                            <span className="min-w-0 flex-1">
                              <span className="flex items-center gap-2">
                                <span className="badge shrink-0">{KIND_LABEL[item.kind] || "消息"}</span>
                                <span className="shrink-0 text-[11px] text-muted">{relativeTime(item.created_at)}</span>
                                {!isRead ? <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-rose-500" aria-label="未读" /> : null}
                              </span>
                              <span
                                className={
                                  isRead
                                    ? "mt-0.5 block text-[13.5px] leading-6 text-muted line-clamp-2 text-pretty"
                                    : "mt-0.5 block text-[13.5px] font-medium leading-6 text-ink line-clamp-2 text-pretty"
                                }
                              >
                                {item.title}
                              </span>
                              {item.body ? (
                                <span className="mt-0.5 block text-[12px] leading-5 text-muted line-clamp-2 text-pretty">
                                  {item.body}
                                </span>
                              ) : null}
                            </span>

                            <ChevronRight size={14} className="mt-4 shrink-0 text-muted/60 transition group-hover:text-primary" />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))
            )}
          </section>

          <p className="px-1 text-[11px] leading-5 text-muted">
            消息由后端通知表生成（随访安排、报告解读等触发）。短信 / 微信渠道接入后，会在这里显示发送结果。
          </p>
        </div>
      </div>
    </AppShell>
  );
}
