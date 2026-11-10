import type { HomeSummary } from "./types";

export type NoticeKind = "visit" | "report" | "medication" | "followup";

export type Notice = {
  id: string;
  kind: NoticeKind;
  title: string;
  detail: string;
  at?: string | null;
  href?: string;
};

export const NOTICE_LABEL: Record<NoticeKind, string> = {
  visit: "就诊",
  report: "报告",
  medication: "用药",
  followup: "随访",
};

/**
 * 从首页聚合数据推导消息列表。
 * 目前后端还没有独立通知表，先把已有事实（随访/用药/报告/时间线）转成消息；
 * 后续接入通知中心时，只需替换数据源，页面与筛选不变。
 */
export function buildNotices(summary: HomeSummary | null): Notice[] {
  if (!summary) return [];
  const items: Notice[] = [];

  for (const item of summary.followups || []) {
    items.push({
      id: "followup:" + item.id,
      kind: "followup",
      title: item.title || "随访待办",
      detail: item.due_date ? "计划日期 " + item.due_date : "待安排时间",
      at: item.due_date || null,
      href: "/health/",
    });
  }

  for (const item of summary.reports || []) {
    items.push({
      id: "report:" + item.id,
      kind: "report",
      title: "报告已解读",
      detail: item.title || "检查报告",
      at: item.created_at || null,
      href: "/reports/",
    });
  }

  for (const item of summary.medications || []) {
    items.push({
      id: "medication:" + item.id,
      kind: "medication",
      title: "用药提醒",
      detail: [item.name, item.dose, item.frequency].filter(Boolean).join(" · "),
      at: item.created_at || null,
      href: "/meds/",
    });
  }

  if (summary.latest_session) {
    items.push({
      id: "visit:" + summary.latest_session.id,
      kind: "visit",
      title: "上次问诊",
      detail: summary.latest_session.title || "继续上次问诊",
      at: summary.latest_session.updated_at || null,
      href: "/consult/?session=" + encodeURIComponent(summary.latest_session.id),
    });
  }

  const seen = new Set<string>();
  return items
    .filter((item) => (seen.has(item.id) ? false : (seen.add(item.id), true)))
    .sort((a, b) => (b.at || "").localeCompare(a.at || ""));
}
