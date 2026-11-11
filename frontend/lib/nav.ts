export type EndKey = "patient" | "clinician" | "ops" | "platform";

export type NavItem = { href: string; label: string; badge?: number };
export type NavGroup = { title: string; items: NavItem[] };

export const END_LABEL: Record<EndKey, string> = {
  patient: "患者端",
  clinician: "医生端",
  ops: "运营 / 质控",
  platform: "平台管理",
};

export const NAV: Record<EndKey, NavGroup[]> = {
  patient: [
    {
      title: "常用",
      items: [
        { href: "/", label: "首页" },
        { href: "/messages/", label: "消息" },
        { href: "/followups/", label: "随访" },
        { href: "/consult/", label: "AI 诊室" },
        { href: "/reports/", label: "报告解读" },
        { href: "/meds/", label: "药管家" },
      ],
    },
    {
      title: "就医",
      items: [
        { href: "/services/", label: "挂号服务" },
        { href: "/previsit/", label: "就诊准备" },
        { href: "/queue/", label: "排队叫号" },
        { href: "/pickup/", label: "取药进度" },
        { href: "/insurance/", label: "医保结算" },
        { href: "/health/", label: "健康档案" },
      ],
    },
    {
      title: "我的",
      items: [
        { href: "/profile/", label: "设置与家庭" },
        { href: "/privacy/", label: "隐私与安全" },
      ],
    },
  ],
  clinician: [
    {
      title: "诊疗",
      items: [
        { href: "/clinician/", label: "诊室工作台" },
        { href: "/clinician/print/", label: "文书打印" },
        { href: "/clinician/followup/", label: "诊毕随访" },
        { href: "/clinician/review/", label: "药师审方" },
      ],
    },
    {
      title: "协同",
      items: [
        { href: "/clinician/schedule/", label: "我的排班" },
        { href: "/clinician/ward/", label: "移动查房" },
        { href: "/clinician/todo/", label: "待办" },
      ],
    },
  ],
  ops: [
    {
      title: "监控",
      items: [
        { href: "/ops/", label: "概览" },
        { href: "/ops/reviews/", label: "支付审核" },
        { href: "/ops/audit/", label: "审计复盘" },
        { href: "/ops/analytics/", label: "运营分析" },
      ],
    },
    {
      title: "治理",
      items: [
        { href: "/ops/insurance/", label: "医保对账" },
        { href: "/ops/crisis/", label: "危机事件" },
        { href: "/ops/notifications/", label: "通知记录" },
      ],
    },
  ],
  platform: [
    {
      title: "概览",
      items: [{ href: "/platform/", label: "运行总览" }],
    },
    {
      title: "接入",
      items: [
        { href: "/platform/integrations/", label: "接入与实验" },
        { href: "/platform/insight/", label: "可观测" },
        { href: "/platform/spec/", label: "规范与数据流" },
      ],
    },
    {
      title: "建设",
      items: [
        { href: "/platform/config/", label: "配置管理" },
        { href: "/platform/knowledge/", label: "知识库" },
      ],
    },
  ],
};

export function isActive(pathname: string, href: string): boolean {
  const base = href.replace(/\/$/, "");
  if (base === "" || base === "/") return pathname === "/" || pathname === "";
  if (base === "/platform") return pathname === "/platform" || pathname === "/platform/";
  return pathname === base || pathname.startsWith(base + "/");
}
