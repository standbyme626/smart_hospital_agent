export function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

export function formatDateTime(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${formatDate(value)} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export function relativeTime(value?: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  const diff = Date.now() - date.getTime();
  const minute = 60 * 1000;
  if (diff < minute) return "刚刚";
  if (diff < 60 * minute) return `${Math.floor(diff / minute)} 分钟前`;
  if (diff < 24 * 60 * minute) return `${Math.floor(diff / (60 * minute))} 小时前`;
  if (diff < 7 * 24 * 60 * minute) return `${Math.floor(diff / (24 * 60 * minute))} 天前`;
  return formatDate(value);
}

export const urgencyMeta: Record<string, { label: string; tone: string; hint: string }> = {
  normal: { label: "可先观察 / 普通门诊", tone: "mint", hint: "目前未见明确急危信号，建议按建议就诊。" },
  urgent: { label: "建议尽快就诊", tone: "amber", hint: "建议今天或明天内就医，若加重请随时急诊。" },
  critical: { label: "危急 · 立即急诊", tone: "coral", hint: "请立即拨打 120 或前往最近急诊。" },
};

export const kindMeta: Record<string, { label: string; color: string }> = {
  consult: { label: "问诊", color: "bg-primary-soft text-primary" },
  report: { label: "报告", color: "bg-amber-50 text-amber-700" },
  medication: { label: "用药", color: "bg-emerald-50 text-emerald-700" },
  booking: { label: "挂号", color: "bg-sky-50 text-sky-700" },
  followup: { label: "随访", color: "bg-violet-50 text-violet-700" },
};

export const relationOptions = ["self", "父母", "配偶", "子女", "其他"];

export const bodyAreas = ["头面部", "胸部", "腹部", "腰背部", "四肢", "皮肤", "其他"];

export const symptomOptions = ["疼痛", "发热", "咳嗽", "呼吸困难", "恶心呕吐", "腹泻", "头晕", "皮疹", "乏力", "其他"];

export const durationOptions = ["不到 1 天", "1–3 天", "4–7 天", "1 周以上"];

export const redFlagOptions = [
  { value: "no", label: "没有这些情况" },
  { value: "yes", label: "有：胸痛伴出汗 / 呼吸困难 / 意识不清 / 大出血 / 抽搐" },
  { value: "unknown", label: "不确定" },
];
