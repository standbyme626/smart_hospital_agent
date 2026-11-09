const memberContextPrefixes = [
  "就诊人：",
  "性别：",
  "年龄：",
  "出生日期：",
  "档案过敏史：",
  "档案既往史：",
];

const titleContextPrefixes = [...memberContextPrefixes, "过敏史：", "既往史："];

const titleLabelPrefixes = [
  "主要部位：",
  "症状：",
  "持续时间：",
  "严重程度：",
  "红旗筛查：",
  "补充描述：",
  "正在用药：",
  "过敏史：",
  "既往史：",
];

export function ageFromBirthDate(birthDate?: string | null): number | null {
  if (!birthDate) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(birthDate.trim());
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const birth = new Date(year, month - 1, day);
  if (birth.getFullYear() !== year || birth.getMonth() !== month - 1 || birth.getDate() !== day) return null;
  const now = new Date();
  if (birth.getTime() > now.getTime()) return null;
  let age = now.getFullYear() - year;
  const hadBirthday = now.getMonth() > month - 1 || (now.getMonth() === month - 1 && now.getDate() >= day);
  if (!hadBirthday) age -= 1;
  if (age < 0 || age > 120) return null;
  return age;
}

export function stripMemberContext(text?: string | null): string {
  if (!text) return "";
  const marker = "用户输入：";
  const index = text.lastIndexOf(marker);
  const body = index >= 0 ? text.slice(index + marker.length) : text;
  return body
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !memberContextPrefixes.some((prefix) => line.startsWith(prefix)))
    .join("\n");
}

export function formatSessionTitle(title?: string | null, fallback = "上次问诊"): string {
  const stripped = stripMemberContext(title);
  if (!stripped) return fallback;
  const parts = stripped
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !titleContextPrefixes.some((prefix) => line.startsWith(prefix)))
    .map((line) => {
      let value = line;
      const label = titleLabelPrefixes.find((prefix) => value.startsWith(prefix));
      if (label) value = value.slice(label.length).trim();
      return value.replace(/\s+/g, " ").trim();
    })
    .filter(Boolean);
  let joined = parts.join(" · ").replace(/\s+/g, " ").trim();
  if (!joined) return fallback;
  if (joined.length > 60) joined = `${joined.slice(0, 60).trimEnd()}…`;
  return joined;
}

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
