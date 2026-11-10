"use client";

import { useEffect, useState } from "react";
import { RefreshCw, WifiOff } from "lucide-react";

/** 数据来源标识：真实接口 / 演示数据 */
export function SourceBadge({ live = true }: { live?: boolean }) {
  return (
    <span
      className={
        live
          ? "inline-flex items-center gap-1 rounded-full bg-primary-soft px-2.5 py-0.5 text-[11px] font-medium text-primary"
          : "inline-flex items-center gap-1 rounded-full bg-amber-50 px-2.5 py-0.5 text-[11px] font-medium text-amber-700"
      }
    >
      <span className={live ? "h-1.5 w-1.5 rounded-full bg-primary" : "h-1.5 w-1.5 rounded-full bg-amber-500"} />
      {live ? "真实数据" : "演示数据"}
    </span>
  );
}

type StateKind = "empty" | "error" | "timeout" | "offline" | "denied";

const STATE_TEXT: Record<StateKind, { title: string; hint: string }> = {
  empty: { title: "暂无内容", hint: "产生记录后会显示在这里" },
  error: { title: "加载失败", hint: "请稍后重试，已保留你的输入" },
  timeout: { title: "同步超时", hint: "正显示最近一次同步结果，可重试" },
  offline: { title: "网络已断开", hint: "已显示缓存内容，恢复后自动重连" },
  denied: { title: "无权限", hint: "该内容不在你的授权范围内" },
};

/** 统一状态块：空 / 错误 / 超时 / 断网 / 无权限 */
export function StateBlock({
  kind,
  title,
  description,
  onRetry,
}: {
  kind: StateKind;
  title?: string;
  description?: string;
  onRetry?: () => void;
}) {
  const meta = STATE_TEXT[kind];
  return (
    <div className="flex flex-col items-center gap-2 rounded-2xl bg-canvas p-6 text-center">
      <p className="text-[13px] font-medium text-ink">{title || meta.title}</p>
      <p className="text-[12px] text-muted">{description || meta.hint}</p>
      {onRetry && (kind === "error" || kind === "timeout" || kind === "offline") ? (
        <button onClick={onRetry} className="mt-1 inline-flex min-h-9 items-center gap-1.5 rounded-full bg-white px-4 text-[12px] font-medium text-primary">
          <RefreshCw size={13} /> 重试
        </button>
      ) : null}
    </div>
  );
}

/** 全局断网提示条 */
export function OfflineBanner() {
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    if (typeof navigator !== "undefined") setOffline(!navigator.onLine);
    const on = () => setOffline(false);
    const off = () => setOffline(true);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", off);
    };
  }, []);
  if (!offline) return null;
  return (
    <div className="flex items-center gap-2 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-[12px] text-amber-800">
      <WifiOff size={14} /> 网络已断开 · 已显示缓存内容，恢复后会自动重试
    </div>
  );
}
