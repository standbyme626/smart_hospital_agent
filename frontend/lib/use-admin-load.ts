"use client";

import { useCallback, useState } from "react";

export type LoadState = "loading" | "ready" | "denied" | "error" | "timeout";

export function statusOf(error: unknown): LoadState {
  const status = (error as { status?: number } | null)?.status ?? 0;
  if (status === 401 || status === 403) return "denied";
  if (status === 408 || status === 504) return "timeout";
  return "error";
}

/** 管理端页面的统一取数：六态（加载 / 成功 / 空 / 错误 / 无权限 / 超时）+ 操作提示。
 * 注意：loader 必须用 useCallback 包裹，否则会重复请求。
 */
export function useAdminLoad<T>(loader: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setState("loading");
    try {
      setData(await loader());
      setState("ready");
    } catch (error) {
      setState(statusOf(error));
    }
  }, [loader]);

  const run = useCallback(async (action: () => Promise<void>) => {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (error) {
      const next = statusOf(error);
      if (next === "denied" || next === "timeout") setState(next);
      setMessage((error as Error).message || "操作失败");
    } finally {
      setBusy(false);
    }
  }, []);

  return { data, state, message, setMessage, busy, run, load };
}
