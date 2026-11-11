/** 管理端配置注册表客户端（规则库 / 通知模板 / Prompt / Agent 配置）。
 *
 * 单独成文件而不是塞进 lib/api.ts：Wave E 的四屏只依赖这一组接口，
 * 独立文件可以让管理端迭代不与其他端的 api.ts 改动冲突。
 */
import { API_BASE, ApiError, getToken } from "./api";

export type ConfigKind = "rule" | "notify_template" | "prompt" | "agent_config";

export type ConfigKindInfo = {
  kind: ConfigKind;
  name: string;
  description: string;
  total: number;
  published: number;
  disabled: number;
};

export type ConfigEntry = {
  kind: ConfigKind;
  key: string;
  draft: Record<string, unknown>;
  published: Record<string, unknown> | null;
  version: number;
  enabled: boolean;
  effective: Record<string, unknown> | null;
  dirty: boolean;
  updated_by: string | null;
  updated_at: string | null;
  published_by: string | null;
  published_at: string | null;
};

export type ConfigHistoryItem = {
  version: number;
  value: Record<string, unknown>;
  action: "publish" | "rollback" | string;
  actor: string | null;
  note: string | null;
  created_at: string | null;
};

export type ConfigChangeItem = {
  kind: ConfigKind;
  kind_name: string;
  key: string;
  version: number;
  action: string;
  actor: string | null;
  note: string | null;
  created_at: string | null;
};

export type NotifyPreview = {
  template_key: string;
  title: string;
  body: string;
  channels: string[];
  retry: Record<string, unknown>;
  missing_vars: string[];
  source: "published" | "draft";
};

const CONFIG_ROOT = "/api/v1/admin/config";

function entryPath(kind: ConfigKind, key: string): string {
  return CONFIG_ROOT + "/" + encodeURIComponent(kind) + "/" + encodeURIComponent(key);
}

async function confFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", "Bearer " + token);
  if (init.body) headers.set("Content-Type", "application/json");

  let response: Response;
  try {
    response = await fetch(API_BASE + path, { ...init, headers });
  } catch {
    throw new ApiError(0, "网络连接失败，请检查网络后重试");
  }

  if (!response.ok) {
    let message = "";
    try {
      const data = (await response.json()) as { detail?: unknown };
      const detail = data?.detail;
      message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : "";
    } catch {
      message = "";
    }
    throw new ApiError(response.status, message || "请求失败（" + response.status + "）");
  }
  return (await response.json()) as T;
}

export async function getConfigKinds(): Promise<ConfigKindInfo[]> {
  const data = await confFetch<{ items: ConfigKindInfo[] }>(CONFIG_ROOT + "/kinds");
  return data.items;
}

export async function listConfigEntries(kind: ConfigKind): Promise<ConfigEntry[]> {
  const data = await confFetch<{ items: ConfigEntry[] }>(CONFIG_ROOT + "/" + encodeURIComponent(kind));
  return data.items;
}

export async function saveConfigDraft(
  kind: ConfigKind,
  key: string,
  value: Record<string, unknown>,
  note?: string,
): Promise<ConfigEntry> {
  const data = await confFetch<{ entry: ConfigEntry }>(entryPath(kind, key), {
    method: "PUT",
    body: JSON.stringify({ value, note }),
  });
  return data.entry;
}

export async function publishConfig(kind: ConfigKind, key: string, note?: string): Promise<ConfigEntry> {
  const data = await confFetch<{ entry: ConfigEntry }>(entryPath(kind, key) + "/publish", {
    method: "POST",
    body: JSON.stringify({ note }),
  });
  return data.entry;
}

export async function rollbackConfig(
  kind: ConfigKind,
  key: string,
  version: number,
): Promise<ConfigEntry> {
  const data = await confFetch<{ entry: ConfigEntry }>(entryPath(kind, key) + "/rollback", {
    method: "POST",
    body: JSON.stringify({ version }),
  });
  return data.entry;
}

export async function setConfigEnabled(
  kind: ConfigKind,
  key: string,
  enabled: boolean,
): Promise<ConfigEntry> {
  const data = await confFetch<{ entry: ConfigEntry }>(entryPath(kind, key) + "/enabled", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
  return data.entry;
}

export async function getConfigHistory(kind: ConfigKind, key: string): Promise<ConfigHistoryItem[]> {
  const data = await confFetch<{ items: ConfigHistoryItem[] }>(entryPath(kind, key) + "/history");
  return data.items;
}

export async function getConfigChanges(limit = 20): Promise<ConfigChangeItem[]> {
  const data = await confFetch<{ items: ConfigChangeItem[] }>(CONFIG_ROOT + "/changes?limit=" + limit);
  return data.items;
}

export async function previewNotifyTemplate(
  key: string,
  vars: Record<string, string>,
): Promise<NotifyPreview> {
  return confFetch<NotifyPreview>(CONFIG_ROOT + "/notify_template/" + encodeURIComponent(key) + "/preview", {
    method: "POST",
    body: JSON.stringify({ vars }),
  });
}
