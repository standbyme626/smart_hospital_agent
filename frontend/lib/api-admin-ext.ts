/** 管理端 Wave E 第二批客户端：知识库 / 检索 / 未命中 / 模型 / 回放 / 审计 / 验收 / HIS / 多院区 / 实验。 */
import { API_BASE, ApiError, getToken } from "./api";

export type AdminCheck = { name: string; ok: boolean; status: "ok" | "warn" | "fail"; detail: string };

export type PlaybookItem = {
  pid: string;
  name: string;
  area: string;
  status: "ready" | "partial" | "blocked";
  endpoint: string;
  evidence: string;
  note: string;
};

export type KbOverview = {
  documents: number;
  departments: Array<{ department: string; count: number }>;
  coverage: { covered: number; total: number; missing: string[] };
  retrieval_config: Record<string, unknown>;
  unmatched: number;
  resolved: number;
};

export type RagDocument = {
  id: number;
  doc_key: string;
  title: string;
  department: string | null;
  source: string | null;
  chars: number;
  created_at: string | null;
};

export type RetrievalHit = {
  id: number;
  doc_key: string;
  title: string;
  snippet: string;
  department: string | null;
  source: string | null;
  score: number;
};

export type RetrievalDebug = {
  query: string;
  department: string | null;
  latency_ms: number;
  total: number;
  filtered: boolean;
  dept_unavailable: boolean;
  settings: { top_k: number; rerank_top_n: number };
  hits: RetrievalHit[];
};

export type PipelineStage = { name: string; ok: boolean; detail: string };
export type PipelineCheck = {
  query: string;
  latency_ms: number;
  hits: number;
  error: string | null;
  stages: PipelineStage[];
  top: RetrievalHit[];
};

export type UnmatchedItem = {
  question_key: string;
  question: string;
  department: string | null;
  hits: number;
  first_seen: string | null;
  last_seen: string | null;
  resolved: boolean;
  resolved_doc_key: string | null;
  resolved_by: string | null;
  resolved_at: string | null;
};

export type ModelSummary = {
  days: number;
  totals: { calls: number; prompt_tokens: number; completion_tokens: number; cost: number };
  items: Array<{
    provider: string;
    model: string;
    task: string;
    calls: number;
    prompt_tokens: number;
    completion_tokens: number;
    cost: number;
  }>;
};

export type SessionItem = {
  id: string;
  user_tag: string;
  title: string;
  messages: number;
  created_at: string | null;
  last_at: string | null;
};

export type SessionTrace = {
  session: { id: string; user_tag: string; title: string; created_at: string | null; updated_at: string | null };
  messages: Array<{ role: string; content: string; created_at: string | null }>;
  events: Array<{ kind: string; data: Record<string, unknown>; created_at: string | null }>;
  audit: Array<{ actor: string | null; action: string; detail: Record<string, unknown>; created_at: string | null }>;
};

export type AuditAggregate = {
  days: number;
  total: number;
  by_action: Array<{ action: string; count: number }>;
  by_actor: Array<{ actor: string; count: number }>;
  recent: Array<{ request_id: string | null; actor: string | null; action: string; created_at: string | null }>;
  config_changes: Array<{
    kind: string;
    key: string;
    version: number;
    action: string;
    actor: string | null;
    note: string | null;
    created_at: string | null;
  }>;
};

export type AcceptanceData = {
  checks: AdminCheck[];
  playbooks: PlaybookItem[];
  playbook_summary: Record<string, number>;
  summary: { ok: number; warn: number; fail: number; total: number };
};

export type HisOverview = {
  config: { mcp_url: string; payment_mode: string; clinician_write_enabled: boolean; auth_enabled: boolean };
  capabilities: Array<{ name: string; owner: string; note: string }>;
  checks: Array<{ ok: boolean; detail: string; latency_ms: number | null; actor: string | null; created_at: string | null }>;
};

export type HisCheckResult = { ok: boolean; detail: string; latency_ms: number };

export type TenantItem = {
  id: string;
  code: string;
  name: string;
  status: string;
  isolation: string;
  note: string | null;
  created_at: string | null;
};

export type ExperimentItem = {
  id: string;
  key: string;
  name: string;
  status: string;
  rollout: number;
  variants: string[];
  note: string | null;
  started_at: string | null;
  created_at: string | null;
  assignments: Record<string, number>;
};

export type FlowData = { stages: Array<{ name: string; count: number }> };
export type StatesData = {
  items: Array<{
    label: string;
    table: string;
    column: string;
    values: Array<{ value: string; count: number }>;
    error?: string;
  }>;
};

const ADMIN_ROOT = "/api/v1/admin";

async function adminFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
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

export const getKbOverview = () => adminFetch<KbOverview>(ADMIN_ROOT + "/kb");

export const listRagDocuments = (department?: string) =>
  adminFetch<{ items: RagDocument[]; total: number }>(
    ADMIN_ROOT + "/rag/documents?limit=100" + (department ? "&department=" + encodeURIComponent(department) : ""),
  );

export const debugRetrieval = (query: string, department?: string) =>
  adminFetch<RetrievalDebug>(ADMIN_ROOT + "/retrieval/debug", {
    method: "POST",
    body: JSON.stringify({ query, department: department || null, top_k: 5 }),
  });

export const runPipelineCheck = (query: string, department?: string) =>
  adminFetch<PipelineCheck>(ADMIN_ROOT + "/rag/pipeline-check", {
    method: "POST",
    body: JSON.stringify({ query, department: department || null }),
  });

export const listUnmatched = (onlyOpen = true) =>
  adminFetch<{ items: UnmatchedItem[]; open_total: number }>(
    ADMIN_ROOT + "/unmatched?limit=50&only_open=" + (onlyOpen ? "true" : "false"),
  );

export const resolveUnmatched = (payload: {
  question: string;
  answer: string;
  department?: string | null;
  question_key?: string;
}) =>
  adminFetch<{ doc_key: string; written: number; question_key: string }>(
    ADMIN_ROOT + "/unmatched/resolve",
    { method: "POST", body: JSON.stringify(payload) },
  );

export const getModelSummary = (days = 7) =>
  adminFetch<ModelSummary>(ADMIN_ROOT + "/model-calls/summary?days=" + days);

export const listSessions = () => adminFetch<{ items: SessionItem[] }>(ADMIN_ROOT + "/sessions?limit=30");

export const getSessionTrace = (sessionId: string) =>
  adminFetch<SessionTrace>(ADMIN_ROOT + "/sessions/" + encodeURIComponent(sessionId) + "/trace");

export const getAuditAggregate = (days = 1) =>
  adminFetch<AuditAggregate>(ADMIN_ROOT + "/audit/aggregate?days=" + days);

export const getAcceptance = () => adminFetch<AcceptanceData>(ADMIN_ROOT + "/acceptance");

export const getHisOverview = () => adminFetch<HisOverview>(ADMIN_ROOT + "/his");

export const runHisCheck = () =>
  adminFetch<HisCheckResult>(ADMIN_ROOT + "/his/check", { method: "POST" });

export const listTenants = () =>
  adminFetch<{ items: TenantItem[]; statuses: string[]; isolation_modes: string[] }>(ADMIN_ROOT + "/tenants");

export const createTenant = (payload: { code: string; name: string; isolation: string; note?: string }) =>
  adminFetch<{ tenant: TenantItem }>(ADMIN_ROOT + "/tenants", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const updateTenant = (tenantId: string, payload: Record<string, unknown>) =>
  adminFetch<{ tenant: TenantItem }>(ADMIN_ROOT + "/tenants/" + encodeURIComponent(tenantId), {
    method: "PATCH",
    body: JSON.stringify(payload),
  });

export const listExperiments = () =>
  adminFetch<{ items: ExperimentItem[]; statuses: string[] }>(ADMIN_ROOT + "/experiments");

export const createExperiment = (payload: { key: string; name: string; variants: string[]; note?: string }) =>
  adminFetch<{ experiment: ExperimentItem }>(ADMIN_ROOT + "/experiments", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const updateExperiment = (key: string, payload: Record<string, unknown>) =>
  adminFetch<{ experiment: ExperimentItem }>(ADMIN_ROOT + "/experiments/" + encodeURIComponent(key), {
    method: "PATCH",
    body: JSON.stringify(payload),
  });

export const getFlow = () => adminFetch<FlowData>(ADMIN_ROOT + "/flow");
export const getStates = () => adminFetch<StatesData>(ADMIN_ROOT + "/states");

export const getConfigChanges = () =>
  adminFetch<{ items: Array<Record<string, unknown>> }>(ADMIN_ROOT + "/config/changes?limit=20");
