import type {
  AdminUser,
  AuditEntry,
  BookingOrder,
  Doctor,
  DocumentDetail,
  EvalResult,
  Followup,
  HomeSummary,
  Medication,
  Member,
  ModelCall,
  RagStatus,
  ReportRecord,
  ReviewRequest,
  ScheduleSlot,
  Session,
  StreamEvent,
  TimelineEvent,
  User,
  VisitGuide,
} from "./types";

const rawBase = process.env.NEXT_PUBLIC_API_BASE;
export const API_BASE = rawBase === undefined ? "http://127.0.0.1:8020" : rawBase.replace(/\/$/, "");

const TOKEN_KEY = "sha_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  window.localStorage.removeItem(TOKEN_KEY);
}

export class UnauthorizedError extends Error {
  constructor() {
    super("unauthorized");
  }
}

function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = authHeaders(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let lastError: unknown = null;
  const method = (init.method || "GET").toUpperCase();
  const canRetry = method === "GET" || path === "/api/v1/auth/login";
  const attempts = canRetry ? 2 : 1;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
      if (response.status === 401) {
        clearToken();
        throw new UnauthorizedError();
      }
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      return (await response.json()) as T;
    } catch (err) {
      if (err instanceof UnauthorizedError) throw err;
      lastError = err;
      if (canRetry && attempt === 0) {
        await new Promise((resolve) => setTimeout(resolve, 400));
        continue;
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error("network error");
}

export async function login(username: string, password: string): Promise<{ access_token: string; role: string }> {
  const result = await apiFetch<{ access_token: string; role: string }>("/api/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setToken(result.access_token);
  return result;
}

export async function getMe(): Promise<User> {
  return apiFetch("/api/v1/auth/me");
}

export async function getFamily(): Promise<Member[]> {
  const data = await apiFetch<{ members: Member[] }>("/api/v1/family");
  return data.members;
}

export async function createMember(payload: Partial<Member>): Promise<Member> {
  const data = await apiFetch<{ member: Member }>("/api/v1/family", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data.member;
}

export async function updateMember(id: string, payload: Partial<Member>): Promise<Member> {
  const data = await apiFetch<{ member: Member }>(`/api/v1/family/${id}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  return data.member;
}

export async function deleteMember(id: string): Promise<void> {
  await apiFetch(`/api/v1/family/${id}`, { method: "DELETE" });
}

export async function getHomeSummary(memberId?: string): Promise<HomeSummary> {
  const query = memberId ? `?member_id=${encodeURIComponent(memberId)}` : "";
  return apiFetch(`/api/v1/home/summary${query}`);
}

export async function getSessions(): Promise<Session[]> {
  const data = await apiFetch<{ sessions: Session[] }>("/api/v1/sessions");
  return data.sessions;
}

export async function getMedications(memberId?: string): Promise<Medication[]> {
  const query = memberId ? `?member_id=${encodeURIComponent(memberId)}` : "";
  const data = await apiFetch<{ medications: Medication[] }>(`/api/v1/medications${query}`);
  return data.medications;
}

export async function createMedication(payload: {
  name: string;
  member_id?: string;
  dose?: string;
  frequency?: string;
  notes?: string;
}): Promise<Medication> {
  const data = await apiFetch<{ medication: Medication }>("/api/v1/medications", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data.medication;
}

export async function toggleMedication(id: string, active: boolean): Promise<Medication> {
  const data = await apiFetch<{ medication: Medication }>(`/api/v1/medications/${id}?active=${active}`, {
    method: "PATCH",
  });
  return data.medication;
}

export async function deleteMedication(id: string): Promise<void> {
  await apiFetch(`/api/v1/medications/${id}`, { method: "DELETE" });
}

export async function checkMedications(text: string, memberId?: string): Promise<{
  drugs: string[];
  interactions: { drug_a: string; drug_b: string; severity: string; description: string }[];
  pregnancy_contraindications: { drug: string; description: string }[];
  highest_severity: string;
}> {
  return apiFetch("/api/v1/medications/check", {
    method: "POST",
    body: JSON.stringify({ text, member_id: memberId }),
  });
}

export async function interpretReport(
  file: File,
  memberId?: string,
  title?: string,
  kind: "report" | "skin" = "report",
): Promise<ReportRecord> {
  const form = new FormData();
  form.append("file", file);
  if (memberId) form.append("member_id", memberId);
  if (title) form.append("title", title);
  form.append("kind", kind);
  const data = await apiFetch<{ report: ReportRecord }>("/api/v1/reports/interpret", {
    method: "POST",
    body: form,
  });
  return data.report;
}

export async function getReports(memberId?: string): Promise<ReportRecord[]> {
  const query = memberId ? `?member_id=${encodeURIComponent(memberId)}` : "";
  const data = await apiFetch<{ reports: ReportRecord[] }>(`/api/v1/reports${query}`);
  return data.reports;
}

export async function getReport(id: string): Promise<ReportRecord> {
  const data = await apiFetch<{ report: ReportRecord }>(`/api/v1/reports/${id}`);
  return data.report;
}

export async function getFollowups(status?: string): Promise<Followup[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  const data = await apiFetch<{ followups: Followup[] }>(`/api/v1/followups${query}`);
  return data.followups;
}

export async function completeFollowup(id: string): Promise<void> {
  await apiFetch(`/api/v1/followups/${id}/complete`, { method: "PATCH" });
}

export async function getTimeline(memberId?: string): Promise<TimelineEvent[]> {
  const query = memberId ? `?member_id=${encodeURIComponent(memberId)}` : "";
  const data = await apiFetch<{ timeline: TimelineEvent[] }>(`/api/v1/health/timeline${query}`);
  return data.timeline;
}

export async function getDepartments(): Promise<string[]> {
  const data = await apiFetch<{ departments: string[] }>("/api/v1/booking/departments");
  return data.departments;
}

export async function getDoctors(department: string): Promise<Doctor[]> {
  const data = await apiFetch<{ doctors: Doctor[] }>(
    `/api/v1/services/doctors?department=${encodeURIComponent(department)}`,
  );
  return data.doctors;
}

export async function getSchedules(department: string, date?: string): Promise<ScheduleSlot[]> {
  const query = new URLSearchParams({ department });
  if (date) query.set("date", date);
  const data = await apiFetch<{ slots: ScheduleSlot[] }>(`/api/v1/booking/schedules?${query.toString()}`);
  return data.slots;
}

export async function lockSlot(slotId: string): Promise<{ lock_id: string; slot_id: string; expires_in: number }> {
  const data = await apiFetch<{ lock: { lock_id: string; slot_id: string; expires_in: number } }>(
    "/api/v1/booking/lock",
    { method: "POST", body: JSON.stringify({ slot_id: slotId }) },
  );
  return data.lock;
}

export async function createOrder(lockId: string): Promise<BookingOrder> {
  const data = await apiFetch<{ order: BookingOrder }>("/api/v1/booking/orders", {
    method: "POST",
    body: JSON.stringify({ lock_id: lockId }),
  });
  return data.order;
}

export async function getOrder(orderId: string): Promise<{ order: BookingOrder; review: ReviewRequest | null }> {
  return apiFetch(`/api/v1/booking/orders/${orderId}`);
}

export async function requestPayment(
  orderId: string,
): Promise<{ review: ReviewRequest; order: BookingOrder; duplicate?: boolean }> {
  return apiFetch(`/api/v1/booking/orders/${orderId}/request-payment`, { method: "POST" });
}

export async function getVisitGuide(orderId: string): Promise<VisitGuide> {
  return apiFetch(`/api/v1/services/visit-guide?order_id=${encodeURIComponent(orderId)}`);
}

export async function track(name: string, props: Record<string, unknown> = {}, sessionId?: string): Promise<void> {
  try {
    await apiFetch("/api/v1/analytics/event", {
      method: "POST",
      body: JSON.stringify({ name, props, session_id: sessionId }),
    });
  } catch {
    // analytics must never block the UI
  }
}

export async function getRagStatus(): Promise<RagStatus> {
  return apiFetch("/api/v1/rag/status");
}

export async function getDocument(docKey: string): Promise<DocumentDetail> {
  return apiFetch(`/api/v1/rag/documents/${encodeURIComponent(docKey)}`);
}

export async function getReviews(status = "pending"): Promise<ReviewRequest[]> {
  const data = await apiFetch<{ reviews: ReviewRequest[] }>(
    `/api/v1/admin/reviews?status=${encodeURIComponent(status)}`,
  );
  return data.reviews;
}

export async function approveReview(
  reviewId: string,
): Promise<{ review: ReviewRequest; mcp_called: boolean }> {
  return apiFetch(`/api/v1/admin/reviews/${reviewId}/approve`, { method: "POST" });
}

export async function rejectReview(
  reviewId: string,
  note: string,
): Promise<{ review: ReviewRequest }> {
  return apiFetch(`/api/v1/admin/reviews/${reviewId}/reject`, {
    method: "POST",
    body: JSON.stringify({ note }),
  });
}

export async function getUsers(): Promise<AdminUser[]> {
  const data = await apiFetch<{ users: AdminUser[] }>("/api/v1/admin/users");
  return data.users;
}

export async function createUser(payload: {
  username: string;
  password: string;
  role?: "admin" | "patient";
}): Promise<AdminUser> {
  const data = await apiFetch<{ user: AdminUser }>("/api/v1/admin/users", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data.user;
}

export async function updateUser(
  userId: string,
  payload: { enabled?: boolean; password?: string; role?: "admin" | "patient" },
): Promise<AdminUser> {
  const data = await apiFetch<{ user: AdminUser }>(`/api/v1/admin/users/${userId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  return data.user;
}

export async function getAudit(): Promise<AuditEntry[]> {
  const data = await apiFetch<{ audit: AuditEntry[] }>("/api/v1/admin/audit");
  return data.audit;
}

export async function getModelCalls(): Promise<ModelCall[]> {
  const data = await apiFetch<{ calls: ModelCall[] }>("/api/v1/admin/model-calls");
  return data.calls;
}

export async function runEval(limit = 30): Promise<Record<string, unknown>> {
  const data = await apiFetch<{ metrics: Record<string, unknown> }>("/api/v1/eval/run", {
    method: "POST",
    body: JSON.stringify({ name: "manual", limit }),
  });
  return data.metrics;
}

export async function getEvalResults(): Promise<EvalResult[]> {
  const data = await apiFetch<{ results: EvalResult[] }>("/api/v1/eval/results");
  return data.results;
}

export async function getAnalytics(): Promise<{ name: string; count: number }[]> {
  const data = await apiFetch<{ events: { name: string; count: number }[] }>("/api/v1/admin/analytics");
  return data.events;
}

function parseBlock(block: string): StreamEvent | null {
  let type = "message";
  let seq = 0;
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    else if (line.startsWith("id:")) seq = Number(line.slice(3).trim()) || 0;
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  const raw = dataLines.join("\n");
  if (raw === "[DONE]") return { seq, type: "done", data: {} };
  try {
    return { seq, type, data: JSON.parse(raw) };
  } catch {
    return { seq, type: "message", data: { text: raw } };
  }
}

export async function* streamChat(payload: {
  message: string;
  session_id?: string;
  member_id?: string;
}): AsyncGenerator<StreamEvent> {
  const headers = authHeaders({ "Content-Type": "application/json" });
  const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (response.status === 401) {
    clearToken();
    throw new UnauthorizedError();
  }
  if (!response.ok || !response.body) throw new Error(`chat failed: HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index = buffer.indexOf("\n\n");
    while (index >= 0) {
      const block = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      const event = parseBlock(block);
      if (event) {
        yield event;
        if (event.type === "done") return;
      }
      index = buffer.indexOf("\n\n");
    }
  }
}
