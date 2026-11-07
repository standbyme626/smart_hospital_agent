import type {
  BookingOrder,
  Citation,
  EvalResult,
  ModelCall,
  RagStatus,
  ScheduleSlot,
  Session,
  StreamEvent,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8020";

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

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
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
}

export async function login(username: string, password: string): Promise<{ access_token: string; role: string }> {
  const result = await apiFetch<{ access_token: string; role: string }>("/api/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setToken(result.access_token);
  return result;
}

export async function getMe(): Promise<{ user: string; role: string }> {
  return apiFetch("/api/v1/auth/me");
}

export async function getSessions(): Promise<Session[]> {
  const result = await apiFetch<{ sessions: Session[] }>("/api/v1/sessions");
  return result.sessions;
}

export async function ragSearch(query: string, topK = 5): Promise<Citation[]> {
  const result = await apiFetch<{ results: Citation[] }>("/api/v1/rag/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK }),
  });
  return result.results;
}

export async function getRagStatus(): Promise<RagStatus> {
  return apiFetch("/api/v1/rag/status");
}

export async function getDepartments(): Promise<string[]> {
  const result = await apiFetch<{ departments: string[] }>("/api/v1/booking/departments");
  return result.departments;
}

export async function getSchedules(department: string, date?: string): Promise<ScheduleSlot[]> {
  const query = new URLSearchParams({ department });
  if (date) query.set("date", date);
  const result = await apiFetch<{ slots: ScheduleSlot[] }>(`/api/v1/booking/schedules?${query.toString()}`);
  return result.slots;
}

export async function lockSlot(slotId: string, patientId = "P001"): Promise<{ lock_id: string; slot_id: string; expires_in: number }> {
  const result = await apiFetch<{ lock: { lock_id: string; slot_id: string; expires_in: number } }>(
    "/api/v1/booking/lock",
    { method: "POST", body: JSON.stringify({ slot_id: slotId, patient_id: patientId }) },
  );
  return result.lock;
}

export async function createOrder(lockId: string, patientId = "P001"): Promise<BookingOrder> {
  const result = await apiFetch<{ order: BookingOrder }>("/api/v1/booking/orders", {
    method: "POST",
    body: JSON.stringify({ lock_id: lockId, patient_id: patientId }),
  });
  return result.order;
}

export async function payOrder(orderId: string, confirmToken: string): Promise<BookingOrder> {
  const result = await apiFetch<{ order: BookingOrder }>(`/api/v1/booking/orders/${orderId}/pay`, {
    method: "POST",
    body: JSON.stringify({ confirm_token: confirmToken, method: "mock" }),
  });
  return result.order;
}

export async function getOrder(orderId: string): Promise<BookingOrder> {
  const result = await apiFetch<{ order: BookingOrder }>(`/api/v1/booking/orders/${orderId}`);
  return result.order;
}

export async function cancelOrder(orderId: string): Promise<BookingOrder> {
  const result = await apiFetch<{ order: BookingOrder }>(`/api/v1/booking/orders/${orderId}/cancel`, {
    method: "POST",
  });
  return result.order;
}

export async function getModelCalls(): Promise<ModelCall[]> {
  const result = await apiFetch<{ calls: ModelCall[] }>("/api/v1/admin/model-calls");
  return result.calls;
}

export async function runEval(limit = 50): Promise<Record<string, unknown>> {
  const result = await apiFetch<{ metrics: Record<string, unknown> }>("/api/v1/eval/run", {
    method: "POST",
    body: JSON.stringify({ name: "manual", limit }),
  });
  return result.metrics;
}

export async function getEvalResults(): Promise<EvalResult[]> {
  const result = await apiFetch<{ results: EvalResult[] }>("/api/v1/eval/results");
  return result.results;
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
}): AsyncGenerator<StreamEvent> {
  const token = getToken();
  const headers = new Headers({ "Content-Type": "application/json" });
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (response.status === 401) {
    clearToken();
    throw new UnauthorizedError();
  }
  if (!response.ok || !response.body) {
    throw new Error(`chat failed: HTTP ${response.status}`);
  }
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
