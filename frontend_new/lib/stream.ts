import type { BookingPayload, ParsedStreamEvent, SlotItem, StreamEventType } from "@/lib/types";
import type { RuntimeSettings } from "@/lib/runtime-settings";

const DEFAULT_BACKEND_BASE = "http://127.0.0.1:8001";

export function getApiBase(): string {
  const envBase = process.env.NEXT_PUBLIC_BACKEND_BASE_URL;
  return (envBase || DEFAULT_BACKEND_BASE).replace(/\/$/, "");
}

function createRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `req-${crypto.randomUUID().replace(/-/g, "")}`;
  }
  return `req-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function normalizeSlots(candidate: unknown): SlotItem[] {
  if (Array.isArray(candidate)) {
    return candidate as SlotItem[];
  }
  if (candidate && typeof candidate === "object") {
    return [candidate as SlotItem];
  }
  return [];
}

function normalizeBooking(candidate: unknown): BookingPayload | undefined {
  if (!candidate || typeof candidate !== "object") {
    return undefined;
  }
  return candidate as BookingPayload;
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function toEvent(payload: Record<string, unknown>): ParsedStreamEvent | null {
  const typeRaw = String(payload.type || payload.event_type || payload.event || "").trim();
  if (!typeRaw) {
    return null;
  }
  const type = typeRaw as StreamEventType;

  const content = String(payload.content || payload.message || payload.text || "");
  const node = typeof payload.node === "string" ? payload.node : undefined;
  const requestId =
    typeof payload.request_id === "string"
      ? payload.request_id
      : typeof payload.requestId === "string"
        ? payload.requestId
        : undefined;
  const seq = asNumber(payload.seq);
  const stage = typeof payload.stage === "string" ? payload.stage : undefined;
  const ts = asNumber(payload.ts);
  const meta = payload.meta && typeof payload.meta === "object" ? (payload.meta as Record<string, unknown>) : {};
  for (const key of ["runtime_config_effective", "runtime_config_requested", "rewrite_fallback", "fallback_reason", "crisis_fastlane", "metrics"]) {
    if (payload[key] !== undefined && meta[key] === undefined) {
      meta[key] = payload[key];
    }
  }

  if (type === "doctor_slots") {
    const slots = normalizeSlots(payload.data ?? payload.slots ?? meta?.slots);
    return { type, content, node, requestId, seq, stage, ts, meta, slots };
  }

  if (type === "booking_preview" || type === "payment_required" || type === "booking_confirmed" || type === "booking_error") {
    const booking = normalizeBooking(payload.data ?? meta?.data);
    return { type, content, node, requestId, seq, stage, ts, meta, booking };
  }

  return { type, content, node, requestId, seq, stage, ts, meta };
}

function parseEventBlock(block: string): string[] {
  const lines = block.split("\n");
  const parts: string[] = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      parts.push(line.slice(5).trim());
    }
  }
  return parts;
}

export async function streamChat(params: {
  message: string;
  sessionId: string;
  runtimeSettings: RuntimeSettings;
  rag?: {
    top_k?: number;
    use_rerank?: boolean;
    rerank_threshold?: number;
  };
  signal?: AbortSignal;
  onEvent: (event: ParsedStreamEvent) => void;
  onDone: () => void;
  onError: (error: string) => void;
}): Promise<void> {
  const { message, sessionId, runtimeSettings, rag, signal, onEvent, onDone, onError } = params;
  const endpoint = "/api/chat";
  const requestId = createRequestId();
  const normalizedRag = rag || {
    top_k: runtimeSettings.topK,
    use_rerank: true,
    rerank_threshold: runtimeSettings.rerankThreshold,
  };
  const debugNodes = runtimeSettings.debugIncludeNodes.filter(Boolean);

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Request-ID": requestId,
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      request_id: requestId,
      rag: normalizedRag,
      debug_include_nodes: debugNodes,
      rewrite_timeout: runtimeSettings.rewriteTimeout,
      crisis_fastlane: runtimeSettings.crisisFastlane,
    }),
    signal,
    cache: "no-store",
  });

  if (!response.ok || !response.body) {
    onError(`请求失败: HTTP ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let tokenFallback = "";
  let seenFinal = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (!seenFinal && tokenFallback.trim()) {
        onEvent({ type: "final", content: tokenFallback });
      }
      onDone();
      return;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const datas = parseEventBlock(block);
      for (const data of datas) {
        if (!data) {
          continue;
        }
        if (data === "[DONE]") {
          if (!seenFinal && tokenFallback.trim()) {
            onEvent({ type: "final", content: tokenFallback });
          }
          onDone();
          return;
        }
        try {
          const payload = JSON.parse(data) as Record<string, unknown>;
          const parsed = toEvent(payload);
          if (!parsed) {
            continue;
          }
          if (parsed.type === "token") {
            tokenFallback += parsed.content;
          } else if (parsed.type === "final") {
            seenFinal = true;
          }
          onEvent(parsed);
        } catch {
          onError(`流解析失败: ${data.slice(0, 120)}`);
        }
      }
    }
  }
}
