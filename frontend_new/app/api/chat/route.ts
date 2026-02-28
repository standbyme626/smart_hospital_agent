import { randomUUID } from "crypto";

const DEFAULT_BACKEND_BASE = "http://127.0.0.1:8001";
const CONNECT_TIMEOUT_MS = 8_000;
const FIRST_EVENT_TIMEOUT_MS = 20_000;
const TOTAL_TIMEOUT_MS = 75_000;
const PING_INTERVAL_MS = 10_000;

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type AnyMap = Record<string, unknown>;

function resolveBackendBase(): string {
  const envBase =
    process.env.BACKEND_BASE_URL ||
    process.env.NEXT_PUBLIC_BACKEND_BASE_URL ||
    DEFAULT_BACKEND_BASE;
  return envBase.replace(/\/$/, "");
}

function toRequestId(input: unknown): string {
  const candidate = String(input || "").trim();
  return candidate || `req-${randomUUID().replace(/-/g, "")}`;
}

function parseDataLines(block: string): string[] {
  return block
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .filter(Boolean);
}

function inferStage(node: string, type: string): string {
  if (["Query_Rewrite", "Quick_Triage"].includes(node)) return "rewrite";
  if (["Hybrid_Retriever", "retriever"].includes(node)) return "retrieve";
  if (["DSPy_Reasoner", "Decision_Judge"].includes(node)) return "judge";
  if (type === "token" || type === "final") return "respond";
  return "route";
}

function normalizeEvent(raw: unknown, fallbackRequestId: string, nextSeq: () => number): AnyMap | null {
  if (!raw || typeof raw !== "object") return null;
  const payload = raw as AnyMap;

  const type = String(payload.type || payload.event_type || payload.event || "").trim();
  if (!type) return null;

  const node = String(payload.node || payload.name || "").trim();
  const content = String(
    payload.content ||
      payload.message ||
      payload.text ||
      (typeof payload.payload === "object" && payload.payload ? (payload.payload as AnyMap).content : "") ||
      "",
  );
  const requestId = toRequestId(payload.request_id || payload.requestId || fallbackRequestId);
  const seqRaw = Number(payload.seq);
  const seq = Number.isFinite(seqRaw) && seqRaw > 0 ? Math.floor(seqRaw) : nextSeq();
  const tsRaw = Number(payload.ts);
  const ts = Number.isFinite(tsRaw) ? tsRaw : Date.now() / 1000;
  const stage = String(payload.stage || inferStage(node, type));

  return {
    ...payload,
    type,
    content,
    node,
    request_id: requestId,
    seq,
    stage,
    ts,
  };
}

function jsonEvent(payload: AnyMap): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function doneEvent(): string {
  return "data: [DONE]\n\n";
}

export async function POST(request: Request) {
  const encoder = new TextEncoder();
  const backendBase = resolveBackendBase();
  const upstreamAbort = new AbortController();

  request.signal.addEventListener("abort", () => {
    upstreamAbort.abort("client-aborted");
  });

  let body: AnyMap = {};
  try {
    const parsed = (await request.json()) as AnyMap;
    body = parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    body = {};
  }

  const resolvedRequestId = toRequestId(body.request_id || request.headers.get("x-request-id"));
  const upstreamPayload: AnyMap = { ...body, request_id: resolvedRequestId };
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      void (async () => {
        let closed = false;
        let seq = 0;
        let firstEventMs = -1;
        let firstTokenMs = -1;
        const tokenParts: string[] = [];
        let sawFinal = false;
        const startedAt = Date.now();

        const nextSeq = () => {
          seq += 1;
          return seq;
        };

        const write = (chunk: string) => {
          if (closed) return;
          controller.enqueue(encoder.encode(chunk));
        };

        const writeEvent = (payload: AnyMap) => {
          write(jsonEvent(payload));
        };

        const closeStream = () => {
          if (closed) return;
          closed = true;
          controller.close();
        };

        const totalTimeout = setTimeout(() => upstreamAbort.abort("proxy-total-timeout"), TOTAL_TIMEOUT_MS);
        const pingInterval = setInterval(() => {
          writeEvent({
            type: "ping",
            content: "proxy_keep_alive",
            node: "next_proxy",
            request_id: resolvedRequestId,
            seq: nextSeq(),
            stage: "route",
            ts: Date.now() / 1000,
            meta: { elapsed_ms: Date.now() - startedAt },
          });
        }, PING_INTERVAL_MS);

        try {
          const connectAbort = new AbortController();
          const connectTimer = setTimeout(() => {
            connectAbort.abort("proxy-connect-timeout");
            upstreamAbort.abort("proxy-connect-timeout");
          }, CONNECT_TIMEOUT_MS);
          upstreamAbort.signal.addEventListener("abort", () => connectAbort.abort("upstream-aborted"));

          const upstream = await fetch(`${backendBase}/api/v1/chat/stream`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "text/event-stream",
              "X-Request-ID": resolvedRequestId,
            },
            body: JSON.stringify(upstreamPayload),
            signal: connectAbort.signal,
            cache: "no-store",
          });
          clearTimeout(connectTimer);

          const tConnect = Date.now() - startedAt;
          writeEvent({
            type: "status",
            content: "proxy_connected",
            node: "next_proxy",
            request_id: resolvedRequestId,
            seq: nextSeq(),
            stage: "route",
            ts: Date.now() / 1000,
            meta: {
              metrics: {
                t_connect_ms: tConnect,
              },
            },
          });

          if (!upstream.ok || !upstream.body) {
            writeEvent({
              type: "error",
              content: `upstream_http_${upstream.status}`,
              node: "next_proxy",
              request_id: resolvedRequestId,
              seq: nextSeq(),
              stage: "route",
              ts: Date.now() / 1000,
            });
            write(doneEvent());
            closeStream();
            return;
          }

          let firstEventWatchdog: NodeJS.Timeout | null = setTimeout(() => {
            upstreamAbort.abort("proxy-first-event-timeout");
          }, FIRST_EVENT_TIMEOUT_MS);

          const reader = upstream.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const blocks = buffer.split("\n\n");
            buffer = blocks.pop() || "";

            for (const block of blocks) {
              const datas = parseDataLines(block);
              for (const data of datas) {
                if (data === "[DONE]") {
                  continue;
                }
                let parsed: unknown = null;
                try {
                  parsed = JSON.parse(data);
                } catch {
                  continue;
                }

                const normalized = normalizeEvent(parsed, resolvedRequestId, nextSeq);
                if (!normalized) continue;

                if (firstEventMs < 0) {
                  firstEventMs = Date.now() - startedAt;
                  if (firstEventWatchdog) {
                    clearTimeout(firstEventWatchdog);
                    firstEventWatchdog = null;
                  }
                }
                if (normalized.type === "token") {
                  if (firstTokenMs < 0) firstTokenMs = Date.now() - startedAt;
                  tokenParts.push(String(normalized.content || ""));
                }
                if (normalized.type === "final") sawFinal = true;

                writeEvent(normalized);
              }
            }
          }

          if (firstEventWatchdog) {
            clearTimeout(firstEventWatchdog);
            firstEventWatchdog = null;
          }

          if (!sawFinal) {
            writeEvent({
              type: "final",
              content: tokenParts.join("").trim() || "proxy_stream_completed",
              node: "next_proxy",
              request_id: resolvedRequestId,
              seq: nextSeq(),
              stage: "respond",
              ts: Date.now() / 1000,
            });
          }

          writeEvent({
            type: "status",
            content: "proxy_closed",
            node: "next_proxy",
            request_id: resolvedRequestId,
            seq: nextSeq(),
            stage: "route",
            ts: Date.now() / 1000,
            meta: {
              metrics: {
                t_connect_ms: Date.now() - startedAt,
                t_first_event_ms: firstEventMs,
                t_first_token_ms: firstTokenMs,
                t_final_ms: Date.now() - startedAt,
              },
            },
          });
          write(doneEvent());
          closeStream();
        } catch (error) {
          writeEvent({
            type: "error",
            content: String(error instanceof Error ? error.message : error || "proxy_error"),
            node: "next_proxy",
            request_id: resolvedRequestId,
            seq: nextSeq(),
            stage: "route",
            ts: Date.now() / 1000,
          });
          write(doneEvent());
          closeStream();
        } finally {
          clearTimeout(totalTimeout);
          clearInterval(pingInterval);
        }
      })();
    },
    cancel() {
      upstreamAbort.abort("client-cancel");
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
      "X-Request-ID": resolvedRequestId,
    },
  });
}
